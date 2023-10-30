import torch
import os
from .resampler import Resampler

import contextlib
import comfy.model_management
from comfy.ldm.modules.attention import optimized_attention
from comfy.clip_vision import clip_preprocess

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

# attention_channels of input, output, middle
SD_V12_CHANNELS = [320] * 4 + [640] * 4 + [1280] * 4 + [1280] * 6 + [640] * 6 + [320] * 6 + [1280] * 2
SD_XL_CHANNELS = [640] * 8 + [1280] * 40 + [1280] * 60 + [640] * 12 + [1280] * 20

def get_file_list(path):
    return [file for file in os.listdir(path) if file != "put_models_here.txt"]

def set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    if key not in to["patches_replace"]["attn2"]:
        patch = CrossAttentionPatch(**patch_kwargs)
        to["patches_replace"]["attn2"][key] = patch
    else:
        to["patches_replace"]["attn2"][key].set_new_condition(**patch_kwargs)

class ImageProjModel(torch.nn.Module):
    """Projection Model"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens
    
# Cross Attention to_k, to_v for IPAdapter
class To_KV(torch.nn.Module):
    def __init__(self, cross_attention_dim):
        super().__init__()

        channels = SD_XL_CHANNELS if cross_attention_dim == 2048 else SD_V12_CHANNELS
        self.to_kvs = torch.nn.ModuleList([torch.nn.Linear(cross_attention_dim, channel, bias=False) for channel in channels])
        
    def load_state_dict(self, state_dict):
        # input -> output -> middle
        for i, key in enumerate(state_dict.keys()):
            self.to_kvs[i].weight.data = state_dict[key]
    
class IPAdapterModel(torch.nn.Module):
    def __init__(self, state_dict, plus, cross_attention_dim=768, clip_embeddings_dim=1024, clip_extra_context_tokens=4, sdxl_plus=False):
        super().__init__()
        self.plus = plus
        if self.plus:
            self.image_proj_model = Resampler(
                dim=1280 if sdxl_plus else cross_attention_dim,
                depth=4,
                dim_head=64,
                heads=20 if sdxl_plus else 12,
                num_queries=clip_extra_context_tokens,
                embedding_dim=clip_embeddings_dim,
                output_dim=cross_attention_dim,
                ff_mult=4
            )   
        else:
            self.image_proj_model = ImageProjModel(
                cross_attention_dim=cross_attention_dim,
                clip_embeddings_dim=clip_embeddings_dim,
                clip_extra_context_tokens=clip_extra_context_tokens
            )
        
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.ip_layers = To_KV(cross_attention_dim)
        self.ip_layers.load_state_dict(state_dict["ip_adapter"])
        
    @torch.inference_mode()
    def get_image_embeds(self, cond, uncond):
        image_prompt_embeds = self.image_proj_model(cond)
        uncond_image_prompt_embeds = self.image_proj_model(uncond)
        return image_prompt_embeds, uncond_image_prompt_embeds
    

class IPAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "image": ("IMAGE", ),
                "clip_vision": ("CLIP_VISION", ),
                "weight": ("FLOAT", {
                    "default": 1, 
                    "min": -1, #Minimum value
                    "max": 3, #Maximum value
                    "step": 0.05 #Slider's step
                }),
                "model_name": (get_file_list(os.path.join(CURRENT_DIR,"models")), ),
                "dtype": (["fp16", "fp32"], ),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP_VISION_OUTPUT")
    FUNCTION = "adapter"
    CATEGORY = "loaders"

    def adapter(self, model, image, clip_vision, weight, model_name, dtype, mask=None):
        device = comfy.model_management.get_torch_device()
        self.dtype = torch.float32 if dtype == "fp32" or device.type == "mps" else torch.float16
        self.weight = weight # ip_adapter scale

        ip_state_dict = torch.load(os.path.join(CURRENT_DIR, os.path.join(CURRENT_DIR, "models", model_name)), map_location="cpu")
        self.plus = "latents" in ip_state_dict["image_proj"]

        # cross_attention_dim is equal to text_encoder output
        self.cross_attention_dim = ip_state_dict["ip_adapter"]["1.to_k_ip.weight"].shape[1]

        self.sdxl = self.cross_attention_dim == 2048
        self.sdxl_plus = self.sdxl and self.plus

        # number of tokens of ip_adapter embedding
        if self.plus:
            self.clip_extra_context_tokens = ip_state_dict["image_proj"]["latents"].shape[1]
        else:
            self.clip_extra_context_tokens = ip_state_dict["image_proj"]["proj.weight"].shape[0] // self.cross_attention_dim            

        cond, uncond, outputs = self.clip_vision_encode(clip_vision, image, self.plus)
        self.clip_embeddings_dim = cond.shape[-1]
        
        self.ipadapter = IPAdapterModel(
            ip_state_dict,
            plus = self.plus,
            cross_attention_dim = self.cross_attention_dim,
            clip_embeddings_dim = self.clip_embeddings_dim,
            clip_extra_context_tokens = self.clip_extra_context_tokens,
            sdxl_plus = self.sdxl_plus
        )

        self.ipadapter.to(device, dtype=self.dtype)

        self.image_emb, self.uncond_image_emb = self.ipadapter.get_image_embeds(cond.to(device, dtype=self.dtype), uncond.to(device, dtype=self.dtype))
        self.image_emb = self.image_emb.to(device, dtype=self.dtype)
        self.uncond_image_emb = self.uncond_image_emb.to(device, dtype=self.dtype)
        # Not sure of batch size at this point.
        self.cond_uncond_image_emb = None
        
        new_model = model.clone()

        if mask is not None:
            mask = mask.squeeze().to(device)

        '''
        patch_name of sdv1-2: ("input" or "output" or "middle", block_id)
        patch_name of sdxl: ("input" or "output" or "middle", block_id, transformer_index)
        '''
        patch_kwargs = {
            "number": 0,
            "weight": self.weight,
            "ipadapter": self.ipadapter,
            "dtype": self.dtype,
            "cond": self.image_emb,
            "uncond": self.uncond_image_emb,
            "mask": mask
        }

        if not self.sdxl:
            for id in [1,2,4,5,7,8]: # id of input_blocks that have cross attention
                set_model_patch_replace(new_model, patch_kwargs, ("input", id))
                patch_kwargs["number"] += 1
            for id in [3,4,5,6,7,8,9,10,11]: # id of output_blocks that have cross attention
                set_model_patch_replace(new_model, patch_kwargs, ("output", id))
                patch_kwargs["number"] += 1
            set_model_patch_replace(new_model, patch_kwargs, ("middle", 0))
        else:
            for id in [4,5,7,8]: # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, patch_kwargs, ("input", id, index))
                    patch_kwargs["number"] += 1
            for id in range(6): # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, patch_kwargs, ("output", id, index))
                    patch_kwargs["number"] += 1
            for index in range(10):
                set_model_patch_replace(new_model, patch_kwargs, ("middle", 0, index))
                patch_kwargs["number"] += 1

        return (new_model, outputs)
    
    def clip_vision_encode(self, clip_vision, image, plus=False):

        inputs = clip_preprocess(image)
        comfy.model_management.load_model_gpu(clip_vision.patcher)
        pixel_values = inputs.to(clip_vision.load_device)

        if clip_vision.dtype != torch.float32:
            precision_scope = torch.autocast
        else:
            precision_scope = lambda a, b: contextlib.nullcontext(a)

        with precision_scope(comfy.model_management.get_autocast_device(clip_vision.load_device), torch.float32):
            outputs = clip_vision.model(pixel_values=pixel_values, output_hidden_states=True)

        if plus:
            cond = outputs.hidden_states[-2]
            with precision_scope(comfy.model_management.get_autocast_device(clip_vision.load_device), torch.float32):
                uncond = clip_vision.model(torch.zeros_like(pixel_values), output_hidden_states=True).hidden_states[-2]
        else:
            cond = outputs.image_embeds
            uncond = torch.zeros_like(cond)
        for k in outputs:
            t = outputs[k]
            if k == "hidden_states":
                outputs[k] = None
            elif t is not None:
                outputs[k] = t.cpu()
        return cond, uncond, outputs


class CrossAttentionPatch:
    # forward for patching
    def __init__(self, weight, ipadapter, dtype, number, cond, uncond, mask=None):
        self.weights = [weight]
        self.ipadapters = [ipadapter]
        self.conds = [cond]
        self.unconds = [uncond]
        self.dtype = dtype
        self.number = number
        self.masks = [mask]
    
    def set_new_condition(self, weight, ipadapter, cond, uncond, dtype, number, mask=None):
        self.weights.append(weight)
        self.ipadapters.append(ipadapter)
        self.conds.append(cond)
        self.unconds.append(uncond)
        self.masks.append(mask)
        self.dtype = dtype

    def __call__(self, n, context_attn2, value_attn2, extra_options):
        org_dtype = n.dtype
        cond_or_uncond = extra_options["cond_or_uncond"]
        original_shape = (extra_options["original_shape"][2], extra_options["original_shape"][3]) 
        with torch.autocast("cuda", dtype=self.dtype):
            q = n
            k = context_attn2
            v = value_attn2
            b, _, _ = q.shape
            batch_prompt = b // len(cond_or_uncond)
            out = optimized_attention(q, k, v, extra_options["n_heads"])

            for weight, cond, uncond, ipadapter, mask in zip(self.weights, self.conds, self.unconds, self.ipadapters, self.masks):
                uncond_cond = torch.cat([(cond.repeat(batch_prompt, 1, 1), uncond.repeat(batch_prompt, 1, 1))[i] for i in cond_or_uncond], dim=0)

                # k, v for ip_adapter
                ip_k = ipadapter.ip_layers.to_kvs[self.number*2](uncond_cond)
                ip_v = ipadapter.ip_layers.to_kvs[self.number*2+1](uncond_cond)

                # Convert ip_k and ip_v to the same dtype as q
                ip_k = ip_k.to(dtype=q.dtype)
                ip_v = ip_v.to(dtype=q.dtype)
                
                ip_out = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
                
                if mask is not None:
                    # 良い方法募集
                    if original_shape[0] * original_shape[1] == q.shape[1]:
                        down_sample_rate = 1
                    elif (original_shape[0] // 2) * (original_shape[1] // 2) == q.shape[1]:
                        down_sample_rate = 2
                    elif (original_shape[0] // 4) * (original_shape[1] // 4) == q.shape[1]:
                        down_sample_rate = 4
                    else:
                        down_sample_rate = 8
                    mask_downsample = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(original_shape[0] // down_sample_rate, original_shape[1] // down_sample_rate), mode="nearest").squeeze(0)
                    mask_downsample = mask_downsample.view(1, -1, 1).repeat(out.shape[0], 1, out.shape[2])
                    ip_out = ip_out * mask_downsample

                out = out + ip_out * weight

        return out.to(dtype=org_dtype)

