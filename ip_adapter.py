from typing import Any, Mapping
import torch
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image
import os
import copy
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

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
    
class To_KV(torch.nn.Module):
    def __init__(self):
        super().__init__()
        channels = [320, 320, 320, 320, 640, 640, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 640, 640, 640, 320, 320, 320, 320, 320, 320]
        self.to_kvs = torch.nn.ModuleList([torch.nn.Linear(768, channel, bias=False) for channel in channels])
        
    def load_state_dict(self, state_dict):
        indice = list(range(0,12)) # down
        indice.extend(list(range(14, 32))) # up
        indice.extend([12,13]) # down
        for i, key in enumerate(state_dict.keys()):
            self.to_kvs[indice[i]].weight.data = state_dict[key]
    
class IPAdapterModel:
    def __init__(self, image_encoder_path, ip_ckpt):
        super().__init__()
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.device = "cuda"

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to("cuda", dtype=torch.float16)
        self.clip_image_processor = CLIPImageProcessor()
        self.image_proj_model = ImageProjModel(cross_attention_dim=768, clip_embeddings_dim=1024,
                clip_extra_context_tokens=4).to("cuda", dtype=torch.float16)
        
        self.load_ip_adapter()

    def load_ip_adapter(self):
        state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.ip_layers = To_KV()
        self.ip_layers.load_state_dict(state_dict["ip_adapter"])
        self.ip_layers.to("cuda")
        
    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

class IPAdapter:

    def __init__(self):
        self.ipadapter = IPAdapterModel(
            os.path.join(CURRENT_DIR,"IP-Adapter/models/image_encoder"),
            os.path.join(CURRENT_DIR,"IP-Adapter/models/ip-adapter_sd15.bin")
        )
        print(self.ipadapter)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", {
                    "default": 1, 
                    "min": -1, #Minimum value
                    "max": 3, #Maximum value
                    "step": 0.05 #Slider's step
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "adapter"
    CATEGORY = "loaders"

    def adapter(self, model, image, weight):
        dtype = model.model.diffusion_model.dtype
        device = "cuda"
        self.weight = weight
        self.ipadapter.ip_layers.to(device, dtype=dtype)

        tensor = image*255
        tensor = np.array(tensor, dtype=np.uint8)
        image = Image.fromarray(tensor[0])

        self.image_emb, self.uncond_image_emb = self.ipadapter.get_image_embeds(image)
        self.image_emb = self.image_emb.to(device, dtype=dtype)
        self.uncond_image_emb = self.uncond_image_emb.to(device, dtype=dtype)
        self.cond_uncond_image_emb = None
        new_model = copy.deepcopy(model)
        self.hook_forwards(new_model.model.diffusion_model)
        
        return (new_model,)
    
    def hook_forwards(self, root_module: torch.nn.Module):
        i = 0
        for name, module in root_module.named_modules():
            if "attn2" in name and "CrossAttention" in module.__class__.__name__:
                module.forward = self.hook_forward(module, i)
                i += 1
    
    def hook_forward(self, module, i):
        def forward(x, context=None, value=None, mask=None):
            q = module.to_q(x)
            k = module.to_k(context)
            v = module.to_v(context)
            b, _, _ = q.shape

            if self.cond_uncond_image_emb is None or self.cond_uncond_image_emb.shape[0] != b:
                self.cond_uncond_image_emb = torch.cat([self.uncond_image_emb.repeat(b//2, 1, 1), self.image_emb.repeat(b//2, 1, 1)], dim=0)

            ip_k = self.ipadapter.ip_layers.to_kvs[i*2](self.cond_uncond_image_emb)
            ip_v = self.ipadapter.ip_layers.to_kvs[i*2+1](self.cond_uncond_image_emb)

            q, k, v, ip_k, ip_v = map(
                lambda t: t.view(b, -1, module.heads, module.dim_head).transpose(1, 2),
                (q, k, v, ip_k, ip_v),
            )

            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
            out = out.transpose(1, 2).reshape(b, -1, module.heads * module.dim_head)

            ip_out = torch.nn.functional.scaled_dot_product_attention(q, ip_k, ip_v, attn_mask=None, dropout_p=0.0, is_causal=False)
            ip_out = ip_out.transpose(1, 2).reshape(b, -1, module.heads * module.dim_head)

            out = out + ip_out * self.weight

            return module.to_out(out)

        return forward
        
NODE_CLASS_MAPPINGS = {
    "IPAdapter": IPAdapter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapter": "Load IPAdapter",
}
