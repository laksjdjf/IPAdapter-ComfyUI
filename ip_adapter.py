import torch
import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def get_file_list(path):
    return [file for file in os.listdir(path) if file != "put_models_here.txt"]

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
        channels = [320, 320, 320, 320, 640, 640, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 640, 640, 640, 320, 320, 320, 320, 320, 320, 1280, 1280]
        self.to_kvs = torch.nn.ModuleList([torch.nn.Linear(768, channel, bias=False) for channel in channels])
        
    def load_state_dict(self, state_dict):
        for i, key in enumerate(state_dict.keys()):
            self.to_kvs[i].weight.data = state_dict[key]
    
class IPAdapterModel:
    def __init__(self, ip_ckpt):
        super().__init__()
        self.ip_ckpt = ip_ckpt
        self.device = "cuda"

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
    def get_image_embeds(self, clip_image_embeds):
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

class IPAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                "weight": ("FLOAT", {
                    "default": 1, 
                    "min": -1, #Minimum value
                    "max": 3, #Maximum value
                    "step": 0.05 #Slider's step
                }),
                "model_name": (get_file_list(os.path.join(CURRENT_DIR,"models")), ),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "adapter"
    CATEGORY = "loaders"

    def adapter(self, model, clip_vision_output, weight, model_name):
        dtype = model.model.diffusion_model.dtype
        device = "cuda"
        self.weight = weight
        self.ipadapter = IPAdapterModel(
            os.path.join(CURRENT_DIR, os.path.join(CURRENT_DIR, "models", model_name)),
        )
        self.ipadapter.ip_layers.to(device, dtype=dtype)
        clip_vision_emb = clip_vision_output.image_embeds.to(device, dtype=torch.float16)
        self.image_emb, self.uncond_image_emb = self.ipadapter.get_image_embeds(clip_vision_emb)
        self.image_emb = self.image_emb.to(device, dtype=dtype)
        self.uncond_image_emb = self.uncond_image_emb.to(device, dtype=dtype)
        self.cond_uncond_image_emb = None
        
        new_model = model.clone()
        #input
        number = 0
        for id in [1,2,4,5,7,8]:
            new_model.set_model_attn2_replace(self.patch_forward(number), "input", id)
            number += 1
        #output
        for id in [3,4,5,6,7,8,9,10,11]:
            new_model.set_model_attn2_replace(self.patch_forward(number), "output", id)
            number += 1
        #middle
        new_model.set_model_attn2_replace(self.patch_forward(number), "middle", 0)

        return (new_model,)
    
    def patch_forward(self, number):
        def forward(n, context_attn2, value_attn2, extra_options):
            q = n
            k = context_attn2
            v = value_attn2
            b, _, _ = q.shape

            if self.cond_uncond_image_emb is None or self.cond_uncond_image_emb.shape[0] != b:
                self.cond_uncond_image_emb = torch.cat([self.uncond_image_emb.repeat(b//2, 1, 1), self.image_emb.repeat(b//2, 1, 1)], dim=0)

            ip_k = self.ipadapter.ip_layers.to_kvs[number*2](self.cond_uncond_image_emb)
            ip_v = self.ipadapter.ip_layers.to_kvs[number*2+1](self.cond_uncond_image_emb)

            q, k, v, ip_k, ip_v = map(
                lambda t: t.view(b, -1, extra_options["n_heads"], extra_options["dim_head"]).transpose(1, 2),
                (q, k, v, ip_k, ip_v),
            )

            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
            out = out.transpose(1, 2).reshape(b, -1, extra_options["n_heads"] * extra_options["dim_head"])

            ip_out = torch.nn.functional.scaled_dot_product_attention(q, ip_k, ip_v, attn_mask=None, dropout_p=0.0, is_causal=False)
            ip_out = ip_out.transpose(1, 2).reshape(b, -1, extra_options["n_heads"] * extra_options["dim_head"])

            out = out + ip_out * self.weight

            return out

        return forward
        
NODE_CLASS_MAPPINGS = {
    "IPAdapter": IPAdapter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapter": "Load IPAdapter",
}
