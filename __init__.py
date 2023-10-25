from .image_preprocessor import ImageCrop
from .ip_adapter import IPAdapter

NODE_CLASS_MAPPINGS = {
    "IPAdapter": IPAdapter,
    "ImageCrop": ImageCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapter": "Load IPAdapter",
    "ImageCrop": "furusu Image Crop",
}

__all__ = [NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS]