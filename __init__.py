from .image_preprocessor import ImageCrop
from .ip_adapter import IPAdapter, FooocusIPAdapter

NODE_CLASS_MAPPINGS = {
    "IPAdapter": IPAdapter,
    "FooocusIPAdapter": FooocusIPAdapter,
    "ImageCrop": ImageCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapter": "Load IPAdapter",
    "FooocusIPAdapter": "Fooocus IPAdapter",
    "ImageCrop": "furusu Image Crop",
}

__all__ = [NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS]