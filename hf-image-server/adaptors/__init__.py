"""
Model adapter registry for the image generation server.
"""

from .utils import (
    DEVICE,
    DTYPE,
    _effective_steps,
    _effective_guidance,
    _decode_image,
    _encode_image,
    _encode_video,
    _progress,
)
from .model_adapter import ModelAdapter
from .glm_image_adapter import GLMImageAdapter
from .qwen_image_adapter import QwenImageAdapter
from .flux_adapter import FluxAdapter
from .flux_klein_adapter import FluxKleinAdapter
from .flux_klein_kv_adapter import FluxKleinKVAdapter
from .flux_kontext_adapter import FluxKontextAdapter
from .zimage_adapter import ZImageAdapter
from .ernie_image_adapter import ErnieImageAdapter
from .sd3_adapter import SD3Adapter
from .wan_adapter import WanAdapter
from .diffusion_adapter import DiffusionAdapter


# ── ModelRegistry ─────────────────────────────────────────────────────────────
_REGISTRY: list[ModelAdapter] = [
    GLMImageAdapter(),
    QwenImageAdapter(),
    FluxKleinKVAdapter(),    # most specific — before FluxKleinAdapter
    FluxKleinAdapter(),
    FluxKontextAdapter(),    # before generic FluxAdapter
    FluxAdapter(),
    SD3Adapter(),
    ZImageAdapter(),
    ErnieImageAdapter(),
    WanAdapter(),            # video models — before DiffusionAdapter catch-all
    DiffusionAdapter(),      # catch-all — must be last
]


def get_adapter(model_id: str) -> ModelAdapter:
    for adapter in _REGISTRY:
        if adapter.matches(model_id):
            return adapter
    return _REGISTRY[-1]


__all__ = [
    "DEVICE",
    "DTYPE",
    "_effective_steps",
    "_effective_guidance",
    "_decode_image",
    "_encode_image",
    "_encode_video",
    "_progress",
    "ModelAdapter",
    "GLMImageAdapter",
    "QwenImageAdapter",
    "FluxAdapter",
    "FluxKleinAdapter",
    "FluxKleinKVAdapter",
    "FluxKontextAdapter",
    "ZImageAdapter",
    "ErnieImageAdapter",
    "SD3Adapter",
    "WanAdapter",
    "DiffusionAdapter",
    "get_adapter",
]