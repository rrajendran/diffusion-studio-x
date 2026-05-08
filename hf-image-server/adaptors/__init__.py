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
    _progress,
)
from .model_adapter import ModelAdapter
from .glm_image_adapter import GLMImageAdapter
from .qwen_image_adapter import QwenImageAdapter
from .diffusion_adapter import DiffusionAdapter


# ── ModelRegistry ─────────────────────────────────────────────────────────────
_REGISTRY: list[ModelAdapter] = [
    GLMImageAdapter(),
    QwenImageAdapter(),
    DiffusionAdapter(),  # catch-all — must be last
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
    "_progress",
    "ModelAdapter",
    "GLMImageAdapter",
    "QwenImageAdapter",
    "DiffusionAdapter",
    "get_adapter",
]