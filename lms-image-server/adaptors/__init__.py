"""
Model adapter registry for the lms-image-server.

Resolution order (first match wins):
  GGUFZImageAdapter  — Z-Image-Turbo GGUF
  GGUFFluxAdapter    — FLUX.1-schnell / FLUX.1-dev GGUF
  GGUFSD3Adapter     — Stable Diffusion 3 GGUF
  MLXFluxAdapter     — FLUX MLX directories (Apple Silicon)
  GGUFGenericAdapter — everything else (SDXL, SD1.5, …) — must be last
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
from .gguf_zimage_adapter import GGUFZImageAdapter
from .gguf_flux_adapter import GGUFFluxAdapter
from .gguf_sd3_adapter import GGUFSD3Adapter
from .mlx_flux_adapter import MLXFluxAdapter
from .gguf_generic_adapter import GGUFGenericAdapter

_REGISTRY: list[ModelAdapter] = [
    GGUFZImageAdapter(),
    GGUFFluxAdapter(),
    GGUFSD3Adapter(),
    MLXFluxAdapter(),
    GGUFGenericAdapter(),   # catch-all — must be last
]


def get_adapter(model_path: str) -> ModelAdapter:
    for adapter in _REGISTRY:
        if adapter.matches(model_path):
            return adapter
    return _REGISTRY[-1]


__all__ = [
    "DEVICE", "DTYPE",
    "_effective_steps", "_effective_guidance",
    "_decode_image", "_encode_image", "_progress",
    "ModelAdapter",
    "GGUFZImageAdapter", "GGUFFluxAdapter", "GGUFSD3Adapter",
    "MLXFluxAdapter", "GGUFGenericAdapter",
    "get_adapter",
]
