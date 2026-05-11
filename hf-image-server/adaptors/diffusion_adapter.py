"""
Generic diffusion adapter for the image generation server.
"""

from PIL import Image as PILImage

from .model_adapter import ModelAdapter
from .utils import _apply_pipeline_optimizations, _call_with_progress, DTYPE, DEVICE


class DiffusionAdapter(ModelAdapter):
    """
    Fallback for any model loadable via DiffusionPipeline (e.g. turbo models).
    Supports text-to-image only; passes through num_inference_steps as-is so
    turbo models (which need exactly 4 steps) work correctly.
    """

    default_steps = 4
    default_guidance = 0.0
    supports_i2i = False

    @classmethod
    def matches(cls, model_id: str) -> bool:
        return True  # catch-all — must be last in registry

    def load(self, model_id: str) -> tuple:
        from diffusers import DiffusionPipeline

        load_kwargs = {"torch_dtype": DTYPE}
        if DEVICE != "cuda":
            # Meta-tensor allocation (low_cpu_mem_usage default) requires
            # to_empty() instead of to() and fails on MPS/CPU — disable it.
            load_kwargs["low_cpu_mem_usage"] = False

        pipe = DiffusionPipeline.from_pretrained(model_id, **load_kwargs).to(DEVICE)
        _apply_pipeline_optimizations(pipe)
        return pipe, None

    def text_to_image(self, pipes: tuple, req) -> PILImage.Image:
        pipe, _ = pipes
        return _call_with_progress(
            pipe,
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
        ).images[0]