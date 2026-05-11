"""Adapter for Tongyi-MAI/Z-Image-Turbo."""

import torch
from diffusers import ZImagePipeline

from .model_adapter import ModelAdapter


class ZImageAdapter(ModelAdapter):
    """Adapter for Z-Image-Turbo (text-to-image, 9 steps, cfg=0.0)."""

    supports_i2i = False

    def __init__(self):
        self._pipe = None
        self._loaded_model = None

    def matches(self, model_id: str) -> bool:
        lower = model_id.lower()
        return "z-image" in lower or "tongyi-mai/z-image" in lower

    def load(self, model_id: str) -> None:
        if self._loaded_model == model_id:
            return
        self._pipe = ZImagePipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        self._pipe.enable_model_cpu_offload()
        self._loaded_model = model_id

    def text_to_image(self, prompt: str, model_id: str, **kwargs):
        self.load(model_id)
        steps = kwargs.get("num_inference_steps") or 9
        guidance = kwargs.get("guidance_scale", 0.0)
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 1024)
        seed = kwargs.get("seed")
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        result = self._pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
        )
        return result.images[0]

    def image_to_image(self, prompt: str, model_id: str, ref_image, **kwargs):
        raise NotImplementedError("Z-Image-Turbo is text-to-image only")

    def capabilities(self) -> dict:
        return {
            "supports_i2i": self.supports_i2i,
            "default_steps": 9,
            "default_guidance": 0.0,
        }
