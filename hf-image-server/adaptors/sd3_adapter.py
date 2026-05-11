"""Adapter for stabilityai/stable-diffusion-3.5-large."""

import torch
from diffusers import StableDiffusion3Pipeline

from .model_adapter import ModelAdapter


class SD3Adapter(ModelAdapter):
    """Adapter for Stable Diffusion 3.5 Large (text-to-image)."""

    supports_i2i = False

    def __init__(self):
        self._pipe = None
        self._loaded_model = None

    def matches(self, model_id: str) -> bool:
        return "stable-diffusion-3" in model_id.lower()

    def load(self, model_id: str) -> None:
        if self._loaded_model == model_id:
            return
        self._pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        self._pipe.enable_model_cpu_offload()
        self._loaded_model = model_id

    def text_to_image(self, prompt: str, model_id: str, **kwargs):
        self.load(model_id)
        steps = kwargs.get("num_inference_steps") or 28
        guidance = kwargs.get("guidance_scale")
        if guidance is None or guidance == 0.0:
            guidance = 3.5
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
            max_sequence_length=512,
            generator=generator,
        )
        return result.images[0]

    def image_to_image(self, prompt: str, model_id: str, ref_image, **kwargs):
        raise NotImplementedError("SD 3.5 Large uses text-to-image pipeline only")

    def capabilities(self) -> dict:
        return {
            "supports_i2i": self.supports_i2i,
            "default_steps": 28,
            "default_guidance": 3.5,
        }
