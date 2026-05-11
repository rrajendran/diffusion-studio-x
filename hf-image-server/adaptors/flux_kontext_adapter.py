"""Adapter for black-forest-labs/FLUX.1-Kontext-dev (image editing model)."""

import torch
from diffusers import FluxKontextPipeline

from .model_adapter import ModelAdapter


class FluxKontextAdapter(ModelAdapter):
    """Adapter for FLUX.1-Kontext-dev — instruction-based image editing."""

    supports_i2i = True

    def __init__(self):
        self._pipe = None
        self._loaded_model = None

    def matches(self, model_id: str) -> bool:
        return "kontext" in model_id.lower()

    def load(self, model_id: str) -> None:
        if self._loaded_model == model_id:
            return
        self._pipe = FluxKontextPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        import torch
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        # Always use CPU offload for FLUX-class models (~24 GB)
        self._pipe.enable_model_cpu_offload()
        self._loaded_model = model_id

    def text_to_image(self, prompt: str, model_id: str, **kwargs):
        self.load(model_id)
        steps = kwargs.get("num_inference_steps") or 28
        guidance = kwargs.get("guidance_scale")
        if guidance is None or guidance == 0.0:
            guidance = 2.5
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
        self.load(model_id)
        steps = kwargs.get("num_inference_steps") or 28
        guidance = kwargs.get("guidance_scale")
        if guidance is None or guidance == 0.0:
            guidance = 2.5
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 1024)
        seed = kwargs.get("seed")
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        result = self._pipe(
            prompt=prompt,
            image=ref_image,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
        )
        return result.images[0]

    def capabilities(self) -> dict:
        return {
            "supports_i2i": self.supports_i2i,
            "default_steps": 28,
            "default_guidance": 2.5,
        }
