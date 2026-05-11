"""Adapter for baidu/ERNIE-Image and baidu/ERNIE-Image-Turbo."""

import torch
from diffusers import ErnieImagePipeline

from .model_adapter import ModelAdapter


class ErnieImageAdapter(ModelAdapter):
    """Adapter for ERNIE-Image family (t2i).

    - ERNIE-Image:       50 steps, guidance_scale=4.0
    - ERNIE-Image-Turbo:  8 steps, guidance_scale=1.0
    """

    supports_i2i = False

    def __init__(self):
        self._pipe = None
        self._loaded_model = None

    def matches(self, model_id: str) -> bool:
        lower = model_id.lower()
        return "ernie-image" in lower or "baidu/ernie" in lower

    def _is_turbo(self, model_id: str) -> bool:
        return "turbo" in model_id.lower()

    def load(self, model_id: str) -> None:
        if self._loaded_model == model_id:
            return
        self._pipe = ErnieImagePipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        self._pipe.enable_model_cpu_offload()
        self._loaded_model = model_id

    def text_to_image(self, prompt: str, model_id: str, **kwargs):
        self.load(model_id)
        turbo = self._is_turbo(model_id)
        default_steps = 8 if turbo else 50
        default_guidance = 1.0 if turbo else 4.0
        steps = kwargs.get("num_inference_steps") or default_steps
        guidance = kwargs.get("guidance_scale")
        if guidance is None or guidance == 0.0:
            guidance = default_guidance
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
            use_pe=True,
        )
        return result.images[0]

    def image_to_image(self, prompt: str, model_id: str, ref_image, **kwargs):
        raise NotImplementedError("ERNIE-Image is text-to-image only")

    def capabilities(self) -> dict:
        return {
            "supports_i2i": self.supports_i2i,
            "default_steps": 50,
            "default_guidance": 4.0,
        }
