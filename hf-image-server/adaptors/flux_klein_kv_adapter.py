"""
FLUX.2 Klein KV-Cache adapter for the image generation server.

Supports:
  black-forest-labs/FLUX.2-klein-9b-kv — KV-cache optimised variant of Klein 9B.
                                          Reference image KV pairs are computed
                                          once at step 0 and reused for steps 1-3,
                                          giving up to 2.5× speed-up on i2i edits.
                                          Same unified t2i + i2i architecture as Klein.
                                          4 steps, guidance_scale=1.0, bfloat16,
                                          non-commercial license.
"""

import torch
from PIL import Image as PILImage

from .model_adapter import ModelAdapter
from .utils import _apply_pipeline_optimizations, _call_with_progress, _effective_steps, _effective_guidance, DEVICE


class FluxKleinKVAdapter(ModelAdapter):
    """
    Adapter for the black-forest-labs FLUX.2 Klein 9B-KV family.

    Identical call signature to FluxKleinAdapter; the KV-cache optimisation is
    transparent — the pipeline handles it internally when `image=` is supplied.
    """

    default_steps = 4
    default_guidance = 1.0
    supports_i2i = True

    @classmethod
    def matches(cls, model_id: str) -> bool:
        lower = model_id.lower()
        return "klein" in lower and "kv" in lower

    def load(self, model_id: str) -> tuple:
        from diffusers import Flux2KleinKVPipeline

        pipe = Flux2KleinKVPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

        if DEVICE in ("cuda", "mps"):
            # ~29 GB in bfloat16 — cpu_offload prevents OOM on both CUDA and MPS
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(DEVICE)

        _apply_pipeline_optimizations(pipe)
        return pipe, pipe  # unified t2i / i2i pipeline

    def text_to_image(self, pipes: tuple, req) -> PILImage.Image:
        pipe, _ = pipes
        generator = torch.Generator("cpu").manual_seed(req.seed) if req.seed is not None else None

        return _call_with_progress(
            pipe,
            prompt=req.prompt,
            height=req.height,
            width=req.width,
            num_inference_steps=_effective_steps(req.num_inference_steps, self.default_steps),
            guidance_scale=_effective_guidance(req.guidance_scale, self.default_guidance),
            generator=generator,
        ).images[0]

    def image_to_image(self, pipes: tuple, req, ref_img: PILImage.Image) -> PILImage.Image:
        pipe, _ = pipes

        # Reference image must match target dimensions for the KV encoder.
        if ref_img.size != (req.width, req.height):
            ref_img = ref_img.resize((req.width, req.height), PILImage.LANCZOS)

        generator = torch.Generator("cpu").manual_seed(req.seed) if req.seed is not None else None

        # Passing image= activates KV-cache mode: ref KVs computed at step 0,
        # reused at steps 1-3 for up to 2.5× faster multi-reference editing.
        return _call_with_progress(
            pipe,
            prompt=req.prompt,
            image=ref_img,
            height=req.height,
            width=req.width,
            num_inference_steps=_effective_steps(req.num_inference_steps, self.default_steps),
            guidance_scale=_effective_guidance(req.guidance_scale, self.default_guidance),
            generator=generator,
        ).images[0]
