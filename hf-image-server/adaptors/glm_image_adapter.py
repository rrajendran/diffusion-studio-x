"""
GLM-Image adapter for the image generation server.
"""

import torch
from PIL import Image as PILImage

from .model_adapter import ModelAdapter
from .utils import _apply_pipeline_optimizations, _call_with_progress, _effective_steps, _effective_guidance, DEVICE, DTYPE


class GLMImageAdapter(ModelAdapter):
    """Adapter for zai-org/GLM-Image (flow-matching model)."""

    default_steps = 50
    default_guidance = 5.0
    supports_i2i = True

    @classmethod
    def matches(cls, model_id: str) -> bool:
        return "GLM-Image" in model_id

    def load(self, model_id: str) -> tuple:
        from diffusers.pipelines.glm_image import GlmImagePipeline

        # GLM-Image uses an LLM-based AR transformer + DiT that requires bfloat16
        # for numerical stability. On MPS, most bfloat16 ops run natively since
        # PyTorch 2.2+; rare fallbacks to CPU via PYTORCH_ENABLE_MPS_FALLBACK are acceptable.
        dtype = torch.bfloat16 if DEVICE != "cpu" else torch.float32
        load_kwargs = {"torch_dtype": dtype}
        if DEVICE == "cuda":
            load_kwargs["device_map"] = "auto"
        else:
            # Meta-tensor allocation (low_cpu_mem_usage default) requires
            # to_empty() instead of to() and fails on MPS/CPU — disable it.
            load_kwargs["low_cpu_mem_usage"] = False

        pipe = GlmImagePipeline.from_pretrained(model_id, **load_kwargs)

        if DEVICE == "cuda":
            pass  # device_map="auto" already handled placement
        elif DEVICE == "mps":
            # GLM-Image has two large sub-models (AR transformer + DiT).
            # Loading both onto the Metal heap at once exhausts unified memory
            # and triggers an OOM SIGKILL.  enable_model_cpu_offload() keeps
            # weights on CPU and moves each component to MPS only for its
            # forward pass, reducing peak Metal allocation by ~50%.
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(DEVICE)

        _apply_pipeline_optimizations(pipe)
        return pipe, pipe  # same pipeline handles both t2i and i2i modes

    def text_to_image(self, pipes: tuple, req) -> PILImage.Image:
        pipe, _ = pipes
        return _call_with_progress(
            pipe,
            prompt=req.prompt,
            height=req.height,
            width=req.width,
            num_inference_steps=_effective_steps(req.num_inference_steps, self.default_steps),
            guidance_scale=_effective_guidance(req.guidance_scale, self.default_guidance),
        ).images[0]

    def image_to_image(self, pipes: tuple, req, ref_img: PILImage.Image) -> PILImage.Image:
        pipe, _ = pipes
        # GLM-Image encodes the reference image with position embeddings that must
        # match the target output size — mismatched sizes cause a tensor shape error.
        if ref_img.size != (req.width, req.height):
            ref_img = ref_img.resize((req.width, req.height), PILImage.LANCZOS)

        if DEVICE == "mps":
            # enable_model_cpu_offload() keeps weights on CPU but the image
            # processor places pixel_values on MPS, causing a device mismatch
            # inside vision_language_encoder.  Temporarily move the encoder to
            # MPS for the i2i call so weights and inputs are on the same device.
            pipe.vision_language_encoder.to("mps")
            try:
                result = _call_with_progress(
                    pipe,
                    prompt=req.prompt,
                    image=[ref_img],
                    height=req.height,
                    width=req.width,
                    num_inference_steps=_effective_steps(req.num_inference_steps, self.default_steps),
                    guidance_scale=_effective_guidance(req.guidance_scale, self.default_guidance),
                ).images[0]
            finally:
                pipe.vision_language_encoder.to("cpu")
                torch.mps.empty_cache()
            return result

        return _call_with_progress(
            pipe,
            prompt=req.prompt,
            image=[ref_img],
            height=req.height,
            width=req.width,
            num_inference_steps=_effective_steps(req.num_inference_steps, self.default_steps),
            guidance_scale=_effective_guidance(req.guidance_scale, self.default_guidance),
        ).images[0]