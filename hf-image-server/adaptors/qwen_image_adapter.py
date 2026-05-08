"""
Qwen-Image adapter for the image generation server.
"""

import torch
from PIL import Image as PILImage

from .model_adapter import ModelAdapter
from .utils import _apply_pipeline_optimizations, _call_with_progress, _effective_steps, _effective_guidance, DEVICE, DTYPE


class QwenImageAdapter(ModelAdapter):
    """
    Adapter for Qwen/Qwen-Image-* models.
    t2i: AutoPipelineForText2Image (e.g. Qwen/Qwen-Image-2512)
    i2i: QwenImageEditPlusPipeline (e.g. Qwen/Qwen-Image-Edit-2511)
    """

    # Flow-matching based model — good results in 8 steps on MPS.
    # The LLM AR encoding pass before diffusion already takes ~4-6 min on MPS;
    # keeping steps low avoids exceeding inference timeout.
    default_steps = 8
    default_guidance = 4.0
    supports_i2i = True

    @classmethod
    def matches(cls, model_id: str) -> bool:
        return "Qwen" in model_id and "Image" in model_id

    def load(self, model_id: str) -> tuple:
        from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, QwenImageEditPlusPipeline

        # Qwen uses an LLM-based transformer that requires bfloat16 for numerical
        # stability — float16 produces NaN outputs. On MPS, most bfloat16 ops run
        # natively since PyTorch 2.2+; rare unsupported ops fall back to CPU via
        # PYTORCH_ENABLE_MPS_FALLBACK (set in utils.py) which is acceptable.
        dtype = torch.bfloat16 if DEVICE != "cpu" else torch.float32
        # MPS/CPU: low_cpu_mem_usage=False prevents meta-tensor allocation.
        # Meta tensors require .to_empty() instead of .to() and fail on non-CUDA devices.
        # On CUDA we keep the default (True) for memory efficiency.
        load_kwargs = dict(
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if DEVICE != "cuda":
            load_kwargs["low_cpu_mem_usage"] = False

        # Edit-only models (e.g. Qwen/Qwen-Image-Edit-2511) have no AutoPipeline
        # mapping for text-to-image — AutoPipelineForText2Image raises
        # "can't find a pipeline linked to QwenImageEditPlusPipeline for
        # qwenimage-edit-plus". Load the edit pipeline directly and use it for
        # both t2i (no reference image) and i2i (with reference image).
        _is_edit_only = "Edit" in model_id

        if _is_edit_only:
            print(f"[hf-image-server] edit-only model detected — loading QwenImageEditPlusPipeline directly", flush=True)
            pipe_i2i = QwenImageEditPlusPipeline.from_pretrained(model_id, **load_kwargs)
            if DEVICE == "mps":
                # Load directly to MPS — unified memory has plenty of headroom for
                # Qwen's ~14GB on a 52GB machine. The initial .to("mps") takes
                # ~4-5 min (one-time cost), but subsequent inference runs are
                # fully on-device and fast (~2-3 min).  cpu_offload was tried but
                # macOS compresses the CPU-resident weights between requests,
                # causing every run after the first to be slower than the first.
                print(f"[hf-image-server] moving Qwen pipeline to MPS (one-time ~4-5 min transfer) …", flush=True)
                pipe_i2i = pipe_i2i.to(DEVICE)
            else:
                pipe_i2i = pipe_i2i.to(DEVICE)
            _apply_pipeline_optimizations(pipe_i2i)
            # Use the edit pipeline as the t2i slot too (prompt-only calls work fine)
            return pipe_i2i, pipe_i2i

        pipe_t2i = AutoPipelineForText2Image.from_pretrained(
            model_id,
            **load_kwargs,
        )
        if DEVICE == "mps":
            print(f"[hf-image-server] moving Qwen t2i pipeline to MPS (one-time ~4-5 min transfer) …", flush=True)
            pipe_t2i = pipe_t2i.to(DEVICE)
        else:
            pipe_t2i = pipe_t2i.to(DEVICE)

        _apply_pipeline_optimizations(pipe_t2i)

        pipe_i2i = None
        try:
            pipe_i2i = QwenImageEditPlusPipeline.from_pipe(pipe_t2i)
            print(f"[hf-image-server] i2i: QwenImageEditPlusPipeline", flush=True)
        except Exception as e:
            print(f"[hf-image-server] QwenImageEditPlusPipeline unavailable ({e}), trying AutoPipelineForImage2Image", flush=True)
            try:
                pipe_i2i = AutoPipelineForImage2Image.from_pipe(pipe_t2i)
                print(f"[hf-image-server] i2i: AutoPipelineForImage2Image", flush=True)
            except Exception as e2:
                print(f"[hf-image-server] i2i pipeline unavailable for {model_id}: {e2}", flush=True)

        if pipe_i2i is not None:
            _apply_pipeline_optimizations(pipe_i2i)

        return pipe_t2i, pipe_i2i

    def text_to_image(self, pipes: tuple, req) -> PILImage.Image:
        pipe_t2i, _ = pipes
        cfg = _effective_guidance(req.guidance_scale, self.default_guidance)
        steps = _effective_steps(req.num_inference_steps, self.default_steps)
        # QwenImageEditPlusPipeline (edit-only models) does not accept
        # negative_prompt or true_cfg_scale for prompt-only calls.
        from diffusers import QwenImageEditPlusPipeline
        if isinstance(pipe_t2i, QwenImageEditPlusPipeline):
            return _call_with_progress(
                pipe_t2i,
                prompt=req.prompt,
                height=req.height,
                width=req.width,
                num_inference_steps=steps,
                guidance_scale=cfg,
            ).images[0]
        # Standard Qwen t2i pipeline uses true_cfg_scale for CFG.
        # negative_prompt="" is required to activate CFG.
        return _call_with_progress(
            pipe_t2i,
            prompt=req.prompt,
            negative_prompt="",
            height=req.height,
            width=req.width,
            num_inference_steps=steps,
            true_cfg_scale=cfg,
        ).images[0]

    def image_to_image(self, pipes: tuple, req, ref_img: PILImage.Image) -> PILImage.Image:
        _, pipe_i2i = pipes
        if pipe_i2i is None:
            raise ValueError("Qwen i2i pipeline unavailable")
        # true_cfg_scale drives CFG for Qwen i2i; values <= 1.0 disable it.
        # Use a sentinel of 1.0 (no CFG) so that guidance_scale=0.0 (the API
        # sentinel) still maps to the adapter default (4.0), while an explicit
        # guidance_scale=1.0 from the client correctly passes through.
        cfg = _effective_guidance(req.guidance_scale, self.default_guidance)
        return _call_with_progress(
            pipe_i2i,
            image=[ref_img],
            prompt=req.prompt,
            generator=torch.manual_seed(req.seed),
            true_cfg_scale=cfg,
            negative_prompt="" if cfg <= 1.0 else " ",
            guidance_scale=1.0,
            num_inference_steps=_effective_steps(req.num_inference_steps, self.default_steps),
            num_images_per_prompt=1,
            height=req.height,
            width=req.width,
        ).images[0]