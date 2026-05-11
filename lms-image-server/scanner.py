"""
Scans ~/.lmstudio/models for image-generation models.

Returns a flat list of ModelEntry dicts:
  key          — full absolute path (passed back as model ID to /generate)
  relative_key — path relative to the models root (display purposes)
  display_name — human-readable name
  format       — 'gguf' | 'mlx'
  size_bytes   — total file size

Heuristics
----------
GGUF files whose stem matches IMAGE_KEYWORDS are included.
GGUF files whose stem matches TEXT_KEYWORDS (LLMs) are excluded.
MLX directories whose name matches IMAGE_KEYWORDS and that contain a
`model_index.json` (diffusers format) are included.
Plain MLX text-LLM directories (no model_index.json) are excluded.
"""

import os
from pathlib import Path

LMS_MODELS_DIR = Path.home() / ".lmstudio" / "models"

# Lower-case substrings that indicate an image-generation model
IMAGE_KEYWORDS = {
    "flux", "stable-diffusion", "sdxl", "sd3", "sd-xl",
    "z-image", "zimage", "image-turbo", "kandinsky",
    "playgroundai", "wuerstchen", "pixart", "kolors",
    "auraflow", "hunyuan", "lumina", "cogview",
}

# Lower-case substrings that indicate a text / embedding LLM (skip these)
TEXT_KEYWORDS = {
    "llama", "mistral", "gemma", "phi", "falcon", "vicuna",
    "qwen", "deepseek", "yi-", "openchat", "wizard",
    "orca", "hermes", "neural-chat", "starling",
    "mamba", "rwkv", "embed", "e5-", "bge-", "gte-",
    "nomic-embed", "sentence-t5",
}


def _looks_like_image_model(name: str) -> bool:
    lower = name.lower()
    if any(k in lower for k in TEXT_KEYWORDS):
        return False
    if any(k in lower for k in IMAGE_KEYWORDS):
        return True
    return False


def scan_models(models_dir: str | None = None) -> list[dict]:
    root = Path(models_dir or LMS_MODELS_DIR)
    results: list[dict] = []

    if not root.exists():
        return results

    # ── GGUF files ─────────────────────────────────────────────────────────────
    for path in sorted(root.rglob("*.gguf")):
        if not _looks_like_image_model(path.stem):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        rel = path.relative_to(root)
        results.append({
            "key":          str(path),
            "relative_key": str(rel),
            "display_name": f"{rel.parent.name}/{path.stem}",
            "format":       "gguf",
            "size_bytes":   size,
        })

    # ── MLX model directories (diffusers-style: contain model_index.json) ─────
    for index_file in sorted(root.rglob("model_index.json")):
        model_dir = index_file.parent
        if not _looks_like_image_model(model_dir.name):
            continue
        rel = model_dir.relative_to(root)
        size = sum(
            p.stat().st_size
            for p in model_dir.rglob("*")
            if p.is_file()
        )
        results.append({
            "key":          str(model_dir),
            "relative_key": str(rel),
            "display_name": str(rel),
            "format":       "mlx",
            "size_bytes":   size,
        })

    return results
