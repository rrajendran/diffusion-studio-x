"""
Local image generation server — port 8001.

Architecture: a ModelRegistry maps model IDs to ModelAdapter instances.
Each adapter encapsulates loading, default hyperparameters, and inference
for one model family, keeping the REST layer free of per-model branching.

Adapters live in adapters.py. Supported families (first match wins):
  GLMImageAdapter   — zai-org/GLM-Image  (text-to-image + image-to-image)
  QwenImageAdapter  — Qwen/Qwen-Image-*  (text-to-image + image-to-image)
  DiffusionAdapter  — everything else    (text-to-image only, generic fallback)

Speed optimisations applied at load time:
  - Attention slicing (all devices): reduces peak memory → fewer MPS/GPU stalls
  - VAE slicing (all devices): prevents OOM on larger resolutions
  - Progress bar disabled: eliminates tqdm I/O overhead per step
  - xformers memory-efficient attention (CUDA): faster attention kernel
  - torch.compile UNet/transformer (CUDA, opt-out via COMPILE_MODEL=0)
  - MPS high-watermark env override: reduces Metal memory pressure
  - Inference lock: serialises requests so they don't fight over GPU memory
  - Warmup pass (MPS/CUDA): primes Metal shader or CUDA JIT before first request
  - JPEG output (quality 90): 5-10× smaller payload vs PNG, fully transparent to clients
"""

import os
import signal
import socket
import sys

# When running as a PyInstaller one-dir bundle the executable's directory is
# not automatically on sys.path, so local packages (routes, services, adaptors,
# domain) can't be found.  Add the bundle root (_MEIPASS for onefile,
# sys.executable's dir for onedir) before any local imports.
if getattr(sys, "frozen", False):
    _bundle_dir = sys._MEIPASS if hasattr(sys, "_MEIPASS") else os.path.dirname(sys.executable)
    if _bundle_dir not in sys.path:
        sys.path.insert(0, _bundle_dir)

import uvicorn

from routes import app
from services import service


def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    print(f"[hf-image-server] Received signal {signum}, shutting down...", flush=True)
    sys.exit(0)


def check_port_available(port: int) -> bool:
    """Check if a port is available for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


DEFAULT_MODEL = os.environ.get("IMAGE_MODEL", "")

# Optional startup pre-load — if IMAGE_MODEL is set and the model is cached
# locally, load it now to warm Metal/CUDA before the first request.
# Failures are non-fatal so the server always binds port 8001.
if DEFAULT_MODEL:
    print(f"[hf-image-server] Pre-loading model: {DEFAULT_MODEL}", flush=True)
    try:
        service.get_pipes(DEFAULT_MODEL)
        print(f"[hf-image-server] Pre-load completed successfully", flush=True)
    except Exception as _e:
        print(f"[hf-image-server] startup pre-load failed: {_e}", flush=True)


if __name__ == "__main__":
    port = int(os.environ.get("IMAGE_SERVER_PORT", 8001))

    # Check if port is already in use
    if not check_port_available(port):
        print(f"[hf-image-server] ERROR: Port {port} is already in use. Another instance may be running.", flush=True)
        sys.exit(1)

    print(f"[hf-image-server] Starting server on port {port}", flush=True)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    except KeyboardInterrupt:
        print("[hf-image-server] Server stopped by user", flush=True)
    except Exception as e:
        print(f"[hf-image-server] Server error: {e}", flush=True)
        sys.exit(1)