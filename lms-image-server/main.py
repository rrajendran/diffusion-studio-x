"""
LM Studio image-generation server — port 8002.

Serves GGUF and MLX diffusion models stored in ~/.lmstudio/models.
Mirrors the hf-image-server API so the same bridge endpoints work for both.

Environment variables
---------------------
LMS_IMAGE_SERVER_PORT   port to listen on (default 8002)
LMS_MODELS_DIR          override the models root (default ~/.lmstudio/models)
INFERENCE_TIMEOUT       seconds before a /generate call times out (default 1200)
IMAGE_MODEL             if set, preload this model path at startup
"""

import os
import signal
import socket
import sys

if getattr(sys, "frozen", False):
    _bundle_dir = sys._MEIPASS if hasattr(sys, "_MEIPASS") else os.path.dirname(sys.executable)
    if _bundle_dir not in sys.path:
        sys.path.insert(0, _bundle_dir)

import uvicorn
from routes import app
from services import service


def _signal_handler(signum, frame):
    print(f"[lms-image-server] signal {signum} — shutting down", flush=True)
    sys.exit(0)


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


_STARTUP_MODEL = os.environ.get("IMAGE_MODEL", "")
if _STARTUP_MODEL:
    print(f"[lms-image-server] pre-loading {_STARTUP_MODEL}", flush=True)
    try:
        service.get_pipes(_STARTUP_MODEL)
        print("[lms-image-server] pre-load complete", flush=True)
    except Exception as _e:
        print(f"[lms-image-server] pre-load failed (non-fatal): {_e}", flush=True)


if __name__ == "__main__":
    port = int(os.environ.get("LMS_IMAGE_SERVER_PORT", "8002"))

    if not _port_free(port):
        print(f"[lms-image-server] ERROR: port {port} already in use", flush=True)
        sys.exit(1)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    print(f"[lms-image-server] starting on port {port}", flush=True)
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    except KeyboardInterrupt:
        print("[lms-image-server] stopped", flush=True)
    except Exception as e:
        print(f"[lms-image-server] error: {e}", flush=True)
        sys.exit(1)
