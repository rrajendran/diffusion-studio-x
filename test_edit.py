"""Quick test for image editing via the Python image server."""
import base64
import json
import sys
import time
import urllib.request

OUTPUT_DIR = "output"

import os
imgs = [f for f in os.listdir(OUTPUT_DIR) if f.endswith((".jpg", ".png"))]
if not imgs:
    print("No output images found to use as reference")
    sys.exit(1)

img_path = os.path.join(OUTPUT_DIR, imgs[0])
with open(img_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
data_url = f"data:image/jpeg;base64,{b64}"

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen-Image-Edit-2511"
prompt = sys.argv[2] if len(sys.argv) > 2 else "make the sky bright orange at sunset"

print(f"[test] model={model} ref={img_path} ({len(b64)} chars)")
print(f"[test] prompt={prompt!r}")

body = json.dumps({
    "prompt": prompt,
    "model": model,
    "width": 512,
    "height": 512,
    "reference_image": data_url,
}).encode()

t0 = time.time()
req = urllib.request.Request(
    "http://localhost:8001/generate",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=1400) as resp:
        result = json.loads(resp.read())
        elapsed = time.time() - t0
        if "imageUrl" in result:
            print(f"[test] SUCCESS in {elapsed:.1f}s | mode={result.get('mode')} imageUrl_len={len(result['imageUrl'])}")
        else:
            print(f"[test] ERROR in {elapsed:.1f}s | {result}")
except urllib.error.HTTPError as e:
    body_text = e.read().decode()
    elapsed = time.time() - t0
    print(f"[test] HTTP {e.code} in {elapsed:.1f}s | {body_text}")
except Exception as e:
    elapsed = time.time() - t0
    print(f"[test] EXCEPTION in {elapsed:.1f}s | {type(e).__name__}: {e}")
