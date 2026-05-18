# Diffusion Studio X

A chat-based AI image generation app with a glassmorphism UI, available as both a **web app** and a native **desktop app** (via Tauri). Supports Hugging Face cloud inference and local diffusion models via a Python image server.

---

## Features

- **Chat-based workflow** — generate images through a conversational interface with persistent chat history
- **Multi-provider support** — route generation to Hugging Face cloud or local diffusion models
- **Image-to-image** — pass the previous image as edit context for iterative refinement
- **Reference image** — attach a reference image to guide generation
- **Aspect ratio control** — choose from common presets; dimensions are scaled automatically
- **Inference settings** — configurable steps, guidance scale, and seed
- **Model browser** — browse and hot-load Hugging Face diffusion models from the UI
- **Gallery** — browse and download previously generated images
- **Lightbox** — full-screen image preview
- **Desktop app** — packaged as a native macOS/Windows/Linux app via Tauri; bridge and image server run as sidecars

---

## Architecture

```
Browser / Tauri WebView
    │
    ├── React 18 + Vite + Tailwind CSS (glassmorphism UI)
    │
    ├── Express bridge server  (port 3001)  — image saving, Ollama proxy
    │
    └── Python image server   (port 8001)  — local diffusion inference
            ├── GLMImageAdapter       (zai-org/GLM-Image)
            ├── QwenImageAdapter      (Qwen/Qwen-Image-*)
            ├── FluxKleinKVAdapter    (FLUX.2-klein-9b-kv)
            ├── FluxKleinAdapter      (FLUX.2-klein-9B)
            ├── FluxKontextAdapter    (FLUX.1-Kontext-dev)
            ├── FluxAdapter           (FLUX.1-schnell, FLUX.1-dev)
            ├── SD3Adapter            (stable-diffusion-3.5-large)
            ├── ZImageAdapter         (Tongyi-MAI/Z-Image-Turbo)
            ├── ErnieImageAdapter     (baidu/ERNIE-Image, baidu/ERNIE-Image-Turbo)
            └── DiffusionAdapter      (generic diffusers fallback)
```

### Provider endpoints

| Provider | Endpoint | Auth |
|----------|----------|------|
| Hugging Face | `https://api-inference.huggingface.co/models/<model>` | Bearer token |
| Ollama | `http://localhost:3001/api/ollama/generate` (bridge proxy) | none |
| Local (Python server) | `http://localhost:8001` | none |

### Data flow

1. User types a prompt in `ChatInput` → `sendMessage` in `useChat`
2. `useChat` calls `generateImage(prompt, config)` in `src/providers/index.js`
3. Provider adapter makes a `fetch` call and returns `{ imageUrl }`
4. The bridge server saves the image to disk and returns a stable URL
5. `ChatWindow` renders the result via `ImageResult`

---

## Prerequisites

| Dependency | Version | Notes |
|-----------|---------|-------|
| Node.js | ≥ 20 | |
| Python | 3.11+ | For local image server |
| Rust + Cargo | stable | Only for Tauri desktop build |
| PyTorch | ≥ 2.3.0 | MPS (Apple Silicon), CUDA, or CPU |

---

## Supported models (local Python server)

| Adapter | Model ID(s) | Pipeline | Steps | Guidance | i2i | License |
|---------|-------------|----------|-------|----------|-----|---------|
| `GLMImageAdapter` | `zai-org/GLM-Image` | `GLMImagePipeline` | 50 | 5.0 | ✅ | Apache-2.0 |
| `QwenImageAdapter` | `Qwen/Qwen-Image-Edit-2511` | `QwenImagePipeline` | 8 | 0.0 | ✅ | Apache-2.0 |
| `FluxKleinKVAdapter` | `FLUX.2-klein-9b-kv` | `Flux2KleinKVPipeline` | 4 | 1.0 | ✅ | — |
| `FluxKleinAdapter` | `FLUX.2-klein-9B` | `Flux2KleinPipeline` | 4 | 1.0 | ✅ | — |
| `FluxKontextAdapter` | `black-forest-labs/FLUX.1-Kontext-dev` | `FluxKontextPipeline` | 28 | 2.5 | ✅ | BFLA |
| `FluxAdapter` | `black-forest-labs/FLUX.1-schnell`, `.../FLUX.1-dev` | `FluxPipeline` | 4 / 20 | 0.0 / 3.5 | ❌ | Apache / BFLA |
| `SD3Adapter` | `stabilityai/stable-diffusion-3.5-large` | `StableDiffusion3Pipeline` | 28 | 3.5 | ❌ | Stability Community |
| `ZImageAdapter` | `Tongyi-MAI/Z-Image-Turbo` | `ZImagePipeline` | 9 | 0.0 | ❌ | Apache-2.0 |
| `ErnieImageAdapter` | `baidu/ERNIE-Image`, `baidu/ERNIE-Image-Turbo` | `ErnieImagePipeline` | 50 / 8 | 4.0 / 1.0 | ❌ | Apache-2.0 |
| `DiffusionAdapter` | any `diffusers`-compatible model | `StableDiffusionXLPipeline` | 30 | 7.5 | ✅ | varies |

> All local models run via the Python image server (`hf-image-server/`) on port 8001.  
> Large models use `enable_model_cpu_offload()` — first load may take several minutes.

---

## Getting started

### 1. Install Node dependencies

```bash
npm install
```

### 2. Set up the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r hf-image-server/requirements.txt
```

### 3. Run in web mode

Starts the Express bridge, the Python image server, and the Vite dev server together:

```bash
npm run dev
```

| Service | URL |
|---------|-----|
| Web UI | http://localhost:5173 |
| Bridge server | http://localhost:3001 |
| Python image server | http://localhost:8001 |

To start services individually:

```bash
npm run dev:ui       # Vite only (port 5173)
npm run dev:server   # Express bridge only (port 3001)
npm run dev:images   # Python image server only (port 8001)
```

### 4. Run as a desktop app (Tauri)

```bash
npm run tauri:dev
```

The bridge and image server binaries must already be built before running in production mode — see [Building sidecars](#building-sidecars).

---

## Configuration

Provider settings are saved to `localStorage` under the key `img-gen-config`:

- **Provider** — `huggingface` | `ollama` | `local`
- **Model** — model ID or name (provider-specific)
- **HF API key** — required for Hugging Face
- **Inference steps** — number of diffusion steps
- **Guidance scale** — classifier-free guidance strength
- **Seed** — fixed seed or random per generation

---

## Building for production

### Web

```bash
npm run build      # outputs to dist/
npm run preview    # serve the production build locally
```

### Desktop (Tauri)

Sidecars must be built first (see below), then:

```bash
npm run tauri:build
```

---

## Building sidecars

### Express bridge (Node → native binary)

Required when `server.js` changes, and before `tauri:build`:

```bash
npm run build:server
```

Outputs (macOS):
- `src-tauri/binaries/bridge-aarch64-apple-darwin`
- `src-tauri/binaries/bridge-x86_64-apple-darwin`

### Python image server (PyInstaller bundle)

Required when the Python server code changes, and before `tauri:build`:

```bash
source .venv/bin/activate        # Windows: .venv\Scripts\activate
npm run build:pyserver
```

Outputs to `src-tauri/binaries/hf-image-server/`.

---

## CI / Releases

GitHub Actions builds and packages the Tauri desktop app for all three platforms on every version tag (`v*`):

| Platform | Runner | Artifacts |
|----------|--------|-----------|
| Linux | `ubuntu-22.04` | `.AppImage`, `.deb` |
| macOS | `macos-latest` | Universal `.dmg` |
| Windows | `windows-latest` | `.msi`, `.exe` |

Push a tag to trigger a release:

```bash
git tag v1.0.0
git push origin v1.0.0
```

A draft release is created automatically — review and publish from the GitHub Releases page.

---

## Project structure

```
src/
  App.jsx                  # Root layout
  hooks/useChat.js         # Chat state and message handling
  providers/index.js       # Provider router
  providers/*.js           # One adapter per provider
  components/
    Sidebar.jsx            # Chat list + navigation
    SettingsPage.jsx       # Provider/model/API-key config
    ChatWindow.jsx         # Message list
    ChatInput.jsx          # Prompt input bar
    ChatMessage.jsx        # Individual message bubble
    ImageResult.jsx        # Generated image display
    GalleryPage.jsx        # Saved image gallery
    ModelsPage.jsx         # HF model browser
    Lightbox.jsx           # Full-screen image preview
  config/aspectRatios.js   # Aspect ratio presets
  lib/ports.js             # Runtime port constants

hf-image-server/
  main.py                  # FastAPI server entry point
  routes.py                # REST routes
  services.py              # ModelRegistry + inference orchestration
  adaptors/
    diffusion_adapter.py   # Generic diffusers pipeline
    glm_image_adapter.py   # GLM-Image family
    qwen_image_adapter.py  # Qwen-Image family

server.js                  # Express bridge server
src-tauri/                 # Tauri desktop app (Rust)
  src/lib.rs               # Sidecar lifecycle, port management
  tauri.conf.json          # Tauri configuration
.github/workflows/
  release.yml              # Cross-platform Tauri release pipeline
```

---

## Scripts reference

| Script | Description |
|--------|-------------|
| `npm run dev` | Start all services (bridge + image server + Vite) |
| `npm run dev:ui` | Vite only |
| `npm run dev:server` | Express bridge only |
| `npm run dev:images` | Python image server only |
| `npm run build` | Production Vite build |
| `npm run preview` | Serve production build |
| `npm run lint` | ESLint |
| `npm run tauri:dev` | Desktop app (dev mode) |
| `npm run tauri:build` | Desktop app (production bundle) |
| `npm run build:server` | Compile bridge → native binary |
| `npm run build:pyserver` | Bundle Python server with PyInstaller |
| `npm run kill-ports` | Kill any processes on ports 5173, 3001, 8001 |
