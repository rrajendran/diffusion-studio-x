# Diffusion Studio X

A chat-based AI image generation app with a glassmorphism UI, available as both a **web app** and a native **desktop app** (via Tauri). Supports multiple inference backends: Hugging Face, Ollama, LM Studio, and llama.cpp.

---

## Features

- **Chat-based workflow** — generate images through a conversational interface with persistent chat history
- **Multi-provider support** — route generation to Hugging Face, Ollama, LM Studio, or llama.cpp
- **Image-to-image** — pass the previous image as edit context for iterative refinement
- **Reference image** — attach a reference image to guide generation
- **Aspect ratio control** — choose from common presets; dimensions are scaled automatically
- **Inference settings** — configurable steps, guidance scale, and seed
- **Model browser** — browse and hot-load Hugging Face diffusion models from the UI
- **Gallery** — browse and download previously generated images
- **Lightbox** — full-screen image preview
- **Desktop app** — packaged as a native macOS app via Tauri; bridge and image server run as sidecars

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
            ├── GLMImageAdapter   (zai-org/GLM-Image)
            ├── QwenImageAdapter  (Qwen/Qwen-Image-*)
            └── DiffusionAdapter  (generic diffusers fallback)
```

### Provider endpoints

| Provider | Endpoint | Auth |
|----------|----------|------|
| Hugging Face | `https://api-inference.huggingface.co/models/<model>` | Bearer token |
| Ollama | `http://localhost:3001/api/ollama/generate` (bridge proxy) | none |
| LM Studio | `http://localhost:1234/v1/images/generations` | none |
| llama.cpp | `http://localhost:8080/completion` | none |

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
| Python | 3.13 | For local image server |
| Rust + Cargo | stable | Only for Tauri desktop build |
| PyTorch | ≥ 2.3.0 | MPS (Apple Silicon), CUDA, or CPU |

---

## Getting started

### 1. Install Node dependencies

```bash
npm install
```

### 2. Set up the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
| Image server | http://localhost:8001 |

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

Tauri automatically starts Vite (`dev:ui`) before launching the desktop window. The bridge and image server binaries must already be built (see [Building sidecars](#building-sidecars)).

---

## Configuration

Provider settings are saved to `localStorage` under the key `img-gen-config`:

- **Provider** — `huggingface` | `ollama` | `lmstudio` | `llamacpp`
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

Outputs:
- `src-tauri/binaries/bridge-aarch64-apple-darwin`
- `src-tauri/binaries/bridge-x86_64-apple-darwin`

### Python image server (PyInstaller bundle)

Required when the Python server code changes, and before `tauri:build`:

```bash
source .venv/bin/activate
npm run build:pyserver
```

Outputs to `src-tauri/binaries/hf-image-server/`.

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
