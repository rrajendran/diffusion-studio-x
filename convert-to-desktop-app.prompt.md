# Convert Diffusion Studio X to a macOS Desktop App (Tauri 2)

## Context

This is a **Vite + React 18 + Tailwind CSS v3** web application with **no TypeScript** (plain JS).

The current process model is:

| Process | Description |
|---------|-------------|
| `vite` (port 5173) | React frontend |
| `server.js` (port 3001) | Express bridge — proxies to image server, saves output images, lists HF cached models, runs Ollama CLI |
| `image_server.py` (port 8001) | FastAPI Python server — loads diffusion models via `diffusers`, runs inference, returns JPEG base64 |

All inter-process communication is plain HTTP (`fetch` in the browser → localhost Express → localhost FastAPI).

Key source files:
- `src/App.jsx` — root layout
- `src/hooks/useChat.js` — all chat/generation state
- `src/providers/huggingface.js` — calls `http://localhost:3001/api/image/generate`
- `src/providers/index.js` — provider router
- `src/components/SettingsPage.jsx` — provider/model/seed/steps/CFG settings
- `server.js` — Express bridge (Node.js, ESM)
- `image_server.py` — FastAPI diffusion server
- `package.json` — `npm run dev` starts all three with `concurrently`

## Goal

Convert this into a **macOS native desktop application** using **Tauri 2** (Rust + WebView2/WKWebView), keeping the existing React/Vite frontend completely unchanged.

The two backend processes (`server.js` and `image_server.py`) should be **managed as sidecar processes** launched and supervised by the Tauri Rust shell — not required to be running separately by the user.

---

## Requirements

### 1. Tauri 2 scaffold

- Run `npm create tauri-app@latest` or manually install `@tauri-apps/cli@^2` and `@tauri-apps/api@^2`.
- Add Tauri Vite plugin (`@tauri-apps/vite-plugin`) to `vite.config.js`.
- Place Rust source under `src-tauri/`.
- Target: `aarch64-apple-darwin` (Apple Silicon) with a universal binary option (`x86_64-apple-darwin` + `aarch64-apple-darwin`).

### 2. Sidecar: Node bridge (`server.js`)

- Bundle `server.js` and its dependencies (Express, cors) using `pkg` or `esbuild` into a standalone binary, or use Tauri's `sidecar` feature with the Node runtime.
- Preferred approach: **esbuild** bundle → single `server` binary included as a Tauri sidecar.
- The Rust main process should `spawn` the bridge on startup and kill it on exit.
- The bridge must still listen on `127.0.0.1:3001`.

### 3. Sidecar: Python image server (`image_server.py`)

- Bundle with **PyInstaller** (`pyinstaller --onedir image_server.py`) to produce a self-contained binary.
- Include it as a second Tauri sidecar.
- The Rust main process should spawn it on startup and kill it on exit.
- It must still listen on `127.0.0.1:8001`.
- The bundled Python environment must include: `torch`, `diffusers`, `transformers`, `fastapi`, `uvicorn`, `Pillow`.

### 4. Port management

- Before spawning sidecars, check that ports 3001 and 8001 are free; if not, either kill the occupant or pick alternate ports and pass them via env vars (`BRIDGE_PORT`, `IMAGE_SERVER_PORT`).
- Update `src/providers/huggingface.js` and `server.js` to read port from env when the Tauri env var is set.

### 5. Window & app metadata

- App name: **Diffusion Studio X**
- Bundle ID: `com.diffusionstudio.x`
- Window: `1280×800`, min `960×640`, resizable, frameless title bar with custom traffic-light buttons OR native title bar — use native for simplicity.
- Icon: use `public/logo.png` as the source; generate all required macOS icon sizes with `tauri icon`.

### 6. Menu bar

Implement a minimal native macOS menu:
- **App menu** — About, Preferences (opens Settings page), Separator, Quit
- **Edit** — Cut, Copy, Paste, Select All
- **View** — Reload, Toggle Full Screen

Wire "Preferences" to emit a Tauri event that the frontend listens for and responds by navigating to `SettingsPage`.

### 7. Auto-updater (optional, implement if time allows)

- Configure Tauri's built-in updater to check `https://github.com/YOUR_ORG/diffusion-studio-x/releases/latest/download/latest.json`.
- Sign with a self-generated key (`tauri signer generate`).

### 8. Build & packaging

Update `package.json` scripts:

```jsonc
"tauri": "tauri",
"tauri:dev": "tauri dev",         // starts Vite + Tauri shell (sidecars auto-launched)
"tauri:build": "tauri build",     // produces .app and .dmg in src-tauri/target/release/bundle/
"build:server": "esbuild server.js --bundle --platform=node --outfile=src-tauri/binaries/server-aarch64-apple-darwin",
"build:pyinstaller": "pyinstaller --onedir --distpath src-tauri/binaries image_server.py"
```

### 9. Frontend changes (minimal)

- Remove hardcoded `http://localhost:3001` from `src/providers/huggingface.js` and other fetch calls — replace with a helper that reads the bridge port from a Tauri global set at startup:

```js
// src/lib/bridgeUrl.js
import { invoke } from '@tauri-apps/api/core'

let _base = null
export async function bridgeBase() {
  if (!_base) _base = await invoke('get_bridge_url')
  return _base
}
```

- Expose a `get_bridge_url` Tauri command in Rust that returns `http://127.0.0.1:{port}`.
- All `fetch` calls in `src/providers/` and `src/components/` that use `http://localhost:3001` should be updated to `await bridgeBase() + '/api/...'`.

### 10. Permissions (`src-tauri/capabilities/default.json`)

Enable only what is needed:
- `shell:execute` — to spawn sidecars
- `http:fetch` to `http://127.0.0.1:*` — for bridge and image server communication
- `fs:read` and `fs:write` scoped to `$APPDATA/diffusion-studio-x/` — for output images

---

## Implementation order

1. Install Tauri 2 deps, scaffold `src-tauri/`, verify `tauri dev` opens the existing UI.
2. Implement `get_bridge_url` Tauri command; update frontend fetch calls.
3. Build Node sidecar binary with esbuild; register in `tauri.conf.json`; test.
4. Build Python sidecar with PyInstaller; register; test.
5. Implement sidecar lifecycle management in `src-tauri/src/main.rs`.
6. Add native menu with Preferences → Settings navigation event.
7. Set window size, icon, bundle ID.
8. Run `tauri build` and verify the `.dmg` installs and runs without any separate process setup.

---

## Constraints

- **Do not use TypeScript** — all new JS files must remain `.js` / `.jsx`.
- **Do not rewrite** the existing React components or hooks — only update `fetch` base URLs.
- **Do not add a database** — `localStorage` + IndexedDB remain the persistence layer.
- The Python environment bundled by PyInstaller must include the existing `.venv` packages; use `--paths .venv/lib/python*/site-packages` to point PyInstaller at the venv.
- Keep the Express bridge (`server.js`) as-is — do not port it to Rust.
