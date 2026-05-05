# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

```bash
npm install          # install dependencies
npm run dev          # start Vite (port 5173) + Ollama bridge server (port 3001) together
npm run dev:ui       # Vite only
npm run dev:server   # Ollama bridge server only
npm run build        # production build to dist/
npm run preview      # serve the production build locally
npm run lint         # ESLint check
```

## Architecture

**Stack:** Vite + React 18, Tailwind CSS v3, plain JS (no TypeScript).

### Data flow

1. User types a prompt in `ChatInput` → calls `sendMessage` from `useChat`
2. `useChat` appends a user message to state, calls `generateImage(prompt, config)`
3. `generateImage` (in `src/providers/index.js`) routes to the active provider adapter
4. Provider adapter makes a `fetch` call and returns `{ imageUrl: string }`
5. `useChat` appends a bot message (with `imageUrl`) to state
6. `ChatWindow` renders `ChatMessage` bubbles; bot messages with an image render `ImageResult`

### Key files

| Path | Role |
|------|------|
| `src/App.jsx` | Root layout — gradient background, Sidebar + chat column |
| `src/hooks/useChat.js` | All chat state (`messages`, `loading`, `sendMessage`, `clearMessages`) |
| `src/providers/index.js` | Provider router — `generateImage(prompt, config)` |
| `src/providers/*.js` | One file per provider; each exports `generate(prompt, model[, apiKey])` |
| `src/components/Sidebar.jsx` | Provider / model / API-key config, persisted to `localStorage` |
| `src/index.css` | Tailwind directives + `.glass`, `.glass-dark`, `.glass-btn` utility classes |

### Glassmorphism design tokens

All glass effects are composed from Tailwind utilities — no custom CSS variables needed:

| Token | Classes |
|-------|---------|
| Panel | `bg-white/10 backdrop-blur-md border border-white/20 rounded-2xl` |
| Button | `bg-white/20 hover:bg-white/30 backdrop-blur-sm border border-white/20 rounded-xl` |
| Input bar | `bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl` |
| Page background | `bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-800` |

The convenience classes `.glass`, `.glass-dark`, `.glass-btn` are defined in `src/index.css`.

### Provider endpoints

| Provider | Endpoint | Auth |
|----------|----------|------|
| Hugging Face | `https://api-inference.huggingface.co/models/<model>` | `Bearer` token |
| Ollama | `http://localhost:3001/api/ollama/generate` (local bridge) | none |
| LM Studio | `http://localhost:1234/v1/images/generations` | none |
| llama.cpp | `http://localhost:8080/completion` | none |

**Ollama bridge** (`server.js`): Because Ollama image-generation models are CLI-only, a small Express server (`server.js`, port 3001) runs `ollama run <model>` via `child_process`, captures stdout (raw PNG bytes or a file path), and returns a base64 data URL to the React app. Both processes start together with `npm run dev`.

Provider settings (provider name, model, HF API key) are saved to `localStorage` under the key `img-gen-config`.
