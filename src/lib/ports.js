// Dynamic port constants — in Tauri, Rust injects window.__DSX_PORTS__ before the page loads.
// In browser dev mode the variable is undefined and we fall back to the default dev ports.
const p = (typeof window !== 'undefined' && window.__DSX_PORTS__) ?? {}

export const BRIDGE   = `http://127.0.0.1:${p.bridge   ?? 3001}`
export const OLLAMA   = `http://127.0.0.1:${p.ollama   ?? 11434}`
export const LMSTUDIO = `http://127.0.0.1:${p.lmstudio ?? 1234}`
export const LLAMACPP = `http://127.0.0.1:${p.llamacpp ?? 8080}`
