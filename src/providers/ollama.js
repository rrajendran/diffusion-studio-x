// Calls the local bridge server (server.js) which runs `ollama run <model>` via CLI.
// referenceImageUrl is forwarded so the bridge can write it as a temp file for multimodal models.
export async function generate(prompt, model, lastImageUrl = null, width = 512, height = 512, signal = null, referenceImageUrl = null) {
  const res = await fetch('http://localhost:3001/api/ollama/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, model: model || 'stable-diffusion', lastImageUrl, width, height, referenceImageUrl }),
    signal,
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.error ?? `Bridge error: ${res.status}`)
  return { imageUrl: data.imageUrl }
}
