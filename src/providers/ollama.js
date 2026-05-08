import { BRIDGE } from '../lib/ports.js'

export async function generate(prompt, model, lastImageUrl = null, width = 512, height = 512, signal = null, referenceImageUrl = null, ollamaBaseUrl = null) {
  const refImage = referenceImageUrl

  const res = await fetch(`${BRIDGE}/api/ollama/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt,
      model: model || 'stable-diffusion',
      width,
      height,
      referenceImageUrl: refImage ?? null,
      ...(ollamaBaseUrl ? { ollamaBaseUrl } : {}),
    }),
    signal,
  })

  const data = await res.json()
  if (!res.ok) throw new Error(data.error ?? `Ollama bridge error: ${res.status}`)
  if (!data.imageUrl) throw new Error('Ollama returned no image data')

  const imageUrl = data.imageUrl.startsWith('/') ? `${BRIDGE}${data.imageUrl}` : data.imageUrl
  return { imageUrl, meta: data.meta ?? null }
}
