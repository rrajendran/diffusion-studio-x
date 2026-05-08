import { BRIDGE } from '../lib/ports.js'

export async function generate(prompt, model, _lastImageUrl, width = 512, height = 512, signal = null, referenceImageUrl = null) {
  const res = await fetch(`${BRIDGE}/api/lmstudio/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, model, width, height, referenceImageUrl }),
    signal,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(`LMStudio error: ${res.status} ${err.error ?? res.statusText}`)
  }
  const data = await res.json()
  if (!data.imageUrl) throw new Error('LMStudio returned no image data')
  return { imageUrl: data.imageUrl }
}
