import { LLAMACPP } from '../lib/ports.js'

export async function generate(prompt, _model, _lastImageUrl, width = 512, height = 512, signal = null, referenceImageUrl = null, baseUrl = null) {
  // llama.cpp accepts image_data as [{data: base64string, id: 1}] for multimodal models
  const imageData = referenceImageUrl
    ? [{ data: referenceImageUrl.replace(/^data:image\/\w+;base64,/, ''), id: 1 }]
    : undefined

  const endpoint = (baseUrl || LLAMACPP).replace(/\/$/, '')
  const res = await fetch(`${endpoint}/completion`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, n_predict: 256, width, height, ...(imageData ? { image_data: imageData } : {}) }),
    signal,
  })
  if (!res.ok) throw new Error(`llama.cpp error: ${res.status} ${res.statusText}`)
  const data = await res.json()
  if (data.image) {
    return { imageUrl: `data:image/png;base64,${data.image}` }
  }
  throw new Error('llama.cpp returned no image data')
}
