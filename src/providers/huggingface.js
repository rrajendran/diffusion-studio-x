const BASE_URL = 'http://localhost:3001'

export async function generate(prompt, model, _apiKey, lastImageUrl, width = 512, height = 512, signal = null, referenceImageUrl = null) {
  const body = { prompt, model, width, height }
  const refImage = referenceImageUrl || lastImageUrl
  const mode = refImage ? (referenceImageUrl ? 'i2i(attached)' : 'i2i(lastImg)') : 't2i'
  if (refImage) body.reference_image = refImage

  console.log(`[huggingface] request | model=${model} ${width}x${height} mode=${mode}`)

  const t0 = performance.now()
  const res = await fetch(`${BASE_URL}/api/image/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })

  const data = await res.json()
  const elapsed = ((performance.now() - t0) / 1000).toFixed(1)

  if (!res.ok) {
    console.error(`[huggingface] error | status=${res.status} elapsed=${elapsed}s error="${data.error ?? res.statusText}"`)
    throw new Error(`Image server error: ${res.status} — ${data.error ?? res.statusText}`)
  }
  if (!data.imageUrl) throw new Error('Image server: no image in response')

  console.log(`[huggingface] success | mode=${data.mode ?? mode} elapsed=${data.elapsed ?? elapsed}s`)
  return { imageUrl: data.imageUrl }
}
