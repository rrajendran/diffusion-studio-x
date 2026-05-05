const BASE_URL = 'http://localhost:3001'

export async function generate(prompt, model, _apiKey, lastImageUrl, signal = null, referenceImageUrl = null, genParams = {}) {
  const { width = 512, height = 512, inferenceSteps, guidanceScale, seed } = genParams
  const body = { prompt, model, width, height }
  if (inferenceSteps != null) body.num_inference_steps = inferenceSteps
  if (guidanceScale  != null) body.guidance_scale = guidanceScale
  if (seed           != null) body.seed = seed
  const refImage = referenceImageUrl || lastImageUrl
  const mode = refImage ? (referenceImageUrl ? 'i2i(attached)' : 'i2i(lastImg)') : 't2i'
  if (refImage) body.reference_image = refImage

  console.log(`[huggingface] request | model=${model} ${width}x${height} mode=${mode} steps=${body.num_inference_steps ?? 'auto'} cfg=${body.guidance_scale ?? 'auto'} seed=${body.seed ?? 'random'}`)

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

  console.log(`[huggingface] success | mode=${data.mode ?? mode} elapsed=${data.elapsed ?? elapsed}s seed=${data.meta?.seed ?? '?'}`)
  return { imageUrl: data.imageUrl, meta: data.meta ?? null }
}
