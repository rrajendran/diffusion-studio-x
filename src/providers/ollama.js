import { BRIDGE, bridgeLog } from '../lib/ports.js'

export async function generate(prompt, model, lastImageUrl = null, width = 512, height = 512, signal = null, referenceImageUrl = null, ollamaBaseUrl = null) {
  const refImage = referenceImageUrl
  const bridgeUrl = `${BRIDGE}/api/ollama/generate`

  bridgeLog(`ollama.js: → POST ${bridgeUrl} | model=${model || 'stable-diffusion'} ${width}x${height} hasRef=${!!refImage} ollamaBaseUrl=${ollamaBaseUrl ?? '(default)'}`)

  let res
  try {
    res = await fetch(bridgeUrl, {
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
  } catch (fetchErr) {
    bridgeLog(`ollama.js: fetch failed: ${fetchErr.name} — ${fetchErr.message}`, 'error')
    throw fetchErr
  }

  bridgeLog(`ollama.js: ← status=${res.status} ok=${res.ok}`)

  const data = await res.json()
  bridgeLog(`ollama.js: response keys=${Object.keys(data).join(',')} imageUrl=${data.imageUrl ?? '(none)'} error=${data.error ?? '(none)'}`)

  if (!res.ok) throw new Error(data.error ?? `Ollama bridge error: ${res.status}`)
  if (!data.imageUrl) throw new Error('Ollama returned no image data')

  const imageUrl = data.imageUrl.startsWith('/') ? `${BRIDGE}${data.imageUrl}` : data.imageUrl
  return { imageUrl, meta: data.meta ?? null }
}
