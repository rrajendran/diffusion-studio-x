import * as ollama from './ollama.js'
import * as lmstudio from './lmstudio.js'
import * as llamacpp from './llamacpp.js'
import * as huggingface from './huggingface.js'
import { getScaledDimensions as getDimensions } from '../config/aspectRatios.js'

// lastImageUrl: base64 data URL of the previous image, passed as edit context
export async function generateImage(prompt, { provider, model, apiKey }, lastImageUrl = null, aspectRatio = null, signal = null, referenceImageUrl = null) {
  const { width, height } = getDimensions(aspectRatio)
  const hasRef = !!referenceImageUrl
  const hasLast = !!lastImageUrl
  console.log(`[provider] generate | provider=${provider} model=${model} ${width}x${height} | refImage=${hasRef} lastImage=${hasLast} | prompt="${prompt.slice(0, 80)}"`)
  const t0 = performance.now()

  let result
  switch (provider) {
    case 'ollama':       result = await ollama.generate(prompt, model, lastImageUrl, width, height, signal, referenceImageUrl); break
    case 'lmstudio':    result = await lmstudio.generate(prompt, model, lastImageUrl, width, height, signal, referenceImageUrl); break
    case 'llamacpp':    result = await llamacpp.generate(prompt, model, lastImageUrl, width, height, signal, referenceImageUrl); break
    case 'huggingface': result = await huggingface.generate(prompt, model, apiKey, lastImageUrl, width, height, signal, referenceImageUrl); break
    default:            throw new Error(`Unknown provider: ${provider}`)
  }

  console.log(`[provider] done | provider=${provider} elapsed=${((performance.now() - t0) / 1000).toFixed(1)}s`)

  // Persist base64 data URLs to the output folder via the bridge server
  if (result?.imageUrl?.startsWith('data:')) {
    try {
      const saveRes = await fetch('http://localhost:3001/api/save-image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageUrl: result.imageUrl, prompt }),
        signal,
      })
      if (saveRes.ok) {
        const saved = await saveRes.json()
        result = { ...result, imageUrl: saved.imageUrl }
      }
    } catch {
      // non-fatal — keep the data URL if the save fails
    }
  }

  return result
}
