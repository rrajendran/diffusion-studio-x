import * as ollama from './ollama.js'
import * as lmstudio from './lmstudio.js'
import * as llamacpp from './llamacpp.js'
import * as huggingface from './huggingface.js'
import { getScaledDimensions as getDimensions } from '../config/aspectRatios.js'
import { BRIDGE } from '../lib/ports.js'

// lastImageUrl: base64 data URL of the previous image, passed as edit context
export async function generateImage(prompt, { provider, model, apiKey, randomSeed, seed, inferenceSteps, guidanceScale, hfBaseUrl, ollamaBaseUrl, lmstudioBaseUrl, llamacppBaseUrl, numFrames, fps }, lastImageUrl = null, aspectRatio = null, signal = null, referenceImageUrl = null) {
  const { width, height } = getDimensions(aspectRatio)
  const hasRef = !!referenceImageUrl
  const hasLast = !!lastImageUrl
  console.log(`[provider] generate | provider=${provider} model=${model} ${width}x${height} | refImage=${hasRef} lastImage=${hasLast} | prompt="${prompt.slice(0, 80)}"`)
  const t0 = performance.now()

  // Resolve seed: use random if randomSeed flag is set (or not specified)
  const resolvedSeed = (randomSeed ?? true) ? undefined : (seed ?? 42)

  const genParams = { width, height, inferenceSteps, guidanceScale, seed: resolvedSeed, hfBaseUrl: hfBaseUrl || undefined, numFrames, fps }

  let result
  switch (provider) {
    case 'ollama':       result = await ollama.generate(prompt, model, lastImageUrl, width, height, signal, referenceImageUrl, ollamaBaseUrl || undefined); break
    case 'lmstudio':    result = await lmstudio.generate(prompt, model, lastImageUrl, width, height, signal, referenceImageUrl, lmstudioBaseUrl || undefined); break
    case 'llamacpp':    result = await llamacpp.generate(prompt, model, lastImageUrl, width, height, signal, referenceImageUrl, llamacppBaseUrl || undefined); break
    case 'huggingface': result = await huggingface.generate(prompt, model, apiKey, lastImageUrl, signal, referenceImageUrl, genParams); break
    default:            throw new Error(`Unknown provider: ${provider}`)
  }

  console.log(`[provider] done | provider=${provider} elapsed=${((performance.now() - t0) / 1000).toFixed(1)}s`)

  // Persist base64 data URLs to the output folder via the bridge server
  if (result?.imageUrl?.startsWith('data:')) {
    try {
      const saveRes = await fetch(`${BRIDGE}/api/save-image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageUrl: result.imageUrl, prompt }),
        signal,
      })
      if (saveRes.ok) {
        const saved = await saveRes.json()
        const savedUrl = saved.imageUrl?.startsWith('/') ? `${BRIDGE}${saved.imageUrl}` : saved.imageUrl
        result = { ...result, imageUrl: savedUrl }
      }
    } catch {
      // non-fatal — keep the data URL if the save fails
    }
  }

  if (result?.videoUrl?.startsWith('data:')) {
    try {
      const saveRes = await fetch(`${BRIDGE}/api/save-image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ videoUrl: result.videoUrl, prompt }),
        signal,
      })
      if (saveRes.ok) {
        const saved = await saveRes.json()
        const savedUrl = saved.videoUrl?.startsWith('/') ? `${BRIDGE}${saved.videoUrl}` : saved.videoUrl
        result = { ...result, videoUrl: savedUrl }
      }
    } catch {
      // non-fatal — keep the data URL if the save fails
    }
  }

  return result
}
