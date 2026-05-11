import { useState, useEffect } from 'react'
import { OLLAMA, LMSTUDIO, BRIDGE } from '../lib/ports.js'
import { ADAPTER_FAMILIES } from '../config/features.js'

export function filterModelsByFeatures(models) {
  return models.filter(id => {
    for (const { pattern, defaultEnabled } of ADAPTER_FAMILIES) {
      if (pattern.test(id)) return defaultEnabled !== false
    }
    return true  // unrecognised model family: always show
  })
}

// Mirror of scanner.py heuristics — used to split LM Studio API models into image vs LLM
const IMAGE_KW = ['flux', 'stable-diffusion', 'sdxl', 'sd3', 'sd-xl', 'z-image', 'zimage',
  'image-turbo', 'kandinsky', 'playgroundai', 'wuerstchen', 'pixart', 'kolors',
  'auraflow', 'hunyuan', 'lumina', 'cogview', 'diffusion']
const TEXT_KW  = ['llama', 'mistral', 'gemma', 'phi', 'falcon', 'vicuna', 'qwen',
  'deepseek', 'yi-', 'openchat', 'wizard', 'orca', 'hermes', 'neural-chat',
  'starling', 'mamba', 'rwkv', 'embed', 'e5-', 'bge-', 'gte-', 'nomic-embed', 'sentence-t5']

function looksLikeImageModel(key = '') {
  const lower = key.toLowerCase()
  if (TEXT_KW.some(k => lower.includes(k))) return false
  if (IMAGE_KW.some(k => lower.includes(k))) return true
  return false
}

const FETCHERS = {
  ollama: async (baseUrl) => {
    const base = (baseUrl || OLLAMA).replace(/\/$/, '')
    const r = await fetch(`${base}/api/tags`)
    const d = await r.json()
    const names = (d.models ?? []).map(m => m.name)
    return { imageModels: names, llmModels: names }
  },
  lmstudio: async (_baseUrl) => {
    const lmsBase = LMSTUDIO   // port 1234 — LM Studio app
    const [imgRes, llmRes] = await Promise.allSettled([
      fetch(`${BRIDGE}/api/lmstudio/models`),   // lms-image-server (local GGUF/MLX)
      fetch(`${lmsBase}/api/v1/models`),         // LM Studio app REST API
    ])

    // Local GGUF/MLX models from lms-image-server (keys are absolute paths)
    const localImageModels = imgRes.status === 'fulfilled' && imgRes.value.ok
      ? (await imgRes.value.json()).map(m => m.key)
      : []

    const llmAll = llmRes.status === 'fulfilled' && llmRes.value.ok
      ? (await llmRes.value.json()).models ?? []
      : []

    // Split LM Studio app models by keyword: image models + text LLMs
    const lmsImageModels = llmAll.filter(m => looksLikeImageModel(m.key)).map(m => m.key)
    const llmModels = llmAll.filter(m => !looksLikeImageModel(m.key) && m.type === 'llm').map(m => m.key)

    // Merge: local models first (routed to lms-image-server), then LM Studio app image models
    const localSet = new Set(localImageModels)
    const imageModels = [...localImageModels, ...lmsImageModels.filter(k => !localSet.has(k))]

    return { imageModels, llmModels }
  },
}

async function fetchHFCachedModels(hfBaseUrl) {
  const qs = hfBaseUrl ? `?hfBaseUrl=${encodeURIComponent(hfBaseUrl)}` : ''
  const r = await fetch(`${BRIDGE}/api/hf-cached-models${qs}`)
  const d = await r.json()
  return Array.isArray(d) ? d : []
}

async function fetchModelCapabilities(hfBaseUrl) {
  try {
    const qs = hfBaseUrl ? `?hfBaseUrl=${encodeURIComponent(hfBaseUrl)}` : ''
    const r = await fetch(`${BRIDGE}/api/image/capabilities${qs}`)
    if (!r.ok) return []
    return await r.json()
  } catch {
    return []
  }
}

/**
 * Returns whether a model_id supports image-to-image,
 * based on the capabilities list returned by the image server.
 */
export function supportsImageToImage(modelId, capabilities) {
  if (!modelId || !capabilities?.length) return false
  for (const cap of capabilities) {
    if (cap.pattern === '*') return cap.supports_i2i
    if (modelId.includes(cap.pattern)) return cap.supports_i2i
  }
  return false
}

/**
 * Returns whether a model_id generates video (not images),
 * based on the capabilities list returned by the image server.
 */
export function supportsVideo(modelId, capabilities) {
  if (!modelId || !capabilities?.length) return false
  for (const cap of capabilities) {
    if (cap.pattern === '*') return !!cap.is_video
    if (modelId.includes(cap.pattern)) return !!cap.is_video
  }
  return false
}

export function useModels(provider, config = {}) {
  const { hfBaseUrl, ollamaBaseUrl, lmstudioBaseUrl } = config
  const [imageModels, setImageModels] = useState([])
  const [llmModels, setLlmModels] = useState([])
  const [loadingModels, setLoadingModels] = useState(false)
  const [fetchError, setFetchError] = useState(null)
  const [modelCapabilities, setModelCapabilities] = useState([])

  useEffect(() => {
    setImageModels([])
    setLlmModels([])
    setFetchError(null)
    setModelCapabilities([])
    setLoadingModels(false)

    if (provider === 'huggingface') {
      let cancelled = false
      let retryTimer = null
      const MAX_RETRIES = 10
      const RETRY_MS = 3000

      async function attemptFetch(attempt = 0) {
        if (cancelled) return
        setLoadingModels(true)
        try {
          const [list, caps] = await Promise.all([fetchHFCachedModels(hfBaseUrl), fetchModelCapabilities(hfBaseUrl)])
          if (cancelled) return
          if (list.length === 0 && attempt < MAX_RETRIES) {
            retryTimer = setTimeout(() => attemptFetch(attempt + 1), RETRY_MS)
            return
          }
          const filtered = filterModelsByFeatures(list)
          setImageModels(filtered)
          setLlmModels(filtered)
          setModelCapabilities(caps)
        } catch (err) {
          if (!cancelled) setFetchError(err.message)
        } finally {
          if (!cancelled) setLoadingModels(false)
        }
      }

      attemptFetch()
      return () => { cancelled = true; clearTimeout(retryTimer) }
    }

    const fetcher = FETCHERS[provider]
    if (!fetcher) return

    let cancelled = false
    let retryTimer = null
    const MAX_RETRIES = 5
    const RETRY_MS = 2000

    const baseUrl = provider === 'ollama' ? ollamaBaseUrl
                  : provider === 'lmstudio' ? lmstudioBaseUrl
                  : undefined

    async function attempt(n = 0) {
      if (cancelled) return
      retryTimer = null
      setLoadingModels(true)
      let willRetry = false
      try {
        const result = await fetcher(baseUrl)
        if (cancelled) return
        const { imageModels: imgs, llmModels: llms } = result
        // Always update LLM models immediately so the enhance dropdown works
        if (llms.length > 0) setLlmModels(llms)
        if (imgs.length === 0 && n < MAX_RETRIES) {
          willRetry = true
          retryTimer = setTimeout(() => attempt(n + 1), RETRY_MS)
          return
        }
        setImageModels(imgs)
        if (llms.length > 0) setLlmModels(llms)
        setFetchError(null)
      } catch (err) {
        if (cancelled) return
        if (n < MAX_RETRIES) {
          willRetry = true
          retryTimer = setTimeout(() => attempt(n + 1), RETRY_MS)
          return
        }
        setFetchError(err.message)
      } finally {
        if (!cancelled && !willRetry) setLoadingModels(false)
      }
    }

    attempt()
    return () => { cancelled = true; clearTimeout(retryTimer) }
  }, [provider, ollamaBaseUrl, lmstudioBaseUrl])

  return { imageModels, llmModels, loadingModels, fetchError, modelCapabilities }
}
