import { useState, useEffect } from 'react'
import { OLLAMA, BRIDGE } from '../lib/ports.js'
import { ADAPTER_FAMILIES } from '../config/features.js'

export function filterModelsByFeatures(models) {
  return models.filter(id => {
    for (const { pattern, defaultEnabled } of ADAPTER_FAMILIES) {
      if (pattern.test(id)) return defaultEnabled !== false
    }
    return true  // unrecognised model family: always show
  })
}

const FETCHERS = {
  ollama: async (baseUrl) => {
    const base = (baseUrl || OLLAMA).replace(/\/$/, '')
    const r = await fetch(`${base}/api/tags`)
    const d = await r.json()
    const names = (d.models ?? []).map(m => m.name)
    return { imageModels: names, llmModels: names }
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
  const { hfBaseUrl, ollamaBaseUrl } = config
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

    const baseUrl = provider === 'ollama' ? ollamaBaseUrl : undefined

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
  }, [provider, ollamaBaseUrl])

  return { imageModels, llmModels, loadingModels, fetchError, modelCapabilities }
}
