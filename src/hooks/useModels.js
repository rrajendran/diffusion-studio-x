import { useState, useEffect } from 'react'
import { OLLAMA, LMSTUDIO, BRIDGE } from '../lib/ports.js'


const FETCHERS = {
  ollama: async (ollamaBase) => {
    const base = (ollamaBase || OLLAMA).replace(/\/$/, '')
    const r = await fetch(`${base}/api/tags`)
    const d = await r.json()
    return (d.models ?? []).map(m => m.name)
  },
  lmstudio: async () => {
    const r = await fetch(`${LMSTUDIO}/v1/models`)
    const d = await r.json()
    return (d.data ?? []).map(m => m.id)
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
          setImageModels(list)
          setLlmModels(list)
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
    setLoadingModels(true)
    fetcher(provider === 'ollama' ? ollamaBaseUrl : undefined)
      .then(list => {
        if (cancelled) return
        setImageModels(list)
        setLlmModels(list)
      })
      .catch(err => { if (!cancelled) setFetchError(err.message) })
      .finally(() => { if (!cancelled) setLoadingModels(false) })

    return () => { cancelled = true }
  }, [provider])

  return { imageModels, llmModels, loadingModels, fetchError, modelCapabilities }
}
