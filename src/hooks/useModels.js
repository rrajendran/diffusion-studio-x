import { useState, useEffect } from 'react'

const FETCHERS = {
  ollama: async () => {
    const r = await fetch('http://localhost:11434/api/tags')
    const d = await r.json()
    return (d.models ?? []).map(m => m.name)
  },
  lmstudio: async () => {
    const r = await fetch('http://localhost:1234/v1/models')
    const d = await r.json()
    return (d.data ?? []).map(m => m.id)
  },
}

async function fetchHFCachedModels() {
  const r = await fetch('http://localhost:3001/api/hf-cached-models')
  const d = await r.json()
  return Array.isArray(d) ? d : []
}

async function fetchModelCapabilities() {
  try {
    const r = await fetch('http://localhost:3001/api/image/capabilities')
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

export function useModels(provider) {
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
      setLoadingModels(true)
      Promise.all([fetchHFCachedModels(), fetchModelCapabilities()])
        .then(([list, caps]) => {
          if (cancelled) return
          setImageModels(list)
          setLlmModels(list)
          setModelCapabilities(caps)
        })
        .catch(err => { if (!cancelled) setFetchError(err.message) })
        .finally(() => { if (!cancelled) setLoadingModels(false) })
      return () => { cancelled = true }
    }

    const fetcher = FETCHERS[provider]
    if (!fetcher) return

    let cancelled = false
    setLoadingModels(true)
    fetcher()
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
