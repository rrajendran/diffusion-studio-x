import { useState, useEffect, useRef } from 'react'

const POLL_INTERVAL = 10_000 // 10 seconds

async function checkOllama(baseUrl) {
  const url = `${(baseUrl || 'http://127.0.0.1:11434').replace(/\/$/, '')}/api/version`
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(4000) })
    return res.ok
  } catch {
    return false
  }
}

async function checkHFImageServer(baseUrl) {
  const url = `${(baseUrl || 'http://127.0.0.1:8001').replace(/\/$/, '')}/health`
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(4000) })
    return res.ok
  } catch {
    return false
  }
}

/**
 * Polls Ollama and HF image server health.
 * Returns { ollama: bool|null, hfImages: bool|null }
 * null = not yet checked
 */
export function useServerStatus({ ollamaBaseUrl, hfBaseUrl } = {}) {
  const [status, setStatus] = useState({ ollama: null, hfImages: null })
  const timerRef = useRef(null)

  useEffect(() => {
    let cancelled = false

    async function poll() {
      const [ollama, hfImages] = await Promise.all([
        checkOllama(ollamaBaseUrl),
        checkHFImageServer(hfBaseUrl),
      ])
      if (!cancelled) setStatus({ ollama, hfImages })
    }

    poll()
    timerRef.current = setInterval(poll, POLL_INTERVAL)

    return () => {
      cancelled = true
      clearInterval(timerRef.current)
    }
  }, [ollamaBaseUrl, hfBaseUrl])

  return status
}
