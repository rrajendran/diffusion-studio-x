import { useState, useEffect, useRef } from 'react'
import { BRIDGE } from '../lib/ports.js'

const HF_DIRECT = 'http://127.0.0.1:8000'
const HF_BASE   = `${BRIDGE}/api/hf`
const POLL_INTERVAL = 2000

export default function ModelsPage({ onBack, onModelLoaded }) {
  const [models, setModels] = useState([])
  const [search, setSearch] = useState('')
  const [fetching, setFetching] = useState(true)
  const [fetchError, setFetchError] = useState(null)
  const [loadingModel, setLoadingModel] = useState(null)
  const [status, setStatus] = useState(null)
  const pollRef = useRef(null)

  // Fetch initial status + model list on mount
  useEffect(() => {
    fetchStatus()
    fetch(`${HF_BASE}/v1/models`)
      .then(r => r.json())
      .then(d => setModels((d.data ?? []).map(m => m.id)))
      .catch(e => setFetchError(e.message))
      .finally(() => setFetching(false))

    return () => clearInterval(pollRef.current)
  }, [])

  async function fetchStatus() {
    try {
      const r = await fetch(`${HF_DIRECT}/api/images/status`)
      if (r.ok) setStatus(await r.json())
    } catch {}
  }

  function startPolling(modelId) {
    clearInterval(pollRef.current)
    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch(`${HF_DIRECT}/api/images/status`)
        if (!r.ok) return
        const d = await r.json()
        setStatus(d)
        if (d.status === 'ready' || d.status === 'error') {
          clearInterval(pollRef.current)
          setLoadingModel(null)
          if (d.status === 'ready') onModelLoaded?.(modelId)
        }
      } catch {}
    }, POLL_INTERVAL)
  }

  async function handleLoad(modelId) {
    setLoadingModel(modelId)
    setStatus({ status: 'loading', model: modelId })
    try {
      const r = await fetch(`${HF_DIRECT}/api/models/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId }),
      })
      if (!r.ok) {
        const err = await r.text()
        setStatus({ status: 'error', error: err.slice(0, 200) })
        setLoadingModel(null)
        return
      }
      startPolling(modelId)
    } catch (e) {
      setStatus({ status: 'error', error: e.message })
      setLoadingModel(null)
    }
  }

  const filtered = models.filter(m => m.toLowerCase().includes(search.toLowerCase()))
  const isReady = status?.status === 'ready'
  const isLoadingAny = !!loadingModel

  function StatusBanner() {
    if (!status) return null
    const { status: s, model, error, progress } = status
    if (s === 'idle') return null

    const configs = {
      loading: { bg: 'bg-yellow-500/20 border-yellow-400/30', dot: 'bg-yellow-400 animate-pulse', text: 'text-yellow-300' },
      ready:   { bg: 'bg-green-500/20 border-green-400/30',  dot: 'bg-green-400',               text: 'text-green-300' },
      error:   { bg: 'bg-red-500/20 border-red-400/30',      dot: 'bg-red-400',                 text: 'text-red-300' },
    }
    const c = configs[s] ?? configs.loading

    return (
      <div className={`mx-6 mb-3 flex items-center gap-3 px-4 py-3 rounded-xl border ${c.bg}`}>
        <span className={`w-2 h-2 rounded-full flex-shrink-0 ${c.dot}`} />
        <div className="flex-1 min-w-0">
          <span className={`text-xs font-medium ${c.text}`}>
            {s === 'loading' && `Loading${model ? ` ${model}` : ''}…`}
            {s === 'ready'   && `Ready — ${model ?? 'model'} loaded`}
            {s === 'error'   && `Error: ${error ?? 'failed to load'}`}
          </span>
          {s === 'loading' && typeof progress === 'number' && (
            <div className="mt-1.5 h-1 bg-white/10 rounded-full overflow-hidden">
              <div className="h-full bg-yellow-400 transition-all" style={{ width: `${Math.round(progress * 100)}%` }} />
            </div>
          )}
        </div>
        {s === 'ready' && (
          <button onClick={onBack} className="glass-btn px-3 py-1 text-white/80 text-xs flex-shrink-0">
            Go to Chat
          </button>
        )}
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-3 px-6 py-4 border-b border-white/10 flex-shrink-0">
        <button
          onClick={onBack}
          className="glass-btn px-3 py-1.5 text-white/70 hover:text-white text-xs"
        >
          ← Back
        </button>
        <h2 className="text-white font-semibold text-sm">Hugging Face Models</h2>
        <span className="text-white/30 text-xs ml-auto">
          {!fetching && `${filtered.length} model${filtered.length !== 1 ? 's' : ''}`}
        </span>
      </div>

      {/* Status banner */}
      <div className="pt-3">
        <StatusBanner />
      </div>

      {/* Search */}
      <div className="px-6 pb-3">
        <input
          type="text"
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search models…"
          className="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-2.5 text-white text-sm placeholder-white/30 outline-none focus:border-white/40 transition-colors"
        />
      </div>

      {/* Model list */}
      <div className="flex-1 overflow-y-auto px-6 pb-6 flex flex-col gap-2">
        {fetching && (
          <p className="text-white/40 text-sm text-center py-8">Fetching models…</p>
        )}
        {fetchError && (
          <p className="text-red-400 text-sm text-center py-8">{fetchError}</p>
        )}
        {!fetching && !fetchError && filtered.length === 0 && (
          <p className="text-white/40 text-sm text-center py-8">No models found</p>
        )}
        {filtered.map(modelId => {
          const isThisLoading = loadingModel === modelId
          const isThisLoaded  = isReady && status?.model === modelId

          return (
            <div
              key={modelId}
              className="glass-dark flex items-center justify-between px-4 py-3 rounded-xl"
            >
              <span className="text-white text-xs font-mono truncate flex-1 mr-4">{modelId}</span>
              {isThisLoaded ? (
                <span className="text-green-400 text-xs font-medium flex-shrink-0">Loaded ✓</span>
              ) : (
                <button
                  onClick={() => handleLoad(modelId)}
                  disabled={isLoadingAny}
                  className="glass-btn px-4 py-1.5 text-white text-xs font-medium disabled:opacity-40 flex-shrink-0"
                >
                  {isThisLoading ? 'Loading…' : 'Load'}
                </button>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
