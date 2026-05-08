import { useState, useRef, useEffect } from 'react'
import { ASPECT_RATIOS, DEFAULT_RATIO } from '../config/aspectRatios.js'
import { supportsImageToImage } from '../hooks/useModels.js'

const PROVIDERS = [
  { value: 'huggingface', label: 'Hugging Face' },
  { value: 'ollama',      label: 'Ollama' },
  { value: 'lmstudio',    label: 'LM Studio' },
  { value: 'llamacpp',    label: 'llama.cpp' },
]

// ── SVG icons ────────────────────────────────────────────────
function PaperclipIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
    </svg>
  )
}

function MicIcon({ active }) {
  const color = active ? '#22d3ee' : 'currentColor'
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
      <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
      <line x1="12" y1="19" x2="12" y2="23"/>
      <line x1="8" y1="23" x2="16" y2="23"/>
    </svg>
  )
}

function PersonIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
      <circle cx="12" cy="7" r="4"/>
    </svg>
  )
}

function ChipIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="6" height="6"/>
      <path d="M15 9V5h-2M9 9V5h2M15 15v4h-2M9 15v4h2M9 9H5v2M9 15H5v-2M15 9h4v2M15 15h4v-2"/>
    </svg>
  )
}

function BrainIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.88A2.5 2.5 0 0 1 9.5 2z"/>
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.88A2.5 2.5 0 0 0 14.5 2z"/>
    </svg>
  )
}

function AspectIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2"/>
      <path d="M3 9h18M9 21V9"/>
    </svg>
  )
}

function ChevronIcon() {
  return (
    <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="6 9 12 15 18 9"/>
    </svg>
  )
}

function ArrowUpIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="19" x2="12" y2="5"/>
      <polyline points="5 12 12 5 19 12"/>
    </svg>
  )
}

// ── Compact toolbar dropdown ──────────────────────────────────
function ToolbarSelect({ icon, value, onChange, options, placeholder, loading, color }) {
  const textColor = color ?? 'rgba(255,255,255,0.65)'
  return (
    <label
      className="flex items-center gap-1 px-1.5 py-1 rounded-lg cursor-pointer transition-colors hover:bg-white/8 flex-shrink-0"
      style={{ color: textColor }}
    >
      {icon}
      {options.length > 0 ? (
        <select
          value={value}
          onChange={e => onChange(e.target.value)}
          className="bg-transparent text-xs outline-none cursor-pointer appearance-none max-w-[110px] truncate"
          style={{ color: textColor }}
          title={value}
        >
          {options.map(o => (
            <option key={o.value ?? o} value={o.value ?? o} className="bg-gray-950 text-white">
              {o.label ?? (o.split('/').pop().slice(0, 22))}
            </option>
          ))}
        </select>
      ) : (
        <input
          type="text"
          value={value}
          onChange={e => onChange(e.target.value)}
          placeholder={loading ? 'loading…' : placeholder}
          className="bg-transparent text-xs outline-none w-20"
          style={{ color: textColor }}
          title={value}
        />
      )}
      <span className="opacity-40 flex-shrink-0"><ChevronIcon /></span>
    </label>
  )
}

// ── Main component ────────────────────────────────────────────
export default function ChatInput({
  onSend, loading, onStop, onRewrite,
  config, onConfigChange,
  imageModels = [], llmModels = [], loadingModels = false,
  modelCapabilities = [],
  prefill = null,
}) {
  const [value, setValue] = useState('')
  const [ratio, setRatio] = useState(DEFAULT_RATIO)
  const [rewriting, setRewriting] = useState(false)
  const [refImage, setRefImage] = useState(null)
  const [rewriteError, setRewriteError] = useState(null)
  const [listening, setListening] = useState(false)
  const [micError, setMicError] = useState(null)

  // Populate textarea + reference image when the parent requests an edit
  useEffect(() => {
    if (!prefill) return
    setValue(prefill.content ?? '')
    setRefImage(prefill.referenceImageUrl
      ? { dataUrl: prefill.referenceImageUrl, name: 'reference' }
      : null
    )
  }, [prefill])

  const fileInputRef = useRef(null)
  const recognitionRef = useRef(null)
  const baseTextRef = useRef('')
  const finalSuffixRef = useRef('')

  // ── Attachment ────────────────────────────────────────────
  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => setRefImage({ dataUrl: ev.target.result, name: file.name })
    reader.readAsDataURL(file)
    e.target.value = ''
  }

  // ── Microphone ────────────────────────────────────────────
  const toggleMic = () => {
    setMicError(null)

    if (listening) {
      recognitionRef.current?.stop()
      return
    }

    const SR = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SR) {
      setMicError('Speech recognition is not supported in this browser.')
      return
    }

    const rec = new SR()
    recognitionRef.current = rec
    rec.continuous = true
    rec.interimResults = true
    rec.lang = 'en-US'

    baseTextRef.current = value.trimEnd()
    finalSuffixRef.current = ''

    rec.onstart = () => setListening(true)

    rec.onresult = (e) => {
      let interimText = ''
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript
        if (e.results[i].isFinal) {
          finalSuffixRef.current = (finalSuffixRef.current + ' ' + t).trim()
        } else {
          interimText += t
        }
      }
      const parts = [baseTextRef.current, finalSuffixRef.current, interimText].filter(Boolean)
      setValue(parts.join(' '))
    }

    rec.onerror = (e) => {
      setListening(false)
      if (e.error === 'not-allowed') setMicError('Microphone access denied. Allow mic in browser settings.')
      else if (e.error !== 'aborted') setMicError(`Mic error: ${e.error}`)
    }

    rec.onend = () => {
      setListening(false)
      setValue(v => v.trim())
    }

    try {
      rec.start()
    } catch {
      setMicError('Could not start mic. Try again.')
    }
  }

  // ── Submit / Enhance ──────────────────────────────────────
  const submit = () => {
    const trimmed = value.trim()
    if (!trimmed || loading || rewriting) return
    if (listening) recognitionRef.current?.stop()
    onSend(trimmed, ratio, refImage?.dataUrl ?? null)
    setValue('')
    setRefImage(null)
    finalSuffixRef.current = ''
  }

  const rewrite = async () => {
    const trimmed = value.trim()
    if (!trimmed || loading || rewriting) return
    setRewriting(true)
    setRewriteError(null)
    try {
      const newText = await onRewrite(trimmed)
      setValue(newText)
    } catch (e) {
      setRewriteError(e.message)
    }
    setRewriting(false)
  }

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  // ── Config helpers ────────────────────────────────────────
  const cfg = (patch) => onConfigChange?.({ ...config, ...patch })

  const providerOptions = PROVIDERS.map(p => ({ value: p.value, label: p.label }))
  const imgModelOptions = imageModels.length > 0 ? imageModels : []
  const llmModelOptions = llmModels.length > 0 ? llmModels : []
  const ratioOptions = ASPECT_RATIOS.map(r => ({ value: r.value, label: r.label }))
  const dims = ASPECT_RATIOS.find(r => r.value === ratio)
  const modelSupportsI2I = supportsImageToImage(config?.model, modelCapabilities)

  return (
    <div className="px-4 py-4">
      <div className="rounded-2xl border border-cyan-400/40 bg-[#060b18]/92 backdrop-blur-sm shadow-[0_0_28px_rgba(34,211,238,0.10)] overflow-hidden">

        {/* Reference image thumbnail */}
        {refImage && (
          <div className="px-4 pt-3">
            <div className="relative w-fit">
              <img
                src={refImage.dataUrl}
                alt={refImage.name}
                className="max-h-20 rounded-lg object-contain border border-white/20"
              />
              <button
                onClick={() => setRefImage(null)}
                title="Remove"
                className="absolute -top-1.5 -right-1.5 w-4 h-4 flex items-center justify-center rounded-full bg-black/70 text-white text-xs hover:bg-black/90"
              >
                ✕
              </button>
            </div>
          </div>
        )}

        {/* Textarea */}
        <div className="px-4 pt-4 pb-2">
          <textarea
            className="w-full bg-transparent text-white placeholder-white/35 text-sm outline-none leading-relaxed resize-none"
            style={{ minHeight: '48px', maxHeight: '200px' }}
            placeholder="Describe an image…"
            rows={2}
            value={value}
            onChange={e => setValue(e.target.value)}
            onKeyDown={onKeyDown}
          />
        </div>

        {/* Toolbar */}
        <div className="flex items-center gap-0.5 px-3 pb-3 overflow-x-auto">

          {/* Attachment */}
          <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={loading}
            title={modelSupportsI2I ? 'Attach reference image (image-to-image)' : 'Attach reference image'}
            className={`p-1.5 rounded-lg transition-colors flex-shrink-0 ${refImage ? 'text-cyan-400' : 'text-white/50 hover:text-white/80'}`}
          >
            <PaperclipIcon />
          </button>

          {/* Microphone */}
          <button
            onClick={toggleMic}
            title={listening ? 'Stop recording' : 'Voice input'}
            className={`relative p-1.5 rounded-lg transition-colors flex-shrink-0 ${
              listening
                ? 'text-cyan-400 animate-mic-pulse'
                : 'text-white/50 hover:text-white/80'
            }`}
          >
            <MicIcon active={listening} />
          </button>

          {/* Divider */}
          <div className="w-px h-4 bg-white/20 mx-1.5 flex-shrink-0" />

          {/* Provider */}
          <ToolbarSelect
            icon={<PersonIcon />}
            value={config?.provider ?? 'huggingface'}
            onChange={v => cfg({ provider: v })}
            options={providerOptions}
            color="#22d3ee"
          />

          {/* Image model + i2i capability badge */}
          <div className="flex items-center gap-0.5 flex-shrink-0">
            <ToolbarSelect
              icon={<ChipIcon />}
              value={config?.model ?? ''}
              onChange={v => cfg({ model: v })}
              options={imgModelOptions}
              placeholder="img model"
              loading={loadingModels}
            />
            {modelSupportsI2I && (
              <span
                title="This model supports image-to-image (attach a reference image)"
                className="text-[9px] font-semibold px-1 py-0.5 rounded bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 leading-none flex-shrink-0"
              >
                i2i
              </span>
            )}
          </div>

          {/* Conversational model */}
          <ToolbarSelect
            icon={<BrainIcon />}
            value={config?.llmModel ?? ''}
            onChange={v => cfg({ llmModel: v })}
            options={llmModelOptions}
            placeholder="llm model"
            loading={loadingModels}
          />

          {/* Aspect ratio */}
          <ToolbarSelect
            icon={<AspectIcon />}
            value={ratio}
            onChange={setRatio}
            options={ratioOptions}
          />

          {/* Spacer */}
          <div className="flex-1 min-w-[8px]" />

          {/* Enhance — static ring path + orbiting dot only while rewriting */}
          <div
            className={`enhance-btn-wrap transition-opacity ${
              !value.trim() || loading ? 'opacity-35' : 'opacity-100'
            }`}
          >
            {rewriting && <div className="enhance-orbit" />}
            <button
              onClick={rewrite}
              disabled={!value.trim() || loading || rewriting}
              title="Enhance prompt with AI"
              className="absolute inset-0 flex items-center justify-center rounded-full bg-[#060b18] text-white/60 hover:text-white/95 transition-colors disabled:cursor-not-allowed z-10"
            >
              <span className={rewriting ? 'animate-twinkle' : ''}>✨</span>
            </button>
          </div>

          {/* Send / Stop */}
          {loading ? (
            <button
              onClick={onStop}
              title="Stop generation"
              className="w-8 h-8 flex items-center justify-center rounded-full bg-gradient-to-br from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 transition-all flex-shrink-0"
            >
              <span className="w-2.5 h-2.5 bg-white rounded-sm" />
            </button>
          ) : (
            <button
              onClick={submit}
              disabled={!value.trim() || rewriting}
              title="Send (Enter)"
              className="w-8 h-8 flex items-center justify-center rounded-full bg-gradient-to-br from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed transition-all flex-shrink-0 text-white"
            >
              <ArrowUpIcon />
            </button>
          )}
        </div>
      </div>

      {/* Error messages */}
      {(rewriteError || micError) && (
        <p className="text-red-400 text-xs mt-1.5 px-1">{rewriteError ?? micError}</p>
      )}
      {listening && (
        <p className="text-cyan-400/80 text-xs text-center mt-1.5 animate-pulse">Listening… speak now</p>
      )}
      <p className="text-white/22 text-xs text-center mt-1">
        Enter to send · Shift+Enter for new line · {dims?.width}×{dims?.height}px
      </p>
    </div>
  )
}
