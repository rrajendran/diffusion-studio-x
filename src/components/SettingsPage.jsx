import { useState, useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { Flags } from 'react-feature-flags'
import { ASPECT_RATIOS, DEFAULT_RATIO } from '../config/aspectRatios.js'
import { PROVIDERS as ALL_PROVIDERS } from '../config/features.js'
import { useServerStatus } from '../hooks/useServerStatus.js'
import { BRIDGE } from '../lib/ports.js'

// ── Helpers ───────────────────────────────────────────────────────────────────
function Section({ title, children }) {
  return (
    <div className="mb-6">
      <h3 className="text-white/50 text-xs font-semibold uppercase tracking-widest mb-3 px-1">{title}</h3>
      <div className="glass-dark rounded-2xl overflow-hidden divide-y divide-white/10">
        {children}
      </div>
    </div>
  )
}

function Row({ label, hint, children }) {
  return (
    <div className="flex items-center justify-between gap-4 px-4 py-3">
      <div className="min-w-0">
        <p className="text-white text-sm">{label}</p>
        {hint && <p className="text-white/35 text-xs mt-0.5">{hint}</p>}
      </div>
      <div className="flex-shrink-0">{children}</div>
    </div>
  )
}

function SliderRow({ label, hint, value, min, max, step = 1, onChange, displayValue, theme = 'dark' }) {
  const isLight = theme === 'light'
  const pct = ((value - min) / (max - min)) * 100
  const trackFill    = isLight ? '#0071e3' : '#22d3ee'
  const trackFillEnd = isLight ? '#5856d6' : '#818cf8'
  const trackEmpty   = isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)'

  return (
    <div className="px-4 py-3">
      <div className="flex items-center justify-between mb-3">
        <div>
          <p className="text-white text-sm">{label}</p>
          {hint && <p className="text-white/35 text-xs mt-0.5">{hint}</p>}
        </div>
        <span
          className="text-sm font-mono tabular-nums px-2 py-0.5 rounded-lg"
          style={{
            color:      isLight ? '#0071e3' : '#22d3ee',
            background: isLight ? 'rgba(0,113,227,0.08)' : 'rgba(34,211,238,0.1)',
            border:     isLight ? '1px solid rgba(0,113,227,0.18)' : '1px solid rgba(34,211,238,0.2)',
          }}
        >
          {displayValue ?? value}
        </span>
      </div>

      {/* Glassmorphism track wrapper */}
      <div
        className="relative rounded-full px-0 py-2"
        style={{
          background:  isLight ? 'rgba(0,0,0,0.04)' : 'rgba(255,255,255,0.04)',
          border:      isLight ? '1px solid rgba(0,0,0,0.08)' : '1px solid rgba(255,255,255,0.08)',
          backdropFilter: 'blur(8px)',
        }}
      >
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={e => onChange(Number(e.target.value))}
          className="dsx-slider w-full"
          style={{
            background: `linear-gradient(to right, ${trackFill} 0%, ${trackFillEnd} ${pct}%, ${trackEmpty} ${pct}%)`,
          }}
          data-theme={theme}
        />
      </div>

      <div className="flex justify-between mt-1.5 px-0.5">
        <span className="text-white/25 text-[10px]">{min}</span>
        <span className="text-white/25 text-[10px]">{max}</span>
      </div>
    </div>
  )
}

function SelectField({ value, onChange, options, placeholder }) {
  const [open, setOpen] = useState(false)
  const [menuStyle, setMenuStyle] = useState({})
  const triggerRef = useRef(null)

  function openMenu() {
    if (triggerRef.current) {
      const r = triggerRef.current.getBoundingClientRect()
      setMenuStyle({ top: r.bottom + 4, left: r.left, width: r.width })
    }
    setOpen(true)
  }

  useEffect(() => {
    if (!open) return
    function handle(e) {
      const menu = document.getElementById('dsx-select-portal')
      if (!triggerRef.current?.contains(e.target) && !menu?.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', handle)
    return () => document.removeEventListener('mousedown', handle)
  }, [open])

  const selected = options.find(o => (typeof o === 'string' ? o : o.value) === value)
  const displayLabel = selected
    ? (typeof selected === 'string' ? selected : selected.label)
    : (placeholder || value || '—')

  // Outer wrapper carries the theme class so descendant CSS selectors (.theme-light .glass etc.) work
  const themeClass = localStorage.getItem('img-gen-theme') === 'light' ? 'theme-light' : ''

  const menu = open ? createPortal(
    <div className={themeClass} style={{ position: 'fixed', zIndex: 9999, ...menuStyle }}>
      <div
        id="dsx-select-portal"
        className="dsx-select-menu glass rounded-xl overflow-hidden max-h-52 overflow-y-auto scrollbar-thin"
        style={{ boxShadow: '0 8px 32px rgba(0,0,0,0.28)' }}
      >
        {placeholder && (
          <div className="dsx-select-item px-3 py-2 text-white/40 text-sm cursor-default select-none border-b border-white/10">
            {placeholder}
          </div>
        )}
        {options.map(o => {
          const v = typeof o === 'string' ? o : o.value
          const l = typeof o === 'string' ? o : o.label
          return (
            <div
              key={v}
              onMouseDown={() => { onChange(v); setOpen(false) }}
              className={`dsx-select-item px-3 py-2 text-white text-sm cursor-pointer transition-colors ${
                v === value ? 'selected bg-white/15' : 'hover:bg-white/10'
              }`}
            >
              {l}
            </div>
          )
        })}
      </div>
    </div>,
    document.body
  ) : null

  return (
    <>
      <div ref={triggerRef} style={{ minWidth: '180px', maxWidth: '220px' }}>
        <button
          type="button"
          onClick={() => open ? setOpen(false) : openMenu()}
          className="w-full flex items-center justify-between gap-2 bg-white/10 border border-white/20 rounded-xl text-white text-sm px-3 py-1.5 outline-none cursor-pointer"
        >
          <span className="truncate flex-1 text-left">{displayLabel}</span>
          <svg
            width="10" height="10" viewBox="0 0 10 10" fill="none"
            className={`flex-shrink-0 opacity-50 transition-transform ${open ? 'rotate-180' : ''}`}
          >
            <path d="M1 3l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>
      {menu}
    </>
  )
}

function Toggle({ checked, onChange }) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-all border ${
        checked ? 'dsx-toggle-on bg-white/40 border-white/40' : 'dsx-toggle-off bg-white/15 border-white/20'
      }`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform shadow-sm ${
          checked ? 'translate-x-6' : 'translate-x-1'
        }`}
      />
    </button>
  )
}

function StatusDot({ up }) {
  if (up === null) return <span className="inline-block w-2 h-2 rounded-full bg-white/20" title="Checking…" />
  return up
    ? <span className="inline-block w-2 h-2 rounded-full bg-green-400 shadow-[0_0_6px_1px_rgba(74,222,128,0.7)]" title="Online" />
    : <span className="inline-block w-2 h-2 rounded-full bg-red-400 shadow-[0_0_6px_1px_rgba(248,113,113,0.7)]" title="Offline" />
}

function TextInput({ value, onChange, placeholder, monospace = false }) {
  return (
    <input
      type="text"
      value={value}
      onChange={e => onChange(e.target.value)}
      placeholder={placeholder}
      className={`w-full bg-white/10 border border-white/20 rounded-xl text-white text-sm px-3 py-2 outline-none placeholder-white/20 ${monospace ? 'font-mono' : ''}`}
    />
  )
}

// ── Main component ────────────────────────────────────────────────────────────
export default function SettingsPage({ config, onConfigChange, imageModels, llmModels, loadingModels, onClose, theme = 'dark' }) {
  const [local, setLocal] = useState({ ...config })
  const serverStatus = useServerStatus({ ollamaBaseUrl: local.ollamaBaseUrl, hfBaseUrl: local.hfBaseUrl })

  // Keep local in sync if parent config changes externally (e.g. model auto-select)
  useEffect(() => {
    setLocal(c => ({ ...config, ...c }))
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.provider])

  const set = (key, val) => {
    const next = { ...local, [key]: val }
    setLocal(next)
    onConfigChange(next)
  }



  const provider = ALL_PROVIDERS.find(p => p.value === local.provider)

  // Steps display: 4 is the sentinel meaning "adapter default"
  const stepsDisplay = local.inferenceSteps === 4 ? 'auto' : String(local.inferenceSteps)
  // Guidance display: 0.0 is the sentinel
  const guidanceDisplay = local.guidanceScale === 0.0 ? 'auto' : local.guidanceScale.toFixed(1)

  return (
    <div className="flex-1 overflow-y-auto px-5 py-5 scrollbar-thin">
      <div className="max-w-2xl mx-auto">

        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-white font-bold text-lg">Settings</h2>
            <p className="text-white/40 text-xs mt-0.5">Configure providers, models and generation defaults</p>
          </div>
          <button
            onClick={onClose}
            className="glass-btn px-3 py-1.5 text-white/60 hover:text-white text-sm"
          >
            ✕ Close
          </button>
        </div>

        {/* Provider */}
        <Section title="Provider">
          <Row label="Active provider" hint={provider?.hint}>
            <SelectField
              value={local.provider}
              onChange={val => set('provider', val)}
              options={ALL_PROVIDERS.filter(p => p.defaultEnabled !== false)}
            />
          </Row>
          {local.provider === 'huggingface' && (
            <div className="px-4 py-3">
              <p className="text-white text-sm mb-1.5">HuggingFace API key</p>
              <p className="text-white/35 text-xs mb-2">Optional — required only for gated or private models</p>
              <input
                type="password"
                value={local.apiKey ?? ''}
                onChange={e => set('apiKey', e.target.value)}
                placeholder="hf_…"
                className="w-full bg-white/10 border border-white/20 rounded-xl text-white text-sm px-3 py-2 outline-none placeholder-white/20 font-mono"
              />
            </div>
          )}
        </Section>


        {/* Service URLs — shown based on selected provider */}
        <Section title="Service URLs">
          {local.provider === 'huggingface' && (
            <div className="px-4 py-3">
              <div className="flex items-center gap-2 mb-1">
                <p className="text-white text-sm">HF image server</p>
                <StatusDot up={serverStatus.hfImages} />
              </div>
              <p className="text-white/35 text-xs mb-2">Base URL of the diffusers image server you run separately</p>
              <TextInput
                value={local.hfBaseUrl ?? ''}
                onChange={val => set('hfBaseUrl', val)}
                placeholder="http://127.0.0.1:8001"
                monospace
              />
            </div>
          )}
          {local.provider === 'ollama' && (
            <div className="px-4 py-3">
              <div className="flex items-center gap-2 mb-1">
                <p className="text-white text-sm">Ollama base URL</p>
                <StatusDot up={serverStatus.ollama} />
              </div>
              <p className="text-white/35 text-xs mb-2">Change if Ollama runs on a different host or port</p>
              <TextInput
                value={local.ollamaBaseUrl ?? ''}
                onChange={val => set('ollamaBaseUrl', val)}
                placeholder="http://127.0.0.1:11434"
                monospace
              />
            </div>
          )}
        </Section>

        {/* Models */}
        <Section title="Models">
          <Row
            label="Image model"
            hint={loadingModels ? 'Loading available models…' : `${imageModels.length} model${imageModels.length !== 1 ? 's' : ''} available`}
          >
            {loadingModels ? (
              <select disabled className="bg-white/10 border border-white/20 rounded-xl text-white/50 text-sm px-3 py-1.5 w-52 cursor-not-allowed">
                <option>Loading models…</option>
              </select>
            ) : imageModels.length > 0 ? (
              <SelectField
                value={local.model}
                onChange={val => set('model', val)}
                options={imageModels}
              />
            ) : (
              <input
                type="text"
                value={local.model}
                onChange={e => set('model', e.target.value)}
                placeholder="org/model-name"
                className="bg-white/10 border border-white/20 rounded-xl text-white text-sm px-3 py-1.5 outline-none placeholder-white/20 w-52"
              />
            )}
          </Row>
          <Row
            label="Text / LLM model"
            hint="Used for conversational routing and prompt enhancement"
          >
            {llmModels.length > 0 ? (
              <SelectField
                value={local.llmModel}
                onChange={val => set('llmModel', val)}
                options={llmModels}
              />
            ) : (
              <input
                type="text"
                value={local.llmModel ?? ''}
                onChange={e => set('llmModel', e.target.value)}
                placeholder="llama3"
                className="bg-white/10 border border-white/20 rounded-xl text-white text-sm px-3 py-1.5 outline-none placeholder-white/20 w-52"
              />
            )}
          </Row>
        </Section>

        {/* Seed */}
        <Section title="Seed">
          <Row
            label="Random seed"
            hint={local.randomSeed ? 'A new random seed is chosen for each generation' : 'Use the fixed seed value below'}
          >
            <Toggle checked={local.randomSeed ?? true} onChange={val => set('randomSeed', val)} />
          </Row>
          {!(local.randomSeed ?? true) && (
            <div className="px-4 py-3">
              <p className="text-white text-sm mb-1.5">Seed value</p>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min={0}
                  max={4294967295}
                  value={local.seed ?? 42}
                  onChange={e => set('seed', Math.max(0, Math.min(4294967295, Number(e.target.value))))}
                  className="bg-white/10 border border-white/20 rounded-xl text-white text-sm px-3 py-2 outline-none font-mono w-40"
                />
                <button
                  onClick={() => set('seed', Math.floor(Math.random() * 4294967295))}
                  className="glass-btn px-3 py-2 text-white/60 hover:text-white text-xs"
                >
                  🎲 Random
                </button>
              </div>
            </div>
          )}
        </Section>

        {/* Generation parameters */}
        <Section title="Generation parameters">
          <SliderRow
            label="Inference steps"
            hint="Set to 4 to use each model's built-in default"
            min={1}
            max={150}
            step={1}
            value={local.inferenceSteps ?? 4}
            displayValue={stepsDisplay}
            onChange={val => set('inferenceSteps', val)}
            theme={theme}
          />
          <SliderRow
            label="Guidance scale (CFG)"
            hint="Set to 0.0 to use each model's built-in default"
            min={0}
            max={20}
            step={0.5}
            value={local.guidanceScale ?? 0.0}
            displayValue={guidanceDisplay}
            onChange={val => set('guidanceScale', val)}
            theme={theme}
          />
        </Section>

        {/* Video generation — gated on provider.huggingface AND adapter.wan both active */}
        <Flags
          authorizedFlags={['provider.huggingface', 'adapter.wan']}
          exactFlags
          renderOn={() => (
            <Section title="Video generation (Wan2.2)">
              <SliderRow
                label="Number of frames"
                hint="Frames to generate. On Apple Silicon start at 17 (≤2 GB attention); 49 requires 64 GB+ RAM."
                min={9}
                max={121}
                step={8}
                value={local.numFrames ?? 17}
                onChange={val => set('numFrames', val)}
                theme={theme}
              />
              <SliderRow
                label="Frame rate (fps)"
                hint="Playback speed of the output video"
                min={8}
                max={30}
                step={1}
                value={local.fps ?? 16}
                onChange={val => set('fps', val)}
                theme={theme}
              />
            </Section>
          )}
        />

        {/* Gallery */}
        <Section title="Gallery">
          <Row label="Max items" hint="Maximum items loaded in the gallery (10–500)">
            <input
              type="number"
              min={10}
              max={500}
              value={local.maxGalleryItems ?? 100}
              onChange={e => set('maxGalleryItems', Math.max(10, Math.min(500, Number(e.target.value))))}
              className="bg-white/10 border border-white/20 rounded-xl text-white text-sm px-3 py-1.5 outline-none w-24 font-mono"
            />
          </Row>
        </Section>

        {/* Storage */}
        <Section title="Storage">
          <div className="px-4 py-3">
            <p className="text-white text-sm mb-1">Images output folder</p>
            <p className="text-white/35 text-xs mb-2">Where generated images are saved on disk</p>
            <div className="flex gap-2">
              <TextInput
                value={local.outputPath ?? ''}
                onChange={val => set('outputPath', val)}
                placeholder="Default: ~/Library/Application Support/…/output"
                monospace
              />
              <button
                onClick={async () => {
                  if (!window.__TAURI__) return
                  try {
                    const { open } = await import('@tauri-apps/plugin-dialog')
                    const selected = await open({ directory: true, multiple: false })
                    if (selected && selected !== local.outputPath) {
                      const oldPath = local.outputPath || ''
                      try {
                        await fetch(`${BRIDGE}/api/move-output-dir`, {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ newPath: selected }),
                        })
                      } catch { /* non-fatal */ }
                      set('outputPath', selected)
                    }
                  } catch { /* dialog unavailable */ }
                }}
                className="glass-btn px-3 py-2 text-white/60 hover:text-white text-xs whitespace-nowrap flex-shrink-0"
                title="Browse for folder"
              >
                Browse…
              </button>
            </div>
          </div>
        </Section>

        {/* Info box */}
        <div className="glass rounded-xl px-4 py-3 text-white/35 text-xs leading-relaxed">
          <strong className="text-white/50">Tip:</strong> &ldquo;auto&rdquo; values use each model family&apos;s built-in defaults
          — GLM-Image (50 steps / CFG 5.0), Qwen-Image (40 steps / CFG 4.0), and generic diffusion models (4 steps / CFG 0.0).
          Override them here to experiment with quality vs speed trade-offs.
        </div>
      </div>
    </div>
  )
}
