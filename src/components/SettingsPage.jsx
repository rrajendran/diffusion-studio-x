import { useState, useEffect } from 'react'
import { ASPECT_RATIOS, DEFAULT_RATIO } from '../config/aspectRatios.js'
import { useServerStatus } from '../hooks/useServerStatus.js'

const PROVIDERS = [
  { value: 'huggingface', label: 'Hugging Face', desc: 'External diffusers image server + HF Hub models' },
  { value: 'ollama',      label: 'Ollama',        desc: 'Local or remote Ollama server' },
  { value: 'lmstudio',   label: 'LM Studio',     desc: 'Local LM Studio server (port 1234)' },
  { value: 'llamacpp',   label: 'llama.cpp',     desc: 'Local llama.cpp server (port 8080)' },
]

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

function SliderRow({ label, hint, value, min, max, step = 1, onChange, displayValue }) {
  return (
    <div className="px-4 py-3">
      <div className="flex items-center justify-between mb-2">
        <div>
          <p className="text-white text-sm">{label}</p>
          {hint && <p className="text-white/35 text-xs mt-0.5">{hint}</p>}
        </div>
        <span className="text-white/70 text-sm font-mono tabular-nums">{displayValue ?? value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
        style={{ accentColor: 'rgba(255,255,255,0.7)' }}
      />
      <div className="flex justify-between mt-1">
        <span className="text-white/25 text-[10px]">{min}</span>
        <span className="text-white/25 text-[10px]">{max}</span>
      </div>
    </div>
  )
}

function SelectField({ value, onChange, options, placeholder }) {
  return (
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      className="bg-white/10 border border-white/20 rounded-xl text-white text-sm px-3 py-1.5 outline-none cursor-pointer max-w-[220px] truncate"
    >
      {placeholder && <option value="" className="bg-gray-900">{placeholder}</option>}
      {options.map(o => (
        <option key={typeof o === 'string' ? o : o.value} value={typeof o === 'string' ? o : o.value} className="bg-gray-900">
          {typeof o === 'string' ? o : o.label}
        </option>
      ))}
    </select>
  )
}

function Toggle({ checked, onChange }) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
        checked ? 'bg-white/40' : 'bg-white/15'
      } border border-white/20`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
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
export default function SettingsPage({ config, onConfigChange, imageModels, llmModels, loadingModels, onClose }) {
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

  const provider = PROVIDERS.find(p => p.value === local.provider)

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
          <Row label="Active provider" hint={provider?.desc}>
            <SelectField
              value={local.provider}
              onChange={val => set('provider', val)}
              options={PROVIDERS}
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

        {/* Service URLs */}
        <Section title="Service URLs">
          <div className="px-4 py-3">
            <div className="flex items-center gap-2 mb-1">
              <p className="text-white text-sm">HF image server</p>
              <StatusDot up={serverStatus.hfImages} />
            </div>
            <p className="text-white/35 text-xs mb-2">Base URL of the diffusers image server you run separately</p>
            <TextInput
              value={local.hfBaseUrl ?? 'http://127.0.0.1:8001'}
              onChange={val => set('hfBaseUrl', val)}
              placeholder="http://127.0.0.1:8001"
              monospace
            />
          </div>
          <div className="px-4 py-3">
            <div className="flex items-center gap-2 mb-1">
              <p className="text-white text-sm">Ollama base URL</p>
              <StatusDot up={serverStatus.ollama} />
            </div>
            <p className="text-white/35 text-xs mb-2">Change if Ollama runs on a different host or port</p>
            <TextInput
              value={local.ollamaBaseUrl ?? 'http://127.0.0.1:11434'}
              onChange={val => set('ollamaBaseUrl', val)}
              placeholder="http://127.0.0.1:11434"
              monospace
            />
          </div>
        </Section>

        {/* Models */}
        <Section title="Models">
          <Row
            label="Image model"
            hint={loadingModels ? 'Loading available models…' : `${imageModels.length} model${imageModels.length !== 1 ? 's' : ''} available`}
          >
            {imageModels.length > 0 ? (
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
          />
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
