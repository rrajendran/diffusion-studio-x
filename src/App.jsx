import { useState, useCallback, useEffect } from 'react'
import { ToastProvider } from './components/Toast.jsx'
import Sidebar from './components/Sidebar.jsx'
import ChatWindow from './components/ChatWindow.jsx'
import ChatInput from './components/ChatInput.jsx'
import ProviderBadge from './components/ProviderBadge.jsx'
import ImagePreviewPanel from './components/ImagePreviewPanel.jsx'
import SettingsPage from './components/SettingsPage.jsx'
import GalleryPage from './components/GalleryPage.jsx'
import Lightbox from './components/Lightbox.jsx'
import { useChat } from './hooks/useChat.js'
import { useModels } from './hooks/useModels.js'
import { loadConfig, saveConfig } from './store/chatStore.js'
import { FlagsProvider } from 'react-feature-flags'
import { getFeatureFlags } from './lib/featureFlags.js'
import { BRIDGE, LMSTUDIO, OLLAMA } from './lib/ports.js'

const PROVIDER_DEFAULTS = {
  huggingface: 'stabilityai/stable-diffusion-xl-base-1.0',
  ollama:      'stable-diffusion',
  lmstudio:   '',
  llamacpp:   '',
}

export default function App() {
  const [config, setConfig] = useState(() => {
    const saved = loadConfig()
    return {
      provider:        saved.provider        ?? 'huggingface',
      model:           saved.model           ?? 'stabilityai/stable-diffusion-xl-base-1.0',
      apiKey:          saved.apiKey          ?? '',
      llmModel:        saved.llmModel        ?? 'llama3',
      randomSeed:      saved.randomSeed      ?? true,
      seed:            saved.seed            ?? 42,
      inferenceSteps:  saved.inferenceSteps  ?? 4,
      guidanceScale:   saved.guidanceScale   ?? 0.0,
      hfBaseUrl:       saved.hfBaseUrl       ?? '',
      ollamaBaseUrl:   saved.ollamaBaseUrl   ?? '',
      lmstudioBaseUrl: saved.lmstudioBaseUrl ?? '',
      llamacppBaseUrl: saved.llamacppBaseUrl ?? '',
      maxGalleryItems: saved.maxGalleryItems ?? 100,
      outputPath:      saved.outputPath      ?? '',
      numFrames:       saved.numFrames       ?? 17,
      fps:             saved.fps             ?? 16,
    }
  })

  const { imageModels, llmModels, loadingModels, modelCapabilities } = useModels(config.provider, config)

  // Auto-pick first available when the list loads and current selection isn't in it
  useEffect(() => {
    if (imageModels.length > 0 && !imageModels.includes(config.model)) {
      setConfig(prev => { const next = { ...prev, model: imageModels[0] }; saveConfig(next); return next })
    }
  }, [imageModels]) // eslint-disable-line react-hooks/exhaustive-deps

  // Preload the selected model into the image server as soon as the model list is
  // confirmed (so it's warm before the user sends their first prompt).
  useEffect(() => {
    if (config.provider !== 'huggingface' || !config.model || loadingModels) return
    fetch(`${BRIDGE}/api/image/preload`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: config.model }),
    }).catch(() => {/* non-fatal */})
  }, [config.model, config.provider, loadingModels]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (llmModels.length > 0 && !llmModels.includes(config.llmModel)) {
      setConfig(prev => { const next = { ...prev, llmModel: llmModels[0] }; saveConfig(next); return next })
    }
  }, [llmModels]) // eslint-disable-line react-hooks/exhaustive-deps

  const { chats, activeChatId, activeChat, loading, setActiveChatId, createChat, deleteChat, renameChat, sendMessage, stopGeneration } = useChat(config)
  const [preview, setPreview] = useState(null)
  const [editPrefill, setEditPrefill] = useState(null)
  const [theme, setTheme] = useState(() => localStorage.getItem('img-gen-theme') ?? 'dark')
  const [showSettings, setShowSettings] = useState(false)
  const [showGallery, setShowGallery] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => localStorage.getItem('sidebar-collapsed') === 'true')
  const [sidebarWidth, setSidebarWidth] = useState(() => parseInt(localStorage.getItem('sidebar-width') ?? '256', 10))

  const startResize = useCallback((e) => {
    e.preventDefault()
    const startX = e.clientX
    const startW = sidebarWidth
    function onMove(ev) {
      const newW = Math.min(400, Math.max(160, startW + ev.clientX - startX))
      setSidebarWidth(newW)
    }
    function onUp() {
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
      setSidebarWidth(w => { localStorage.setItem('sidebar-width', String(w)); return w })
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
  }, [sidebarWidth])

  const toggleCollapse = useCallback(() => {
    setSidebarCollapsed(c => { const next = !c; localStorage.setItem('sidebar-collapsed', String(next)); return next })
  }, [])

  // Listen for native menu "Preferences" event emitted by Tauri Rust backend
  useEffect(() => {
    let unlisten = null
    // Only available inside Tauri — guard against browser dev mode
    if (window.__TAURI__) {
      import('@tauri-apps/api/event').then(({ listen }) => {
        listen('open-preferences', () => {
          setShowSettings(true)
          setShowGallery(false)
        }).then(fn => { unlisten = fn })
      })
    }
    return () => { if (unlisten) unlisten() }
  }, [])

  const toggleTheme = useCallback(() => {
    setTheme(t => {
      const next = t === 'dark' ? 'light' : 'dark'
      localStorage.setItem('img-gen-theme', next)
      return next
    })
  }, [])

  const handleConfigChange = useCallback((cfg) => {
    setConfig(prev => {
      // When provider switches, reset models to defaults so the new list loads fresh
      const next = prev.provider !== cfg.provider
        ? { ...cfg, model: PROVIDER_DEFAULTS[cfg.provider] ?? '', llmModel: '' }
        : cfg
      saveConfig(next)
      return next
    })
  }, [])

  const handleImageClick = useCallback((mediaUrl, prompt, meta, type) => {
    if (type === 'video') {
      setPreview({ videoUrl: mediaUrl, prompt, meta: meta ?? null, type: 'video' })
    } else {
      setPreview({ imageUrl: mediaUrl, prompt, meta: meta ?? null, type: 'image' })
    }
  }, [])

  const handleRewrite = useCallback(async (text) => {
    const isHuggingFace = config.provider === 'huggingface'
    const isLmStudio    = config.provider === 'lmstudio'

    if (isLmStudio && (llmModels.length === 0 || !config.llmModel || !llmModels.includes(config.llmModel))) {
      throw new Error('No LLM model selected — pick a text LLM in the toolbar or load one in LM Studio')
    }

    const systemPrompt = `You are an expert prompt engineer for an image generation model.
Revise the following text into a highly descriptive, detailed prompt suitable for a text-to-image model.
Respond with ONLY the new prompt string, without any conversational filler or quotes.`

    const ollamaBase = (config.ollamaBaseUrl || OLLAMA).replace(/\/$/, '')
    const lmBase     = (config.lmstudioBaseUrl || LMSTUDIO).replace(/\/$/, '')

    let content

    if (isLmStudio) {
      // LM Studio new REST API — POST /api/v1/chat
      const llmRes = await fetch(`${lmBase}/api/v1/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model:             config.llmModel,
          system_prompt:     systemPrompt,
          input:             text,
          max_output_tokens: 512,
          stream:            false,
        }),
      })
      if (!llmRes.ok) {
        const errText = await llmRes.text()
        throw new Error(`Enhance failed (${llmRes.status}): ${errText.slice(0, 200)}`)
      }
      const data = await llmRes.json()
      // output is an array of message objects: [{ role, content }]
      const item = data.output?.[0]
      content = typeof item?.content === 'string' ? item.content : item?.content?.[0]?.text
    } else {
      // OpenAI-compatible chat completions (HuggingFace bridge, Ollama)
      const url = isHuggingFace ? `${BRIDGE}/api/hf/v1/chat/completions` : `${ollamaBase}/api/chat`
      const messages = isHuggingFace
        ? [{ role: 'user', content: `${systemPrompt}\n\n${text}` }]
        : [{ role: 'system', content: systemPrompt }, { role: 'user', content: text }]
      const llmRes = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model:      config.llmModel || (isHuggingFace ? 'default' : 'llama3'),
          messages,
          max_tokens: 512,
          stream:     false,
        }),
      })
      if (!llmRes.ok) {
        const errText = await llmRes.text()
        throw new Error(`Enhance failed (${llmRes.status}): ${errText.slice(0, 200)}`)
      }
      const raw = await llmRes.text()
      const lastLine = raw.trim().split('\n').filter(l => l.startsWith('data:') ? (l = l.slice(5).trim()) : l).at(-1) ?? raw
      let llmData
      try { llmData = JSON.parse(lastLine) } catch { llmData = JSON.parse(raw) }
      content = llmData.message?.content ?? llmData.choices?.[0]?.message?.content
    }

    if (!content) throw new Error('Enhance: no content in response')

    try {
      const parsed = JSON.parse(content.trim())
      const extracted = parsed.imagePrompt ?? parsed.prompt ?? parsed.enhanced_prompt ?? parsed.result
      if (typeof extracted === 'string' && extracted.trim()) return extracted.trim()
    } catch { /* not JSON — use as-is */ }

    return content.trim()
  }, [config.provider, config.llmModel, config.ollamaBaseUrl, config.lmstudioBaseUrl, llmModels])

  const bgClass = theme === 'dark'
    ? 'dsx-bg-dark'
    : 'dsx-bg-light'

  return (
    <FlagsProvider value={getFeatureFlags()}>
    <ToastProvider>
    <div className={`theme-${theme} h-screen ${bgClass} flex overflow-hidden`}>
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        onSelectChat={id => { setActiveChatId(id); setShowSettings(false); setShowGallery(false) }}
        onNewChat={() => { createChat(); setShowSettings(false); setShowGallery(false) }}
        onDeleteChat={deleteChat}
        onRenameChat={renameChat}
        showSettings={showSettings}
        onToggleSettings={() => { setShowSettings(true); setShowGallery(false) }}
        showGallery={showGallery}
        onToggleGallery={() => { setShowGallery(true); setShowSettings(false) }}
        collapsed={sidebarCollapsed}
        onToggleCollapse={toggleCollapse}
        sidebarWidth={sidebarWidth}
      />

      {/* Drag handle for sidebar resize */}
      {!sidebarCollapsed && (
        <div
          className="w-1 cursor-col-resize hover:bg-white/20 flex-shrink-0 select-none transition-colors"
          onMouseDown={startResize}
        />
      )}

      <div className="flex-1 flex overflow-hidden">
        <main className="flex-1 flex flex-col m-3 ml-0 glass overflow-hidden">
          <div className="flex items-center justify-between px-5 py-4 border-b border-white/10 flex-shrink-0">
            <div className="flex items-center gap-2">
              <span className="text-white font-medium text-sm truncate max-w-xs">
                {showSettings ? 'Settings' : showGallery ? 'Gallery' : (activeChat?.title ?? 'Diffusion Studio X')}
              </span>
              {!showSettings && !showGallery && <ProviderBadge provider={config.provider} />}
            </div>
            <button
              onClick={toggleTheme}
              title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              className="glass-btn px-2.5 py-1.5 text-white text-sm"
            >
              {theme === 'dark' ? '☀️' : '🌙'}
            </button>
          </div>

          {showSettings ? (
            <SettingsPage
              config={config}
              onConfigChange={handleConfigChange}
              imageModels={imageModels}
              llmModels={llmModels}
              loadingModels={loadingModels}
              onClose={() => setShowSettings(false)}
            />
          ) : showGallery ? (
            <GalleryPage
              chats={chats}
              maxGalleryItems={config.maxGalleryItems}
              theme={theme}
              onNavigateChat={(chatId) => {
                setActiveChatId(chatId)
                setShowGallery(false)
              }}
            />
          ) : activeChat ? (
            <>
              <ChatWindow messages={activeChat.messages} loading={loading} onImageClick={handleImageClick} onEdit={setEditPrefill} theme={theme} />
              <ChatInput
                onSend={(prompt, ratio, refImg) => { setEditPrefill(null); sendMessage(prompt, ratio, refImg) }}
                loading={loading}
                onStop={stopGeneration}
                onRewrite={handleRewrite}
                config={config}
                onConfigChange={handleConfigChange}
                imageModels={imageModels}
                llmModels={llmModels}
                loadingModels={loadingModels}
                modelCapabilities={modelCapabilities}
                prefill={editPrefill}
                theme={theme}
              />
            </>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center gap-5">
              <img src="/logo.png" alt="Diffusion Studio X" className="h-48 w-auto opacity-70" />
              <div className="text-center">
                <p className="text-white font-bold text-xl tracking-tight mb-1">Diffusion Studio <span className="dsx-x">X</span></p>
                <p className="text-white/30 text-sm">Click <strong className="text-white/50">+ New Chat</strong> to get started</p>
              </div>
            </div>
          )}
        </main>

        <div className="flex-shrink-0" />
      </div>

      {/* Lightbox — rendered as portal over everything */}
      {preview && (
        <Lightbox
          images={[{ imageUrl: preview.imageUrl, videoUrl: preview.videoUrl, type: preview.type, prompt: preview.prompt, meta: preview.meta }]}
          index={0}
          onIndex={() => {}}
          onClose={() => setPreview(null)}
          theme={theme}
        />
      )}
    </div>
    </ToastProvider>
    </FlagsProvider>
  )
}
