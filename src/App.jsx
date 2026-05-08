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

  const { chats, activeChatId, activeChat, loading, setActiveChatId, createChat, deleteChat, sendMessage, stopGeneration } = useChat(config)
  const [preview, setPreview] = useState(null)   // { imageUrl, prompt, meta? } | null
  const [editPrefill, setEditPrefill] = useState(null)  // message to prefill ChatInput for editing
  const [theme, setTheme] = useState(() => localStorage.getItem('img-gen-theme') ?? 'dark')
  const [showSettings, setShowSettings] = useState(false)
  const [showGallery, setShowGallery] = useState(false)

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

  const handleImageClick = useCallback((imageUrl, prompt, meta) => setPreview({ imageUrl, prompt, meta: meta ?? null }), [])

  const handleRewrite = useCallback(async (text) => {
    const systemPrompt = `You are an expert prompt engineer for an image generation model.
Revise the following text into a highly descriptive, detailed prompt suitable for a text-to-image model.
Respond with ONLY the new prompt string, without any conversational filler or quotes.`

    const isHuggingFace = config.provider === 'huggingface'
    const isLmStudio = config.provider === 'lmstudio'
    const ollamaBase = (config.ollamaBaseUrl || OLLAMA).replace(/\/$/, '')
    const url = isHuggingFace
      ? `${BRIDGE}/api/hf/v1/chat/completions`
      : isLmStudio
        ? `${LMSTUDIO}/v1/chat/completions`
        : `${ollamaBase}/api/chat`

    const messages = (isHuggingFace || isLmStudio)
      ? [{ role: 'user', content: `${systemPrompt}\n\n${text}` }]
      : [{ role: 'system', content: systemPrompt }, { role: 'user', content: text }]

    const llmRes = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: config.llmModel || (isHuggingFace || isLmStudio ? 'default' : 'llama3'),
        messages,
        max_tokens: 512,
        stream: false,
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

    const content = llmData.message?.content ?? llmData.choices?.[0]?.message?.content
    if (!content) throw new Error('Enhance: no content in response')

    try {
      const parsed = JSON.parse(content.trim())
      const extracted = parsed.imagePrompt ?? parsed.prompt ?? parsed.enhanced_prompt ?? parsed.result
      if (typeof extracted === 'string' && extracted.trim()) return extracted.trim()
    } catch { /* not JSON — use as-is */ }

    return content.trim()
  }, [config.provider, config.llmModel])

  const bgClass = theme === 'dark'
    ? 'dsx-bg-dark'
    : 'dsx-bg-light'

  return (
    <ToastProvider>
    <div className={`theme-${theme} h-screen ${bgClass} flex overflow-hidden`}>
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        onSelectChat={id => { setActiveChatId(id); setShowSettings(false); setShowGallery(false) }}
        onNewChat={() => { createChat(); setShowSettings(false); setShowGallery(false) }}
        onDeleteChat={deleteChat}
        showSettings={showSettings}
        onToggleSettings={() => { setShowSettings(s => !s); setShowGallery(false) }}
        showGallery={showGallery}
        onToggleGallery={() => { setShowGallery(g => !g); setShowSettings(false) }}
      />

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
              onNavigateChat={(chatId) => {
                setActiveChatId(chatId)
                setShowGallery(false)
              }}
            />
          ) : activeChat ? (
            <>
              <ChatWindow messages={activeChat.messages} loading={loading} onImageClick={handleImageClick} onEdit={setEditPrefill} />
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
          images={[{ imageUrl: preview.imageUrl, prompt: preview.prompt, meta: preview.meta }]}
          index={0}
          onIndex={() => {}}
          onClose={() => setPreview(null)}
        />
      )}
    </div>
    </ToastProvider>
  )
}
