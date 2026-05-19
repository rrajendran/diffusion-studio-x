import { useState, useEffect, useCallback, useRef } from 'react'
import { loadChats, saveChats, loadActiveId, saveActiveId, newChatObj } from '../store/chatStore.js'
import { generateImage } from '../providers/index.js'
import { OLLAMA, bridgeLog } from '../lib/ports.js'

export function useChat(config) {
  const [chats, setChats] = useState([])
  const [activeChatId, setActiveChatIdInner] = useState(null)
  const [loaded, setLoaded] = useState(false)
  const chatsRef = useRef(chats)
  useEffect(() => { chatsRef.current = chats }, [chats])

  // Load chats and active ID on mount
  useEffect(() => {
    loadChats().then(savedChats => {
      setChats(savedChats)
      const saved = loadActiveId()
      setActiveChatIdInner((saved && savedChats.find(c => c.id === saved)) ? saved : (savedChats[0]?.id ?? null))
      setLoaded(true)
    })
  }, [])

  const [loading, setLoading] = useState(false)
  const abortRef = useRef(null)

  // persist on every change (skip initial load)
  useEffect(() => { 
    if (loaded) saveChats(chats) 
  }, [chats, loaded])
  useEffect(() => { 
    if (loaded) saveActiveId(activeChatId) 
  }, [activeChatId, loaded])

  const activeChat = chats.find(c => c.id === activeChatId) ?? null

  const setActiveChatId = useCallback((id) => setActiveChatIdInner(id), [])

  const createChat = useCallback(() => {
    const chat = newChatObj(config)
    setChats(prev => [chat, ...prev])
    setActiveChatIdInner(chat.id)
    return chat.id
  }, [config])

  const deleteChat = useCallback((id) => {
    setChats(prev => {
      const next = prev.filter(c => c.id !== id)
      if (id === activeChatId) {
        setActiveChatIdInner(next[0]?.id ?? null)
      }
      return next
    })
  }, [activeChatId])

  const renameChat = useCallback((id, title) => {
    setChats(prev => prev.map(c => c.id === id ? { ...c, title: title.trim() || c.title } : c))
  }, [])

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const sendMessage = useCallback(async (prompt, aspectRatio, referenceImageUrl = null) => {
    // create a chat on first message if none active
    let targetId = activeChatId
    if (!targetId) {
      const chat = newChatObj(config)
      targetId = chat.id
      setChats(prev => [chat, ...prev])
      setActiveChatIdInner(chat.id)
    }

    // capture last generated image to use as edit context (read from ref — safe outside updater)
    const currentChat = chatsRef.current.find(c => c.id === targetId)
    const lastImageUrl = currentChat
      ? ([...currentChat.messages].reverse().find(m => m.role === 'assistant' && m.imageUrl)?.imageUrl ?? null)
      : null

    setChats(prev => prev.map(c => {
      if (c.id !== targetId) return c
      return {
        ...c,
        title: c.title === 'New Chat'
          ? prompt.slice(0, 45) + (prompt.length > 45 ? '…' : '')
          : c.title,
        messages: [...c.messages, { id: `msg_${Date.now()}`, role: 'user', content: prompt, referenceImageUrl }],
      }
    }))

    const controller = new AbortController()
    abortRef.current = controller
    const signal = controller.signal

    setLoading(true)
    try {
      const isHuggingFace = config.provider === 'huggingface'

      // HuggingFace: send the prompt directly to image/video generation — no LLM intermediary
      if (isHuggingFace) {
        const genRes = await generateImage(prompt, config, lastImageUrl, aspectRatio, signal, referenceImageUrl)
        setChats(prev => prev.map(c =>
          c.id === targetId
            ? { ...c, messages: [...c.messages, { id: `msg_${Date.now()}`, role: 'assistant', content: prompt, imageUrl: genRes.imageUrl ?? null, videoUrl: genRes.videoUrl ?? null, meta: genRes.meta ?? null }] }
            : c
        ))
        setLoading(false)
        return
      }

      // Other providers: route through conversational LLM to extract imagePrompt
      const chatHistory = currentChat ? [...currentChat.messages] : []
      const historyStr = chatHistory.length > 0
        ? chatHistory.map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`).join('\n')
        : 'No history.'

      const refImageNote = referenceImageUrl
        ? '\nNote: The user has attached a reference image. Incorporate it as visual inspiration or as a subject to include in the generated image.'
        : ''

      const systemPrompt = `You are an AI assistant that manages an image generation tool.
Based on the user's conversational input and history, your task is to respond and generate a detailed prompt for a text-to-image model.
If the user asks for a new image or asks you to edit the previous one (e.g. "make it blue"), construct a full, self-contained text-to-image prompt (e.g. "A red car" -> "A blue car").
Return ONLY a JSON object with this exact structure (no markdown, no extra text):
{
  "response": "Your conversational text response to the user",
  "imagePrompt": "The full, self-contained text-to-image prompt, or null if no image is requested"
}${refImageNote}`

      const llmMessages = [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: `History:\n${historyStr}\n\nCurrent User Input: ${prompt}` },
      ]

      const ollamaBase = (config.ollamaBaseUrl || OLLAMA).replace(/\/$/, '')
      const llmUrl = `${ollamaBase}/api/chat`
      const llmModel = config.llmModel || 'llama3'
      bridgeLog(`useChat: LLM call → ${llmUrl} model=${llmModel}`)

      // 90-second timeout for the LLM routing step — image generation itself
      // has its own separate timeout on the bridge side.
      const llmAbort = new AbortController()
      const llmTimer = setTimeout(() => llmAbort.abort(), 90_000)
      // Combine with the user-cancel signal
      signal?.addEventListener('abort', () => llmAbort.abort(), { once: true })

      let llmRes
      try {
        llmRes = await fetch(llmUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: llmModel,
            messages: llmMessages,
            max_tokens: 512,
            stream: false,
            // No format:'json' — it causes many models (gemma, mistral, etc.) to
            // hang indefinitely. We extract JSON from free-form text instead.
          }),
          signal: llmAbort.signal,
        })
      } catch (fetchErr) {
        clearTimeout(llmTimer)
        if (fetchErr.name === 'AbortError') {
          if (signal?.aborted) throw fetchErr  // user cancelled — bubble up
          throw new Error(`LLM timed out after 90s — model '${llmModel}' may be too slow or not running`)
        }
        throw fetchErr
      }
      clearTimeout(llmTimer)

      bridgeLog(`useChat: LLM response status=${llmRes.status} ok=${llmRes.ok}`)

      if (!llmRes.ok) {
        throw new Error(`Conversational LLM Error: ${llmRes.status}. Is Ollama running with model '${llmModel}'?`)
      }

      const llmData = await llmRes.json()
      const rawContent = llmData.message?.content ?? llmData.choices?.[0]?.message?.content ?? ''
      bridgeLog(`useChat: LLM raw content: ${rawContent.slice(0, 300)}`)

      let parsed = { response: 'I could not parse my own output.', imagePrompt: null }

      // Try strict JSON parse first, then extract the first {...} block from free-form text.
      try {
        parsed = JSON.parse(rawContent)
      } catch {
        const jsonMatch = rawContent.match(/\{[\s\S]*\}/)
        if (jsonMatch) {
          try { parsed = JSON.parse(jsonMatch[0]) } catch { /* fall through */ }
        }
        if (!parsed.imagePrompt) {
          // Model didn't return structured JSON — treat whole response as conversational
          // and fall back to using the original user prompt as the image prompt
          parsed.response = rawContent
          parsed.imagePrompt = prompt
          bridgeLog(`useChat: no JSON from LLM — using raw user prompt as imagePrompt`, 'warn')
        }
      }

      bridgeLog(`useChat: parsed.imagePrompt=${JSON.stringify(parsed.imagePrompt?.slice(0, 100))}`)

      let imageUrl = null
      let imageMeta = null
      if (parsed.imagePrompt) {
        bridgeLog(`useChat: calling generateImage provider=${config.provider} model=${config.model}`)
        const genRes = await generateImage(parsed.imagePrompt, config, lastImageUrl, aspectRatio, signal, referenceImageUrl)
        imageUrl = genRes.imageUrl
        imageMeta = genRes.meta ?? null
        bridgeLog(`useChat: generateImage done imageUrl=${imageUrl}`)
      } else {
        bridgeLog(`useChat: imagePrompt is null/empty — no image generation`, 'warn')
      }

      setChats(prev => prev.map(c =>
        c.id === targetId
          ? { ...c, messages: [...c.messages, { id: `msg_${Date.now()}`, role: 'assistant', content: parsed.response, imageUrl, imagePrompt: parsed.imagePrompt ?? null, meta: imageMeta }] }
          : c
      ))
      setLoading(false)
    } catch (err) {
      if (err.name === 'AbortError') {
        setLoading(false)
        return
      }
      setChats(prev => prev.map(c =>
        c.id === targetId
          ? { ...c, messages: [...c.messages, { id: `msg_${Date.now()}`, role: 'assistant', content: err.message, error: true }] }
          : c
      ))
      setLoading(false)
    }
  }, [activeChatId, config])

  return { chats, activeChatId, activeChat, loading, setActiveChatId, createChat, deleteChat, renameChat, sendMessage, stopGeneration }
}
