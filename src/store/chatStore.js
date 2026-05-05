const CHATS_KEY  = 'img-gen-chats'
const ACTIVE_KEY = 'img-gen-active'
const CONFIG_KEY = 'img-gen-config'
const DB_NAME = 'img-gen-db'
const STORE_NAME = 'images'

// Initialize IndexedDB for storing images
let db = null
async function initDB() {
  if (db) return db
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1)
    req.onerror = () => reject(req.error)
    req.onsuccess = () => {
      db = req.result
      resolve(db)
    }
    req.onupgradeneeded = (e) => {
      const store = e.target.result.createObjectStore(STORE_NAME, { keyPath: 'id' })
      store.createIndex('messageId', 'messageId', { unique: true })
    }
  })
}

// Save image to IndexedDB
async function saveImage(messageId, imageUrl) {
  try {
    const database = await initDB()
    return new Promise((resolve, reject) => {
      const tx = database.transaction([STORE_NAME], 'readwrite')
      const store = tx.objectStore(STORE_NAME)
      store.put({ id: messageId, messageId, imageUrl, savedAt: Date.now() })
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
    })
  } catch (err) {
    console.warn('Failed to save image to IndexedDB:', err)
  }
}

// Load image from IndexedDB
async function loadImage(messageId) {
  try {
    const database = await initDB()
    return new Promise((resolve) => {
      const tx = database.transaction([STORE_NAME], 'readonly')
      const store = tx.objectStore(STORE_NAME)
      const index = store.index('messageId')
      const req = index.get(messageId)
      req.onsuccess = () => resolve(req.result?.imageUrl ?? null)
      req.onerror = () => resolve(null)
    })
  } catch {
    return null
  }
}

// Load all images for a chat
async function loadChatImages(chatId, messages) {
  const imageMap = {}
  await Promise.all(
    messages
      .filter(m => m.role === 'assistant')
      .map(async (m) => {
        const url = await loadImage(m.id)
        if (url) imageMap[m.id] = url
      })
  )
  return imageMap
}

export async function loadChats() {
  try {
    const chats = JSON.parse(localStorage.getItem(CHATS_KEY)) ?? []
    // Restore images from IndexedDB
    await Promise.all(
      chats.map(async (chat) => {
        const imageMap = await loadChatImages(chat.id, chat.messages)
        chat.messages.forEach(msg => {
          if (imageMap[msg.id]) msg.imageUrl = imageMap[msg.id]
        })
      })
    )
    return chats
  } catch {
    return []
  }
}
export async function saveChats(chats) {
  // Save images to IndexedDB before stripping them from localStorage
  await Promise.all(
    chats.flatMap(chat =>
      chat.messages
        .filter(m => m.imageUrl)
        .map(m => saveImage(m.id, m.imageUrl))
    )
  )

  // Strip imageUrl from messages for localStorage
  const chatsToSave = chats.map(chat => ({
    ...chat,
    messages: chat.messages.map(({ imageUrl, ...msg }) => msg)
  }))
  try {
    localStorage.setItem(CHATS_KEY, JSON.stringify(chatsToSave))
  } catch (err) {
    if (err.name === 'QuotaExceededError') {
      console.warn('localStorage quota exceeded, clearing old chats')
      const recentChats = chatsToSave.slice(0, Math.floor(chatsToSave.length / 2))
      try {
        localStorage.setItem(CHATS_KEY, JSON.stringify(recentChats))
      } catch {
        console.error('Failed to save chats to localStorage')
      }
    } else {
      throw err
    }
  }
}
export function loadActiveId() {
  return localStorage.getItem(ACTIVE_KEY) ?? null
}
export function saveActiveId(id) {
  if (id) localStorage.setItem(ACTIVE_KEY, id)
  else localStorage.removeItem(ACTIVE_KEY)
}
export function loadConfig() {
  try { return JSON.parse(localStorage.getItem(CONFIG_KEY)) ?? {} } catch { return {} }
}
export function saveConfig(cfg) {
  localStorage.setItem(CONFIG_KEY, JSON.stringify(cfg))
}
export function newChatObj(config) {
  return {
    id: `chat_${Date.now()}`,
    title: 'New Chat',
    createdAt: Date.now(),
    messages: [],
    provider: config.provider ?? 'huggingface',
    model: config.model ?? '',
    llmModel: config.llmModel ?? 'llama3',
  }
}
