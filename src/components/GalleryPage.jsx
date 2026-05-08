import { useState, useMemo, useEffect, useCallback } from 'react'
import Lightbox from './Lightbox.jsx'
import { BRIDGE } from '../lib/ports.js'

/**
 * Shows all images from the output directory on disk (via /api/gallery),
 * enriched with chat context from the chats prop. Clicking a card opens the
 * Lightbox; a "Go to chat →" button appears when a matching chat exists.
 *
 * Props:
 *   chats            – full chats array from useChat
 *   onNavigateChat   – (chatId: string) => void — called when user wants to open a chat
 */
export default function GalleryPage({ chats, onNavigateChat }) {
  const [galleryItems, setGalleryItems] = useState([])
  const [loadError, setLoadError] = useState(null)
  const [lightboxIndex, setLightboxIndex] = useState(null)

  // Fetch the output directory listing from the bridge server
  const fetchGallery = useCallback(async () => {
    try {
      const res = await fetch(`${BRIDGE}/api/gallery`)
      if (!res.ok) throw new Error(`Gallery fetch failed: ${res.status}`)
      const data = await res.json()
      setGalleryItems(data)
      setLoadError(null)
    } catch (err) {
      setLoadError(err.message)
    }
  }, [])

  useEffect(() => { fetchGallery() }, [fetchGallery])

  // Build a lookup: filename → { chatId, chatTitle } by matching imageUrl tails
  const chatByFilename = useMemo(() => {
    const map = new Map()
    for (const chat of chats) {
      for (const msg of chat.messages) {
        if (msg.role === 'assistant' && msg.imageUrl) {
          // imageUrl may be a full URL like http://localhost:3001/output/foo.jpg
          // or a relative /output/foo.jpg path
          const filename = msg.imageUrl.split('/output/')[1]
          if (filename) map.set(filename, { chatId: chat.id, chatTitle: chat.title })
        }
      }
    }
    return map
  }, [chats])

  // Enrich gallery items with chat context and build lightbox image list
  const images = useMemo(() =>
    galleryItems.map(item => ({
      ...item,
      imageUrl: `${BRIDGE}${item.url}`,
      chatInfo: chatByFilename.get(item.filename) ?? null,
    })),
    [galleryItems, chatByFilename]
  )

  const openAt  = (i) => setLightboxIndex(i)
  const close   = () => setLightboxIndex(null)

  if (loadError) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-3 text-white/40 text-sm">
        <span className="text-4xl">⚠️</span>
        <p>Could not load gallery: {loadError}</p>
        <button onClick={fetchGallery} className="glass-btn px-3 py-1.5 text-white/70 text-xs">Retry</button>
      </div>
    )
  }

  if (images.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-3 text-white/30 text-sm">
        <span className="text-5xl">🖼️</span>
        <p>No images yet — generate some!</p>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin p-5">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-center justify-between mb-4">
          <p className="text-white/35 text-xs">{images.length} image{images.length !== 1 ? 's' : ''}</p>
          <button onClick={fetchGallery} title="Refresh gallery" className="glass-btn px-2.5 py-1 text-white/50 hover:text-white text-xs">↻ refresh</button>
        </div>

        {/* Responsive masonry grid */}
        <div className="columns-2 sm:columns-3 lg:columns-4 gap-3 space-y-3">
          {images.map((img, i) => (
            <GalleryThumbnail
              key={img.filename}
              img={img}
              onClick={() => openAt(i)}
              onGoToChat={img.chatInfo && onNavigateChat
                ? () => onNavigateChat(img.chatInfo.chatId)
                : null}
            />
          ))}
        </div>
      </div>

      {lightboxIndex !== null && (
        <Lightbox
          images={images}
          index={lightboxIndex}
          onIndex={setLightboxIndex}
          onClose={close}
        />
      )}
    </div>
  )
}

function GalleryThumbnail({ img, onClick, onGoToChat }) {
  const [hovered, setHovered] = useState(false)

  return (
    <div
      className="break-inside-avoid relative rounded-xl overflow-hidden cursor-pointer group mb-3"
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <img
        src={img.imageUrl}
        alt={img.prompt}
        className="w-full block rounded-xl transition-transform duration-200 group-hover:scale-[1.02]"
        loading="lazy"
      />

      {hovered && (
        <div
          className="absolute inset-0 rounded-xl flex flex-col justify-end p-2.5"
          style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.75) 0%, transparent 60%)' }}
        >
          {img.prompt && (
            <p className="text-white text-[11px] leading-tight line-clamp-3 mb-1">
              {img.prompt}
            </p>
          )}
          <div className="flex items-center justify-between gap-1 mt-0.5">
            {img.chatInfo ? (
              <p className="text-white/50 text-[10px] truncate flex-1">{img.chatInfo.chatTitle}</p>
            ) : (
              <span className="flex-1" />
            )}
            {onGoToChat && (
              <button
                onClick={(e) => { e.stopPropagation(); onGoToChat() }}
                className="glass-btn px-2 py-0.5 text-white/80 hover:text-white text-[10px] whitespace-nowrap flex-shrink-0"
              >
                Go to chat →
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

