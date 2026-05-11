import { useState, useMemo, useEffect, useCallback } from 'react'
import Lightbox from './Lightbox.jsx'
import { BRIDGE } from '../lib/ports.js'

const PAGE_SIZE = 24

export default function GalleryPage({ chats, onNavigateChat, maxGalleryItems = 500, theme = 'dark' }) {
  const isLight = theme === 'light'

  const [galleryItems, setGalleryItems] = useState([])
  const [loadError, setLoadError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [lightboxIndex, setLightboxIndex] = useState(null)
  const [currentPage, setCurrentPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectionMode, setSelectionMode] = useState(false)
  const [selectedIds, setSelectedIds] = useState(new Set())
  const [showDeleteModal, setShowDeleteModal] = useState(false)

  const fetchGallery = useCallback(async (targetPage = 1) => {
    setLoading(true)
    try {
      const res = await fetch(`${BRIDGE}/api/gallery?page=${targetPage}&limit=${PAGE_SIZE}`)
      if (!res.ok) throw new Error(`Gallery fetch failed: ${res.status}`)
      const data = await res.json()
      setGalleryItems(data)
      if (data.length === PAGE_SIZE) {
        setTotalPages(p => Math.max(p, targetPage + 1))
      } else {
        setTotalPages(targetPage)
      }
      setLoadError(null)
    } catch (err) {
      setLoadError(err.message)
    }
    setLoading(false)
  }, [])

  useEffect(() => { fetchGallery(1) }, [fetchGallery])

  const goToPage = useCallback((p) => {
    if (p < 1 || p > totalPages) return
    setCurrentPage(p)
    setSelectedIds(new Set())
    fetchGallery(p)
  }, [totalPages, fetchGallery])

  const handleRefresh = useCallback(() => {
    setCurrentPage(1)
    setTotalPages(1)
    fetchGallery(1)
  }, [fetchGallery])

  const chatByFilename = useMemo(() => {
    const map = new Map()
    for (const chat of chats) {
      for (const msg of chat.messages) {
        if (msg.role === 'assistant') {
          const mediaUrl = msg.videoUrl || msg.imageUrl
          if (mediaUrl) {
            const filename = mediaUrl.split('/output/')[1]
            if (filename) map.set(filename, { chatId: chat.id, chatTitle: chat.title })
          }
        }
      }
    }
    return map
  }, [chats])

  const allImages = useMemo(() =>
    galleryItems.map(item => ({
      ...item,
      mediaUrl: `${BRIDGE}${item.url}`,
      // Keep imageUrl for lightbox compatibility — use mediaUrl as the canonical URL
      imageUrl: item.type === 'video' ? undefined : `${BRIDGE}${item.url}`,
      videoUrl: item.type === 'video' ? `${BRIDGE}${item.url}` : undefined,
      chatInfo: chatByFilename.get(item.filename) ?? null,
    })),
    [galleryItems, chatByFilename]
  )

  const images = useMemo(() => {
    if (!searchQuery.trim()) return allImages
    const q = searchQuery.toLowerCase()
    return allImages.filter(img =>
      img.filename.toLowerCase().includes(q) ||
      img.prompt?.toLowerCase().includes(q) ||
      img.chatInfo?.chatTitle?.toLowerCase().includes(q)
    )
  }, [allImages, searchQuery])

  const allSelected = images.length > 0 && images.every(img => selectedIds.has(img.filename))

  function toggleSelect(filename) {
    setSelectedIds(prev => {
      const next = new Set(prev)
      next.has(filename) ? next.delete(filename) : next.add(filename)
      return next
    })
  }

  function toggleSelectAll() {
    setSelectedIds(allSelected ? new Set() : new Set(images.map(img => img.filename)))
  }

  function exitSelectionMode() {
    setSelectionMode(false)
    setSelectedIds(new Set())
  }

  const [downloading, setDownloading] = useState(false)

  async function handleDownload() {
    const toDownload = images.filter(img => selectedIds.has(img.filename))
    if (!toDownload.length) return
    setDownloading(true)
    for (const img of toDownload) {
      try {
        const url = img.mediaUrl
        const res = await fetch(url)
        const blob = await res.blob()
        const ext = img.type === 'video' ? 'mp4' : (blob.type === 'image/jpeg' ? 'jpg' : blob.type === 'image/webp' ? 'webp' : 'png')
        const name = img.filename || `media-${Date.now()}.${ext}`
        const objUrl = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = objUrl
        a.download = name
        a.click()
        URL.revokeObjectURL(objUrl)
        await new Promise(r => setTimeout(r, 200))
      } catch { /* skip failed individual downloads */ }
    }
    setDownloading(false)
  }

  async function handleDelete() {
    const filenames = [...selectedIds]
    try {
      await fetch(`${BRIDGE}/api/gallery/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filenames }),
      })
    } catch { /* non-fatal */ }
    setShowDeleteModal(false)
    exitSelectionMode()
    const targetPage = filenames.length >= images.length && currentPage > 1 ? currentPage - 1 : currentPage
    setCurrentPage(targetPage)
    setTotalPages(1)
    fetchGallery(targetPage)
  }

  // ── Glassmorphism token helpers ────────────────────────────
  const glassPanel = {
    background:     isLight ? 'rgba(255,255,255,0.72)' : 'rgba(255,255,255,0.06)',
    border:         isLight ? '1px solid rgba(0,0,0,0.1)' : '1px solid rgba(255,255,255,0.12)',
    backdropFilter: 'blur(16px)',
    WebkitBackdropFilter: 'blur(16px)',
  }

  const glassBtn = (active = false) => ({
    background:     active
      ? (isLight ? 'rgba(0,113,227,0.12)' : 'rgba(34,211,238,0.18)')
      : (isLight ? 'rgba(0,0,0,0.05)' : 'rgba(255,255,255,0.08)'),
    border:         active
      ? (isLight ? '1px solid rgba(0,113,227,0.3)' : '1px solid rgba(34,211,238,0.35)')
      : (isLight ? '1px solid rgba(0,0,0,0.1)' : '1px solid rgba(255,255,255,0.12)'),
    color:          active
      ? (isLight ? '#0071e3' : '#22d3ee')
      : (isLight ? 'rgba(0,0,0,0.5)' : 'rgba(255,255,255,0.55)'),
    backdropFilter: 'blur(8px)',
    WebkitBackdropFilter: 'blur(8px)',
  })

  const mutedText = { color: isLight ? 'rgba(0,0,0,0.38)' : 'rgba(255,255,255,0.35)' }

  const mediaCount = allImages.length
  const videoCount = allImages.filter(i => i.type === 'video').length

  if (loadError) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-3 text-sm" style={mutedText}>
        <span className="text-4xl">⚠️</span>
        <p>Could not load gallery: {loadError}</p>
        <button onClick={handleRefresh} className="glass-btn px-3 py-1.5 text-white/70 text-xs">Retry</button>
      </div>
    )
  }

  if (!loading && allImages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-3 text-sm" style={mutedText}>
        <span className="text-5xl">🖼️</span>
        <p>No media yet — generate some!</p>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin p-5">
      <div className="max-w-5xl mx-auto">

        {/* Toolbar */}
        <div className="flex items-center gap-2 mb-4 flex-wrap">
          <p className="text-xs mr-1" style={mutedText}>
            {mediaCount} item{mediaCount !== 1 ? 's' : ''}{videoCount > 0 ? ` (${videoCount} video${videoCount !== 1 ? 's' : ''})` : ''}
          </p>

          {/* Search */}
          <input
            type="text"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Search gallery…"
            className="flex-1 min-w-[140px] bg-white/10 border border-white/20 rounded-xl text-white text-xs px-3 py-1.5 outline-none placeholder-white/30"
          />

          {selectionMode ? (
            <>
              <button
                onClick={toggleSelectAll}
                className="rounded-xl px-2.5 py-1 text-xs transition-all"
                style={glassBtn(allSelected)}
              >
                {allSelected ? 'Deselect All' : 'Select All'}
              </button>

              <span className="text-xs" style={mutedText}>{selectedIds.size} selected</span>

              <button
                onClick={handleDownload}
                disabled={selectedIds.size === 0 || downloading}
                className="rounded-xl px-2.5 py-1 text-xs disabled:opacity-30 transition-all"
                style={{
                  ...glassBtn(),
                  color: selectedIds.size > 0
                    ? (isLight ? '#0071e3' : '#22d3ee')
                    : undefined,
                }}
                title={downloading ? 'Downloading…' : `Download ${selectedIds.size} item${selectedIds.size !== 1 ? 's' : ''}`}
              >
                {downloading ? '⏳' : '↓'} Download
              </button>

              <button
                onClick={() => selectedIds.size > 0 && setShowDeleteModal(true)}
                disabled={selectedIds.size === 0}
                className="rounded-xl px-2.5 py-1 text-xs disabled:opacity-30 transition-all"
                style={{ ...glassBtn(), color: selectedIds.size > 0 ? '#f87171' : undefined }}
              >
                Delete
              </button>
              <button
                onClick={exitSelectionMode}
                className="rounded-xl px-2.5 py-1 text-xs transition-all"
                style={glassBtn()}
              >
                Cancel
              </button>
            </>
          ) : (
            <>
              <button
                onClick={() => setSelectionMode(true)}
                className="rounded-xl px-2.5 py-1 text-xs transition-all"
                style={glassBtn()}
              >
                Select
              </button>
              <button
                onClick={handleRefresh}
                title="Refresh gallery"
                className="rounded-xl px-2.5 py-1 text-xs transition-all"
                style={glassBtn()}
              >
                ↻
              </button>
            </>
          )}
        </div>

        {/* Grid */}
        {loading ? (
          <div className="flex justify-center items-center py-20">
            <div className="flex gap-1.5">
              {[0, 1, 2].map(i => (
                <span
                  key={i}
                  className="w-2 h-2 rounded-full animate-bounce"
                  style={{
                    background: isLight ? 'rgba(0,0,0,0.3)' : 'rgba(255,255,255,0.4)',
                    animationDelay: `${i * 0.15}s`,
                  }}
                />
              ))}
            </div>
          </div>
        ) : (
          <div className="columns-2 sm:columns-3 lg:columns-4 gap-3 space-y-3">
            {images.map((img, i) => (
              <GalleryThumbnail
                key={img.filename}
                img={img}
                selectionMode={selectionMode}
                selected={selectedIds.has(img.filename)}
                onToggleSelect={() => toggleSelect(img.filename)}
                onClick={() => { if (!selectionMode) setLightboxIndex(i) }}
                onGoToChat={img.chatInfo && onNavigateChat
                  ? () => onNavigateChat(img.chatInfo.chatId)
                  : null}
                isLight={isLight}
              />
            ))}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && !searchQuery && (
          <div className="flex justify-center mt-8">
            <div
              className="flex items-center gap-1 p-1.5 rounded-2xl"
              style={glassPanel}
            >
              <button
                onClick={() => goToPage(currentPage - 1)}
                disabled={currentPage === 1}
                className="flex items-center justify-center w-8 h-8 rounded-xl text-sm transition-all disabled:opacity-30"
                style={glassBtn()}
                title="Previous page"
              >
                ‹
              </button>

              {buildPageNumbers(currentPage, totalPages).map((item, idx) =>
                item === '…' ? (
                  <span
                    key={`ellipsis-${idx}`}
                    className="w-8 h-8 flex items-center justify-center text-xs"
                    style={mutedText}
                  >
                    …
                  </span>
                ) : (
                  <button
                    key={item}
                    onClick={() => goToPage(item)}
                    className="w-8 h-8 flex items-center justify-center rounded-xl text-xs font-medium transition-all"
                    style={glassBtn(item === currentPage)}
                  >
                    {item}
                  </button>
                )
              )}

              <button
                onClick={() => goToPage(currentPage + 1)}
                disabled={currentPage === totalPages}
                className="flex items-center justify-center w-8 h-8 rounded-xl text-sm transition-all disabled:opacity-30"
                style={glassBtn()}
                title="Next page"
              >
                ›
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Lightbox */}
      {lightboxIndex !== null && (
        <Lightbox
          images={images}
          index={lightboxIndex}
          onIndex={setLightboxIndex}
          onClose={() => setLightboxIndex(null)}
          theme={theme}
        />
      )}

      {/* Delete confirmation modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="glass rounded-2xl p-6 max-w-sm w-full mx-4">
            <h3 className="text-white font-semibold text-base mb-2">Delete items?</h3>
            <p className="text-white/60 text-sm mb-5">
              {selectedIds.size} item{selectedIds.size !== 1 ? 's' : ''} will be permanently deleted. This cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowDeleteModal(false)}
                className="glass-btn px-4 py-2 text-white/70 hover:text-white text-sm"
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                className="px-4 py-2 bg-red-500/80 hover:bg-red-500 border border-red-400/30 rounded-xl text-white text-sm transition-all"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function buildPageNumbers(current, total) {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1)
  const pages = new Set([1, total, current])
  for (let d = -2; d <= 2; d++) {
    const p = current + d
    if (p >= 1 && p <= total) pages.add(p)
  }
  const sorted = [...pages].sort((a, b) => a - b)
  const result = []
  for (let i = 0; i < sorted.length; i++) {
    if (i > 0 && sorted[i] - sorted[i - 1] > 1) result.push('…')
    result.push(sorted[i])
  }
  return result
}

function GalleryThumbnail({ img, onClick, onGoToChat, selectionMode, selected, onToggleSelect, isLight }) {
  const [hovered, setHovered] = useState(false)
  const isVideo = img.type === 'video'

  function handleClick(e) {
    if (selectionMode) { e.stopPropagation(); onToggleSelect(); return }
    onClick()
  }

  return (
    <div
      className="break-inside-avoid mb-3 rounded-xl cursor-pointer group transition-all"
      style={{
        outline:       selected ? '2px solid rgba(34,211,238,0.75)' : '2px solid transparent',
        outlineOffset: '0px',
        boxShadow:     selected ? '0 0 14px 3px rgba(34,211,238,0.25)' : 'none',
        borderRadius:  '0.75rem',
      }}
      onClick={handleClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
    <div className="relative rounded-xl overflow-hidden w-full">
      {selectionMode && (
        <div className="absolute top-2 left-2 z-10">
          <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center transition-all ${
            selected
              ? 'bg-cyan-400 border-cyan-400 shadow-[0_0_8px_2px_rgba(34,211,238,0.5)]'
              : 'border-white/70 bg-black/30 backdrop-blur-sm'
          }`}>
            {selected && (
              <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                <path d="M2 5l2.5 2.5L8 3" stroke="#0e0c2e" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            )}
          </div>
        </div>
      )}

      {/* Video badge */}
      {isVideo && (
        <div className="absolute top-2 right-2 z-10 bg-black/60 backdrop-blur-sm rounded-lg px-1.5 py-0.5 text-white text-[10px] font-medium">
          ▶ video
        </div>
      )}

      {isVideo ? (
        <video
          src={img.mediaUrl}
          muted
          loop
          autoPlay
          playsInline
          className={`w-full block rounded-xl transition-transform duration-200 ${selectionMode ? '' : 'group-hover:scale-[1.02]'}`}
        />
      ) : (
        <img
          src={img.mediaUrl}
          alt={img.prompt}
          className={`w-full block rounded-xl transition-transform duration-200 ${selectionMode ? '' : 'group-hover:scale-[1.02]'}`}
          loading="lazy"
        />
      )}

      {hovered && !selectionMode && (
        <div
          className="absolute inset-0 rounded-xl flex flex-col justify-end p-2.5"
          style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.82) 0%, rgba(0,0,0,0.4) 55%, transparent 100%)' }}
        >
          {img.prompt && (
            <p className="text-[11px] leading-snug mb-1" style={{ color: 'rgba(255,255,255,0.92)' }}>
              {img.prompt}
            </p>
          )}
          <div className="flex items-center justify-between gap-1 mt-0.5">
            {img.chatInfo ? (
              <p className="text-[10px] truncate flex-1" style={{ color: 'rgba(255,255,255,0.5)' }}>
                {img.chatInfo.chatTitle}
              </p>
            ) : (
              <span className="flex-1" />
            )}
            {onGoToChat && (
              <button
                onClick={(e) => { e.stopPropagation(); onGoToChat() }}
                className="glass-btn px-2 py-0.5 text-[10px] whitespace-nowrap flex-shrink-0"
                style={{ color: 'rgba(255,255,255,0.8)' }}
              >
                Go to chat →
              </button>
            )}
          </div>
        </div>
      )}
    </div>
    </div>
  )
}
