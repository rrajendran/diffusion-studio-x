import { useEffect, useCallback } from 'react'
import { createPortal } from 'react-dom'
import { useToast } from './Toast.jsx'

/**
 * Full-screen lightbox / carousel.
 *
 * Props:
 *   images   – array of { imageUrl?, videoUrl?, type?, prompt, meta? }
 *   index    – currently shown index
 *   onIndex  – (newIndex) => void
 *   onClose  – () => void
 */
export default function Lightbox({ images, index, onIndex, onClose, theme = 'dark' }) {
  const total = images.length
  const item  = images[index] ?? images[0]
  const showToast = useToast()
  const isLight = theme === 'light'

  const mediaUrl = item.videoUrl || item.imageUrl
  const isVideo = item.type === 'video' || !!item.videoUrl

  const prev = useCallback(() => onIndex((index - 1 + total) % total), [index, total, onIndex])
  const next = useCallback(() => onIndex((index + 1) % total), [index, total, onIndex])

  // Keyboard navigation
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Escape')      onClose()
      else if (e.key === 'ArrowLeft')  prev()
      else if (e.key === 'ArrowRight') next()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose, prev, next])

  // Lock body scroll
  useEffect(() => {
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = '' }
  }, [])

  const download = async () => {
    try {
      const res = await fetch(mediaUrl)
      const blob = await res.blob()
      const ext = isVideo ? 'mp4' : (blob.type === 'image/jpeg' ? 'jpg' : blob.type === 'image/webp' ? 'webp' : 'png')
      const filename = `generated-${Date.now()}.${ext}`
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      a.click()
      URL.revokeObjectURL(url)
      showToast?.(`Saved to Downloads: ${filename}`, { type: 'success' })
    } catch {
      window.open(mediaUrl, '_blank')
      showToast?.('Could not download — opened in new tab', { type: 'error' })
    }
  }

  const meta = item.meta

  return createPortal(
    <div className={`theme-${theme}`}>
    <div
      className="fixed inset-0 z-50 flex flex-col"
      style={{
        background: isLight ? 'rgba(245,245,247,0.96)' : 'rgba(7,5,18,0.96)',
        backdropFilter: 'blur(20px)',
      }}
      onClick={onClose}
    >
      {/* Top bar */}
      <div
        className="flex items-center justify-between px-5 py-3 flex-shrink-0"
        style={{
          background: isLight ? 'rgba(0,0,0,0.04)' : 'rgba(255,255,255,0.04)',
          borderBottom: isLight ? '1px solid rgba(0,0,0,0.08)' : '1px solid rgba(255,255,255,0.08)',
        }}
        onClick={e => e.stopPropagation()}
      >
        <span className="text-xs tabular-nums" style={{ color: isLight ? 'rgba(0,0,0,0.45)' : 'rgba(255,255,255,0.45)' }}>
          {index + 1} / {total}
        </span>

        {meta && (
          <div className="flex items-center gap-3 text-xs font-mono" style={{ color: isLight ? 'rgba(0,0,0,0.4)' : 'rgba(255,255,255,0.38)' }}>
            {meta.model && <span>{meta.model.split('/').pop()}</span>}
            {meta.seed != null && <span>seed {meta.seed}</span>}
            {meta.steps != null && <span>{meta.steps} steps</span>}
            {meta.width && meta.height && <span>{meta.width}×{meta.height}</span>}
            {meta.guidance_scale != null && <span>cfg {meta.guidance_scale}</span>}
            {meta.num_frames != null && <span>{meta.num_frames}f @ {meta.fps}fps</span>}
          </div>
        )}

        <div className="flex items-center gap-2">
          <button
            onClick={download}
            className="glass-btn px-3 py-1.5 text-white/60 hover:text-white text-xs"
          >
            Download
          </button>
          <button
            onClick={onClose}
            className="glass-btn px-2.5 py-1.5 text-white/60 hover:text-white text-sm leading-none"
            title="Close (Esc)"
          >
            ✕
          </button>
        </div>
      </div>

      {/* Media area */}
      <div
        className="flex-1 flex items-center justify-center relative min-h-0 px-16 overflow-auto"
        onClick={e => e.stopPropagation()}
      >
        {/* Prev button */}
        {total > 1 && (
          <button
            onClick={prev}
            title="Previous (←)"
            className="absolute left-3 top-1/2 -translate-y-1/2 z-10 glass-btn w-10 h-10 flex items-center justify-center text-white/70 hover:text-white text-lg"
          >
            ‹
          </button>
        )}

        {isVideo ? (
          <video
            key={mediaUrl}
            src={mediaUrl}
            controls
            autoPlay
            loop
            playsInline
            className="block rounded-xl shadow-2xl max-h-full max-w-full select-none"
          />
        ) : (
          <img
            key={mediaUrl}
            src={mediaUrl}
            alt={item.prompt ?? ''}
            className="block rounded-xl shadow-2xl select-none"
            draggable={false}
          />
        )}

        {/* Next button */}
        {total > 1 && (
          <button
            onClick={next}
            title="Next (→)"
            className="absolute right-3 top-1/2 -translate-y-1/2 z-10 glass-btn w-10 h-10 flex items-center justify-center text-white/70 hover:text-white text-lg"
          >
            ›
          </button>
        )}
      </div>

      {/* Caption */}
      {item.prompt && (
        <div
          className="flex-shrink-0 text-center px-8 py-3 text-xs leading-relaxed"
          onClick={e => e.stopPropagation()}
          style={{
            borderTop: isLight ? '1px solid rgba(0,0,0,0.06)' : '1px solid rgba(255,255,255,0.06)',
            color: isLight ? 'rgba(0,0,0,0.65)' : 'rgba(255,255,255,0.55)',
          }}
        >
          {item.imagePrompt ?? item.prompt}
        </div>
      )}

      {/* Dot strip */}
      {total > 1 && (
        <div
          className="flex-shrink-0 flex items-center justify-center gap-1.5 pb-4"
          onClick={e => e.stopPropagation()}
        >
          {images.map((_, i) => (
            <button
              key={i}
              onClick={() => onIndex(i)}
              className="rounded-full transition-all"
              style={{
                width:  i === index ? '16px' : '6px',
                height: '6px',
                background: i === index
                  ? (isLight ? 'rgba(0,0,0,0.55)' : 'rgba(255,255,255,0.75)')
                  : (isLight ? 'rgba(0,0,0,0.2)'  : 'rgba(255,255,255,0.28)'),
              }}
            />
          ))}
        </div>
      )}
    </div>
    </div>,
    document.body
  )
}
