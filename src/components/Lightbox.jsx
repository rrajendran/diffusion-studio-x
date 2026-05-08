import { useEffect, useCallback } from 'react'
import { createPortal } from 'react-dom'
import { useToast } from './Toast.jsx'

/**
 * Full-screen lightbox / carousel.
 *
 * Props:
 *   images   – array of { imageUrl, prompt, meta? }
 *   index    – currently shown index
 *   onIndex  – (newIndex) => void
 *   onClose  – () => void
 */
export default function Lightbox({ images, index, onIndex, onClose }) {
  const total = images.length
  const item  = images[index] ?? images[0]
  const showToast = useToast()

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
      const res = await fetch(item.imageUrl)
      const blob = await res.blob()
      const ext = blob.type === 'image/jpeg' ? 'jpg' : blob.type === 'image/webp' ? 'webp' : 'png'
      const filename = `generated-${Date.now()}.${ext}`
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      a.click()
      URL.revokeObjectURL(url)
      showToast?.(`Saved to Downloads: ${filename}`, { type: 'success' })
    } catch {
      // fallback: open in new tab
      window.open(item.imageUrl, '_blank')
      showToast?.('Could not download — opened in new tab', { type: 'error' })
    }
  }

  const meta = item.meta

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex flex-col"
      style={{ background: 'rgba(7,5,18,0.96)', backdropFilter: 'blur(20px)' }}
      onClick={onClose}
    >
      {/* Top bar */}
      <div
        className="flex items-center justify-between px-5 py-3 flex-shrink-0"
        style={{ background: 'rgba(255,255,255,0.04)', borderBottom: '1px solid rgba(255,255,255,0.08)' }}
        onClick={e => e.stopPropagation()}
      >
        <span className="text-white/40 text-xs tabular-nums">
          {index + 1} / {total}
        </span>

        {meta && (
          <div className="flex items-center gap-3 text-white/35 text-xs font-mono">
            {meta.model && <span>{meta.model.split('/').pop()}</span>}
            {meta.seed != null && <span>seed {meta.seed}</span>}
            {meta.steps != null && <span>{meta.steps} steps</span>}
            {meta.width && meta.height && <span>{meta.width}×{meta.height}</span>}
            {meta.guidance_scale != null && <span>cfg {meta.guidance_scale}</span>}
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

      {/* Image area */}
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

        <img
          key={item.imageUrl}
          src={item.imageUrl}
          alt={item.prompt ?? ''}
          className="block rounded-xl shadow-2xl select-none"
          draggable={false}
        />

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
          className="flex-shrink-0 text-center px-8 py-3 text-white/45 text-xs leading-relaxed"
          onClick={e => e.stopPropagation()}
          style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}
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
              className={`rounded-full transition-all ${
                i === index
                  ? 'w-4 h-1.5 bg-white/70'
                  : 'w-1.5 h-1.5 bg-white/25 hover:bg-white/50'
              }`}
            />
          ))}
        </div>
      )}
    </div>,
    document.body
  )
}
