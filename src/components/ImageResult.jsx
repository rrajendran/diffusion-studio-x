import { useState } from 'react'
import { useToast } from './Toast.jsx'

export default function ImageResult({ imageUrl, prompt, meta, onExpand }) {
  const [hovered, setHovered] = useState(false)
  const showToast = useToast()

  const download = async (e) => {
    e.stopPropagation()
    try {
      const res = await fetch(imageUrl)
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
      window.open(imageUrl, '_blank')
      showToast?.('Could not download — opened in new tab', { type: 'error' })
    }
  }

  return (
    <div
      className="relative mt-2 rounded-xl overflow-hidden max-w-sm cursor-pointer"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={() => onExpand?.(imageUrl, prompt, meta)}
    >
      <img src={imageUrl} alt={prompt} className="w-full rounded-xl object-cover" />
      {hovered && (
        <div className="absolute inset-0 flex items-end justify-between px-3 pb-3 bg-black/30 backdrop-blur-sm rounded-xl">
          <button
            onClick={download}
            className="glass-btn px-3 py-1.5 text-white text-xs font-medium"
          >
            Download
          </button>
          <button
            onClick={e => { e.stopPropagation(); onExpand?.(imageUrl, prompt, meta) }}
            className="glass-btn px-3 py-1.5 text-white text-xs font-medium"
          >
            Expand
          </button>
        </div>
      )}
    </div>
  )
}
