import { useState } from 'react'
import { useToast } from './Toast.jsx'

export default function VideoResult({ videoUrl, prompt, meta, onExpand }) {
  const [hovered, setHovered] = useState(false)
  const showToast = useToast()

  const download = async (e) => {
    e.stopPropagation()
    try {
      const res = await fetch(videoUrl)
      const blob = await res.blob()
      const filename = `generated-${Date.now()}.mp4`
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      a.click()
      URL.revokeObjectURL(url)
      showToast?.(`Saved to Downloads: ${filename}`, { type: 'success' })
    } catch {
      window.open(videoUrl, '_blank')
      showToast?.('Could not download — opened in new tab', { type: 'error' })
    }
  }

  return (
    <div
      className="relative mt-2 rounded-xl overflow-hidden max-w-sm cursor-pointer"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={() => onExpand?.(videoUrl, prompt, meta, 'video')}
    >
      <video
        src={videoUrl}
        autoPlay
        loop
        muted
        playsInline
        className="w-full rounded-xl object-cover"
      />
      {hovered && (
        <div className="absolute inset-0 flex items-end justify-between px-3 pb-3 bg-black/30 backdrop-blur-sm rounded-xl">
          <button
            onClick={download}
            className="glass-btn px-3 py-1.5 text-white text-xs font-medium"
          >
            Download
          </button>
          <button
            onClick={e => { e.stopPropagation(); onExpand?.(videoUrl, prompt, meta, 'video') }}
            className="glass-btn px-3 py-1.5 text-white text-xs font-medium"
          >
            Expand
          </button>
        </div>
      )}
    </div>
  )
}
