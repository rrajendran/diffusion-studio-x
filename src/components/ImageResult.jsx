import { useState } from 'react'

export default function ImageResult({ imageUrl, prompt, onExpand }) {
  const [hovered, setHovered] = useState(false)

  const download = (e) => {
    e.stopPropagation()
    const a = document.createElement('a')
    a.href = imageUrl
    a.download = `generated-${Date.now()}.png`
    a.click()
  }

  return (
    <div
      className="relative mt-2 rounded-xl overflow-hidden max-w-sm cursor-pointer"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={() => onExpand?.(imageUrl, prompt)}
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
            onClick={e => { e.stopPropagation(); onExpand?.(imageUrl, prompt) }}
            className="glass-btn px-3 py-1.5 text-white text-xs font-medium"
          >
            Expand
          </button>
        </div>
      )}
    </div>
  )
}
