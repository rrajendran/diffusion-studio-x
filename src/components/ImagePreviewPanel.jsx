import { useRef, useState, useEffect } from 'react'

export default function ImagePreviewPanel({ imageUrl, prompt, onClose }) {
  const panelRef = useRef(null)
  const [isFullscreen, setIsFullscreen] = useState(false)

  useEffect(() => {
    const onChange = () => setIsFullscreen(!!document.fullscreenElement)
    document.addEventListener('fullscreenchange', onChange)
    return () => document.removeEventListener('fullscreenchange', onChange)
  }, [])

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      panelRef.current?.requestFullscreen()
    } else {
      document.exitFullscreen()
    }
  }

  const download = () => {
    const a = document.createElement('a')
    a.href = imageUrl
    a.download = `generated-${Date.now()}.png`
    a.click()
  }

  return (
    <aside
      ref={panelRef}
      className="w-full h-full glass m-3 ml-0 flex flex-col overflow-hidden"
      style={isFullscreen ? { margin: 0, borderRadius: 0, background: '#0f0c1a' } : {}}
    >
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/10 flex-shrink-0">
        <span className="text-white text-xs font-medium truncate mr-2">Preview</span>
        <div className="flex items-center gap-2 flex-shrink-0">
          <button
            onClick={toggleFullscreen}
            title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
            className="text-white/40 hover:text-white/80 text-sm transition-colors"
          >
            {isFullscreen ? '⊡' : '⛶'}
          </button>
          <button
            onClick={onClose}
            title="Close preview"
            className="text-white/40 hover:text-white/80 text-sm transition-colors"
          >
            ✕
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin p-4 flex flex-col gap-4">
        <img
          src={imageUrl}
          alt={prompt}
          className="rounded-xl object-contain max-w-full mx-auto block"
          style={{ width: 'auto', height: 'auto' }}
        />
        {prompt && (
          <p className="text-white/60 text-xs leading-relaxed">{prompt}</p>
        )}
      </div>

      <div className="px-4 py-3 border-t border-white/10 flex-shrink-0">
        <button
          onClick={download}
          className="w-full glass-btn py-2 px-3 text-white text-sm font-medium text-center"
        >
          Download
        </button>
      </div>
    </aside>
  )
}
