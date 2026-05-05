import { useEffect, useRef, useState } from 'react'
import ChatMessage from './ChatMessage.jsx'

export default function ChatWindow({ messages, loading, onImageClick }) {
  const bottomRef = useRef(null)
  const [progress, setProgress] = useState({ step: 0, total: 0 })
  const pollRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  useEffect(() => {
    if (!loading) {
      setProgress({ step: 0, total: 0 })
      clearInterval(pollRef.current)
      return
    }
    setProgress({ step: 0, total: 0 })
    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch('http://localhost:3001/api/image/progress')
        if (r.ok) setProgress(await r.json())
      } catch { /* image server not running */ }
    }, 800)
    return () => clearInterval(pollRef.current)
  }, [loading])

  const pct = progress.total > 0 ? Math.round((progress.step / progress.total) * 100) : null

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 scrollbar-thin">
      {messages.length === 0 && (
        <div className="flex flex-col items-center justify-center h-full text-white/40 text-sm gap-2">
          <span className="text-4xl">🎨</span>
          <p>Describe an image to generate it</p>
        </div>
      )}
      {messages.map(msg => (
        <ChatMessage key={msg.id} message={msg} onImageClick={onImageClick} />
      ))}
      {loading && (
        <div className="flex justify-start mb-3">
          <div className="glass px-4 py-3 text-white/60 text-sm flex items-center gap-2">
            <span className="flex gap-1">
              {[0, 1, 2].map(i => (
                <span
                  key={i}
                  className="w-1.5 h-1.5 bg-white/60 rounded-full animate-bounce"
                  style={{ animationDelay: `${i * 0.15}s` }}
                />
              ))}
            </span>
            {pct !== null ? `Generating… ${pct}%` : 'Generating…'}
            {pct !== null && (
              <span className="ml-1 text-white/30 text-xs">
                {progress.step}/{progress.total}
              </span>
            )}
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  )
}
