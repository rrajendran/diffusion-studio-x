import { useState } from 'react'
import ImageResult from './ImageResult.jsx'

export default function ChatMessage({ message, onImageClick }) {
  const isUser = message.role === 'user'
  const [textCopied, setTextCopied] = useState(false)
  const [imgCopied, setImgCopied] = useState(false)

  const copyText = async () => {
    await navigator.clipboard.writeText(message.content)
    setTextCopied(true)
    setTimeout(() => setTextCopied(false), 1500)
  }

  const copyImage = async () => {
    try {
      const res = await fetch(message.imageUrl)
      const blob = await res.blob()
      await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })])
    } catch {
      await navigator.clipboard.writeText(message.imageUrl)
    }
    setImgCopied(true)
    setTimeout(() => setImgCopied(false), 1500)
  }

  return (
    <div className={`group flex ${isUser ? 'justify-end' : 'justify-start'} mb-3`}>
      <div
        className={`max-w-[75%] px-4 py-3 rounded-2xl text-white text-sm leading-relaxed ${
          isUser
            ? 'bg-white/25 backdrop-blur-sm border border-white/30 rounded-br-sm'
            : message.error
            ? 'bg-red-500/20 backdrop-blur-sm border border-red-400/30 rounded-bl-sm'
            : 'bg-white/10 backdrop-blur-sm border border-white/15 rounded-bl-sm'
        }`}
      >
        {isUser && message.referenceImageUrl && (
          <img
            src={message.referenceImageUrl}
            alt="Reference"
            className="max-h-20 rounded-lg object-contain border border-white/20 mb-1"
          />
        )}
        <p className={message.error ? 'text-red-300' : ''}>{message.content}</p>

        {message.imageUrl && (
          <ImageResult imageUrl={message.imageUrl} prompt={message.content} onExpand={onImageClick} />
        )}

        {/* Copy actions — visible on hover */}
        {!message.error && (
          <div className="flex gap-1 mt-2 opacity-0 group-hover:opacity-100 transition-opacity duration-150">
            <button
              onClick={copyText}
              title="Copy text"
              className="glass-btn px-2 py-0.5 text-white/60 hover:text-white text-xs"
            >
              {textCopied ? '✓ copied' : '⎘ text'}
            </button>
            {message.imageUrl && (
              <button
                onClick={copyImage}
                title="Copy image"
                className="glass-btn px-2 py-0.5 text-white/60 hover:text-white text-xs"
              >
                {imgCopied ? '✓ copied' : '⎘ image'}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
