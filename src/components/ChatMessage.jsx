import { useState } from 'react'
import ImageResult from './ImageResult.jsx'
import VideoResult from './VideoResult.jsx'

export default function ChatMessage({ message, onImageClick, onEdit }) {
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

  const hasVideo = !!message.videoUrl
  const hasImage = !!message.imageUrl

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

        {hasVideo && (
          <VideoResult videoUrl={message.videoUrl} prompt={message.imagePrompt ?? message.content} meta={message.meta} onExpand={onImageClick} />
        )}

        {hasImage && (
          <ImageResult imageUrl={message.imageUrl} prompt={message.imagePrompt ?? message.content} meta={message.meta} onExpand={(url, p, m) => onImageClick?.(url, p, m, 'image')} />
        )}

        {message.meta && (
          <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-white/40 text-xs font-mono">
            {message.meta.model && <span title="Model">{message.meta.model.split('/').pop()}</span>}
            {message.meta.seed != null && <span title="Seed">seed {message.meta.seed}</span>}
            {message.meta.steps != null && <span title="Steps">steps {message.meta.steps}</span>}
            {message.meta.width && message.meta.height && <span title="Resolution">{message.meta.width}×{message.meta.height}</span>}
            {message.meta.guidance_scale != null && <span title="Guidance scale">cfg {message.meta.guidance_scale}</span>}
            {message.meta.num_frames != null && <span title="Frames">{message.meta.num_frames}f @ {message.meta.fps}fps</span>}
          </div>
        )}
        {message.imagePrompt && (
          <p className="mt-1 text-white/30 text-[11px] italic leading-snug" title="Image prompt">
            ↳ {message.imagePrompt}
          </p>
        )}

        {/* Copy actions — visible on hover */}
        {!message.error && (
          <div className="flex gap-1 mt-2 opacity-0 group-hover:opacity-100 transition-opacity duration-150">
            {isUser && onEdit && (
              <button
                onClick={() => onEdit(message)}
                title="Edit and resend"
                className="glass-btn px-2 py-0.5 text-white/60 hover:text-white text-xs"
              >
                ✎ edit
              </button>
            )}
            <button
              onClick={copyText}
              title="Copy text"
              className="glass-btn px-2 py-0.5 text-white/60 hover:text-white text-xs"
            >
              {textCopied ? '✓ copied' : '⎘ text'}
            </button>
            {hasImage && (
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
