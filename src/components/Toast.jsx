import { createContext, useCallback, useContext, useState } from 'react'
import { createPortal } from 'react-dom'

const ToastContext = createContext(null)

let _id = 0

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])

  const showToast = useCallback((message, { duration = 4000, type = 'info' } = {}) => {
    const id = ++_id
    setToasts(prev => [...prev, { id, message, type }])
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), duration)
  }, [])

  const dismiss = useCallback((id) => setToasts(prev => prev.filter(t => t.id !== id)), [])

  return (
    <ToastContext.Provider value={showToast}>
      {children}
      {createPortal(
        <div className="fixed top-6 right-6 z-[200] flex flex-col gap-2 pointer-events-none">
          {toasts.map(t => (
            <div
              key={t.id}
              className="pointer-events-auto flex items-start gap-2.5 px-4 py-3 rounded-xl shadow-xl text-sm max-w-xs
                bg-white/10 backdrop-blur-md border border-white/20 text-white"
              style={{ animation: 'toast-in 0.2s ease' }}
            >
              <span className="mt-px flex-shrink-0">
                {t.type === 'success' ? '✓' : t.type === 'error' ? '✕' : 'ℹ'}
              </span>
              <span className="leading-snug">{t.message}</span>
              <button
                onClick={() => dismiss(t.id)}
                className="ml-auto flex-shrink-0 text-white/40 hover:text-white leading-none"
              >
                ×
              </button>
            </div>
          ))}
        </div>,
        document.body
      )}
      <style>{`
        @keyframes toast-in {
          from { opacity: 0; transform: translateY(8px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </ToastContext.Provider>
  )
}

/** Returns a showToast(message, options?) function. */
export function useToast() {
  return useContext(ToastContext)
}
