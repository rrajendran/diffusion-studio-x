import { useState, useRef, useEffect } from 'react'

export default function Sidebar({
  chats, activeChatId, onSelectChat, onNewChat, onDeleteChat, onRenameChat,
  showSettings, onToggleSettings, showGallery, onToggleGallery,
  collapsed, onToggleCollapse, sidebarWidth,
}) {
  const [chatSearch, setChatSearch] = useState('')
  const [renamingId, setRenamingId] = useState(null)
  const [renameValue, setRenameValue] = useState('')
  const renameInputRef = useRef(null)

  useEffect(() => {
    if (renamingId && renameInputRef.current) renameInputRef.current.focus()
  }, [renamingId])

  const filteredChats = chats.filter(c =>
    !chatSearch.trim() || c.title.toLowerCase().includes(chatSearch.toLowerCase())
  )

  function startRename(chat, e) {
    e.stopPropagation()
    if (collapsed) return
    setRenamingId(chat.id)
    setRenameValue(chat.title)
  }

  function commitRename(id) {
    if (renameValue.trim()) onRenameChat(id, renameValue.trim())
    setRenamingId(null)
  }

  function handleRenameKey(e, id) {
    if (e.key === 'Enter') { e.preventDefault(); commitRename(id) }
    if (e.key === 'Escape') setRenamingId(null)
  }

  const effectiveWidth = collapsed ? 56 : sidebarWidth

  return (
    <aside
      className="flex-shrink-0 glass m-3 flex flex-col overflow-hidden transition-all duration-200"
      style={{ width: effectiveWidth + 'px' }}
    >
      {/* Header */}
      <div className="px-3 pt-3 pb-2 flex items-center justify-between gap-2 flex-shrink-0">
        {!collapsed && (
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <img src="/logo.png" alt="Diffusion Studio X" className="h-9 w-auto flex-shrink-0" />
            <div className="min-w-0">
              <h1 className="text-white font-bold text-sm tracking-tight leading-tight">
                Diffusion Studio <span className="dsx-x">X</span>
              </h1>
              <p className="text-white/40 text-[10px] leading-tight">AI image creation</p>
            </div>
          </div>
        )}
        {collapsed && (
          <img src="/logo.png" alt="DSX" className="h-7 w-auto mx-auto" />
        )}
        <button
          onClick={onToggleCollapse}
          className="glass-btn w-6 h-6 flex items-center justify-center text-white/50 hover:text-white flex-shrink-0 text-xs"
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? '›' : '‹'}
        </button>
      </div>

      {/* New Chat button */}
      <div className="px-2 pb-2 flex-shrink-0">
        <button
          onClick={onNewChat}
          className="w-full glass-btn py-2 px-2 text-white text-sm font-medium flex items-center justify-center gap-2"
          title="New Chat"
        >
          <span className="text-base leading-none flex-shrink-0">+</span>
          {!collapsed && <span>New Chat</span>}
        </button>
      </div>

      {/* Search */}
      {!collapsed && (
        <div className="px-2 pb-2 flex-shrink-0">
          <input
            type="text"
            value={chatSearch}
            onChange={e => setChatSearch(e.target.value)}
            placeholder="Search chats…"
            className="w-full bg-white/10 border border-white/20 rounded-xl text-white text-xs px-3 py-1.5 outline-none placeholder-white/30"
          />
        </div>
      )}

      {/* Chat list */}
      <div className="flex-1 overflow-y-auto scrollbar-thin px-2 pb-2" style={{ maxHeight: '780px' }}>
        {!collapsed && filteredChats.length === 0 && (
          <p className="text-white/30 text-xs text-center py-4">
            {chatSearch ? 'No matches' : 'No chats yet'}
          </p>
        )}
        {filteredChats.map(chat => (
          <div
            key={chat.id}
            onClick={() => { if (renamingId !== chat.id) onSelectChat(chat.id) }}
            className={`group flex items-center justify-between rounded-xl px-2 py-2 mb-1 cursor-pointer transition-all ${
              !showSettings && !showGallery && chat.id === activeChatId
                ? 'bg-white/20 border border-white/25'
                : 'hover:bg-white/10'
            }`}
          >
            {collapsed ? (
              <span className="text-white/50 text-xs mx-auto">💬</span>
            ) : renamingId === chat.id ? (
              <input
                ref={renameInputRef}
                value={renameValue}
                onChange={e => setRenameValue(e.target.value)}
                onBlur={() => commitRename(chat.id)}
                onKeyDown={e => handleRenameKey(e, chat.id)}
                onClick={e => e.stopPropagation()}
                className="flex-1 bg-white/20 border border-white/30 rounded-lg text-white text-xs px-2 py-0.5 outline-none min-w-0"
              />
            ) : (
              <div className="flex-1 min-w-0">
                <p
                  className="text-white text-xs font-medium truncate"
                  onDoubleClick={e => startRename(chat, e)}
                  title="Double-click to rename"
                >
                  {chat.title}
                </p>
                <p className="text-white/35 text-[10px] mt-0.5">
                  {chat.messages.length} msg{chat.messages.length !== 1 ? 's' : ''}
                </p>
              </div>
            )}
            {!collapsed && renamingId !== chat.id && (
              <button
                onClick={e => { e.stopPropagation(); onDeleteChat(chat.id) }}
                className="ml-1 opacity-0 group-hover:opacity-100 text-white/40 hover:text-white/80 text-xs transition-all flex-shrink-0"
                title="Delete chat"
              >
                ✕
              </button>
            )}
          </div>
        ))}
      </div>

      {/* Bottom nav */}
      <div className="px-2 pb-2 pt-2 border-t border-white/10 flex flex-col gap-1 flex-shrink-0">
        <button
          onClick={onToggleGallery}
          className={`w-full flex items-center gap-2 px-2 py-2 rounded-xl text-sm transition-all ${
            showGallery
              ? 'bg-white/20 border border-white/25 text-white'
              : 'text-white/50 hover:text-white hover:bg-white/10'
          } ${collapsed ? 'justify-center' : ''}`}
          title="Gallery"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0">
            <rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/>
            <rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>
          </svg>
          {!collapsed && 'Gallery'}
        </button>
        <button
          onClick={onToggleSettings}
          className={`w-full flex items-center gap-2 px-2 py-2 rounded-xl text-sm transition-all ${
            showSettings
              ? 'bg-white/20 border border-white/25 text-white'
              : 'text-white/50 hover:text-white hover:bg-white/10'
          } ${collapsed ? 'justify-center' : ''}`}
          title="Settings"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0">
            <circle cx="12" cy="12" r="3"/>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
          </svg>
          {!collapsed && 'Settings'}
        </button>
      </div>
    </aside>
  )
}
