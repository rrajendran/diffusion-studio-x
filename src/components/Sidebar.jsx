export default function Sidebar({ chats, activeChatId, onSelectChat, onNewChat, onDeleteChat }) {
  return (
    <aside className="w-64 flex-shrink-0 glass m-3 flex flex-col overflow-hidden">
      <div className="px-4 pt-4 pb-3">
        <div className="flex items-center gap-2.5 mb-0.5">
          <img src="/logo.png" alt="Diffusion Studio X" className="h-11 w-auto flex-shrink-0" />
          <div className="min-w-0">
            <h1 className="text-white font-bold text-sm tracking-tight leading-tight">
              Diffusion Studio <span className="dsx-x">X</span>
            </h1>
            <p className="text-white/40 text-[10px] leading-tight">AI-powered image creation</p>
          </div>
        </div>
      </div>

      <div className="px-4 pb-3">
        <button
          onClick={onNewChat}
          className="w-full glass-btn py-2 px-3 text-white text-sm font-medium text-left flex items-center gap-2"
        >
          <span className="text-base leading-none">+</span> New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin px-3 pb-3">
        {chats.length === 0 && (
          <p className="text-white/30 text-xs text-center py-4">No chats yet</p>
        )}
        {chats.map(chat => (
          <div
            key={chat.id}
            onClick={() => onSelectChat(chat.id)}
            className={`group flex items-center justify-between rounded-xl px-3 py-2.5 mb-1 cursor-pointer transition-all ${
              chat.id === activeChatId
                ? 'bg-white/20 border border-white/25'
                : 'hover:bg-white/10'
            }`}
          >
            <div className="flex-1 min-w-0">
              <p className="text-white text-xs font-medium truncate">{chat.title}</p>
              <p className="text-white/35 text-xs mt-0.5">
                {chat.messages.length} message{chat.messages.length !== 1 ? 's' : ''}
              </p>
            </div>
            <button
              onClick={e => { e.stopPropagation(); onDeleteChat(chat.id) }}
              className="ml-2 opacity-0 group-hover:opacity-100 text-white/40 hover:text-white/80 text-xs transition-all flex-shrink-0"
              title="Delete chat"
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </aside>
  )
}
