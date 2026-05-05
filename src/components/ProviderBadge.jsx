const LABELS = {
  ollama: 'Ollama',
  lmstudio: 'LM Studio',
  llamacpp: 'llama.cpp',
  huggingface: 'Hugging Face',
}

export default function ProviderBadge({ provider }) {
  return (
    <span className="px-2 py-0.5 text-xs font-medium text-cyan-300/80 bg-cyan-500/10 border border-cyan-400/20 rounded-full">
      {LABELS[provider] ?? provider}
    </span>
  )
}
