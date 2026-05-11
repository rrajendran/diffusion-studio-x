// ── Feature definitions ───────────────────────────────────────────────────────
// Edit this file to add, remove, or rename providers and model families.
// Changes here automatically propagate to the Settings toggles and model filter.

export const PROVIDERS = [
  { value: 'huggingface', label: 'Hugging Face', hint: 'Diffusers image server + HF Hub models' , defaultEnabled: true },
  { value: 'ollama',      label: 'Ollama',        hint: 'Local Ollama image models', defaultEnabled: true },
  { value: 'lmstudio',   label: 'LM Studio',     hint: 'Local LM Studio models (port 1234)', defaultEnabled: false },
  { value: 'llamacpp',   label: 'llama.cpp',     hint: 'Local llama.cpp server (port 8080)', defaultEnabled: false },
]

// Each entry needs:
//   key     — stable identifier stored in config/localStorage
//   label   — display name shown in Settings
//   hint    — one-line description shown under the toggle
//   pattern — regex tested against a model_id to classify it
//   defaultEnabled — whether the switch is on out of the box
export const ADAPTER_FAMILIES = [
  {
    key:            'flux',
    label:          'FLUX.1 (schnell / dev)',
    hint:           'Black Forest Labs FLUX distilled models',
    pattern:        /FLUX\.1-(schnell|dev)/i,
    defaultEnabled: true,
  },
  {
    key:            'fluxKontext',
    label:          'FLUX.1 Kontext',
    hint:           'FLUX image editing / in-painting variant',
    pattern:        /FLUX\.1-Kontext/i,
    defaultEnabled: true,
  },
  {
    key:            'fluxKlein',
    label:          'FLUX.2 Klein',
    hint:           'Compact FLUX variant (Klein / Klein-KV)',
    pattern:        /FLUX\.2-klein/i,
    defaultEnabled: true,
  },
  {
    key:            'sd3',
    label:          'Stable Diffusion 3',
    hint:           'Stability AI SD3 family',
    pattern:        /stable-diffusion-3/i,
    defaultEnabled: true,
  },
  {
    key:            'wan',
    label:          'Wan2.2 Video',
    hint:           'Wan-AI text-to-video and image-to-video',
    pattern:        /wan-ai\/wan/i,
    defaultEnabled: false,
  },
  {
    key:            'qwen',
    label:          'Qwen-Image',
    hint:           'Alibaba Qwen multimodal image models',
    pattern:        /qwen.*image/i,
    defaultEnabled: true,
  },
  {
    key:            'glm',
    label:          'GLM-Image',
    hint:           'Zhipu AI GLM-Image diffusion models',
    pattern:        /GLM-Image/i,
    defaultEnabled: true,
  },
  {
    key:            'ernie',
    label:          'ERNIE-Image',
    hint:           'Baidu ERNIE image generation models',
    pattern:        /ERNIE-Image/i,
    defaultEnabled: false,
  },
  {
    key:            'zimage',
    label:          'Z-Image',
    hint:           'Zhipu AI Z-Image models',
    pattern:        /[Zz]-[Ii]mage/i,
    defaultEnabled: true,
  },
  {
    key:            'diffusion',
    label:          'Stable Diffusion (SDXL)',
    hint:           'Generic stable-diffusion / SDXL family',
    pattern:        /stable-diffusion/i,
    defaultEnabled: false,
  },
]

