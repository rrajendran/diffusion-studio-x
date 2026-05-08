import express from 'express'
import cors from 'cors'
import http from 'node:http'
import { spawn } from 'child_process'
import { existsSync } from 'fs'
import { writeFile, readFile, unlink, mkdir, readdir, stat as statFile } from 'fs/promises'
import { tmpdir, homedir } from 'os'
import { join } from 'path'
import { fileURLToPath } from 'url'
import { dirname } from 'path'

// In native ESM, __dirname is not a built-in — derive it from import.meta.url.
// In a CJS/pkg bundle (esbuild output), import.meta is {} so .url is undefined;
// __dirname is the CJS global and is already correct (or '/snapshot/...' in pkg).
const _scriptDir = (typeof __dirname !== 'undefined' && __dirname !== '')
  ? __dirname
  : dirname(fileURLToPath(import.meta.url))

// In a pkg-packaged binary __dirname is a virtual /snapshot path — use env var or app support dir
const _isSnapshot = _scriptDir.startsWith('/snapshot')
const OUTPUT_DIR = process.env.OUTPUT_DIR
  ?? (_isSnapshot
      ? join(process.env.HOME ?? tmpdir(), 'Library', 'Application Support', 'com.diffusionstudio.x', 'output')
      : join(_scriptDir, 'output'))

// Long-running HTTP agent for image generation — undici's default headersTimeout
// is 300 s (5 min) which kills Qwen/SDXL inference that can take 10+ min.
// We use Node's built-in http.request instead of fetch for this call to avoid
// any undici-level timeout.
function httpPost(url, bodyObj, signal) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(bodyObj)
    const parsed = new URL(url)
    const options = {
      hostname: parsed.hostname,
      port: parsed.port || 80,
      path: parsed.pathname + parsed.search,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
    }
    const req = http.request(options, (res) => {
      const chunks = []
      res.on('data', c => chunks.push(c))
      res.on('end', () => {
        try {
          const text = Buffer.concat(chunks).toString()
          resolve({ status: res.statusCode, json: () => JSON.parse(text) })
        } catch (e) { reject(e) }
      })
    })
    req.on('error', reject)
    if (signal) {
      signal.addEventListener('abort', () => { req.destroy(new Error('AbortError')); reject(Object.assign(new Error('The operation was aborted'), { name: 'AbortError' })) }, { once: true })
    }
    req.write(body)
    req.end()
  })
}

const app = express()
// Allow any localhost origin and the Tauri custom scheme (tauri://localhost) used in production builds.
app.use(cors({
  origin: (origin, cb) => {
    if (!origin || /^(https?:\/\/(localhost|127\.0\.0\.1)|tauri:\/\/)/.test(origin)) {
      cb(null, true)
    } else {
      cb(null, false)
    }
  },
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: false,
}))
app.use(express.json({ limit: '50mb' }))

// Serve output folder as static files
app.use('/output', express.static(OUTPUT_DIR))

// PNG and JPEG magic bytes
const PNG_MAGIC  = Buffer.from([0x89, 0x50, 0x4e, 0x47])
const JPEG_MAGIC = Buffer.from([0xff, 0xd8, 0xff])

function detectMime(buf) {
  if (buf.slice(0, 4).equals(PNG_MAGIC))  return 'image/png'
  if (buf.slice(0, 3).equals(JPEG_MAGIC)) return 'image/jpeg'
  return null
}


// Proxy image generation to the local diffusers image server (port 8001)
const IMAGE_SERVER_PORT = process.env.IMAGE_SERVER_PORT ?? 8001
const DEFAULT_HF_BASE   = (process.env.HF_BASE_URL ?? `http://127.0.0.1:${IMAGE_SERVER_PORT}`).replace(/\/$/, '')

// Resolve HF hf-image-server base URL from request body, query param, or env/default
function getHfBase(req) {
  return (req.body?.hfBaseUrl || req.query?.hfBaseUrl || DEFAULT_HF_BASE).replace(/\/$/, '')
}
// Resolve Ollama base URL from request body or env/default
function getOllamaBase(req) {
  return (req.body?.ollamaBaseUrl || process.env.OLLAMA_BASE_URL || 'http://127.0.0.1:11434').replace(/\/$/, '')
}

// ── HF-Image-server readiness gate ───────────────────────────────────────────────
// The Python sidecar can take 5-15 s to unpack and bind its port. This promise
// resolves as soon as the /health endpoint replies, or rejects after 5 min.
let _imageServerReadyPromise = null
function imageServerReady() {
  if (!_imageServerReadyPromise) {
    _imageServerReadyPromise = new Promise((resolve, reject) => {
      const deadline = Date.now() + 300_000
      const RETRY_MS = 1_500
      function probe() {
        const req = http.request(
          { hostname: '127.0.0.1', port: IMAGE_SERVER_PORT, path: '/health', method: 'GET' },
          (res) => {
            res.resume()
            if (res.statusCode < 500) {
              console.log(`[bridge] hf-image-server ready on port ${IMAGE_SERVER_PORT}`)
              resolve()
            } else {
              scheduleRetry()
            }
          }
        )
        req.on('error', scheduleRetry)
        req.setTimeout(1000, () => { req.destroy(); scheduleRetry() })
        req.end()
      }
      function scheduleRetry() {
        if (Date.now() >= deadline) {
          _imageServerReadyPromise = null  // reset so next request retries
          reject(new Error('Image server did not start within 5 min'))
        } else {
          setTimeout(probe, RETRY_MS)
        }
      }
      probe()
    })
  }
  return _imageServerReadyPromise
}
imageServerReady().catch(() => {})

// Wait for the HF image server — skipped when a custom URL is provided since
// the user manages that server themselves.
async function waitForHf(hfBase) {
  if (hfBase !== DEFAULT_HF_BASE) return
  return imageServerReady()
}

async function scanHFCache() {
  const cacheDir = process.env.HF_HOME
    ?? join(homedir(), '.cache', 'huggingface', 'hub')
  let entries
  try {
    entries = await readdir(cacheDir)
  } catch {
    return []
  }
  return entries
    .filter(e => e.startsWith('models--'))
    .map(e => e.slice('models--'.length).replace('--', '/'))
}

app.get('/health', (_req, res) => res.json({ status: 'ok' }))

app.get('/api/hf-cached-models', async (req, res) => {
  const models = await scanHFCache()
  res.json(models)
})

app.get('/api/image/progress', async (req, res) => {
  const hfBase = getHfBase(req)
  try { await waitForHf(hfBase) } catch { return res.json({ step: 0, total: 0 }) }
  try {
    const r = await fetch(`${hfBase}/progress`)
    res.json(await r.json())
  } catch { res.json({ step: 0, total: 0 }) }
})

app.get('/api/image/capabilities', async (req, res) => {
  const hfBase = getHfBase(req)
  try { await waitForHf(hfBase) } catch (e) { return res.status(503).json({ error: e.message }) }
  try {
    const r = await fetch(`${hfBase}/model-capabilities`)
    if (!r.ok) return res.status(r.status).json({ error: `Image server returned ${r.status}` })
    res.json(await r.json())
  } catch (err) { res.status(502).json({ error: err.message }) }
})
// Trigger background preload of a model in the image server
app.post('/api/image/preload', async (req, res) => {
  const hfBase = getHfBase(req)
  try { await waitForHf(hfBase) } catch (e) { return res.status(503).json({ error: e.message }) }
  try {
    const r = await fetch(`${hfBase}/preload`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    })
    try { res.json(await r.json()) } catch { res.json({ status: 'ok' }) }
  } catch (err) { res.status(502).json({ error: err.message }) }
})
app.post('/api/image/generate', async (req, res) => {
  const hfBase = getHfBase(req)
  try { await waitForHf(hfBase) } catch (e) { return res.status(503).json({ error: `Image server not ready: ${e.message}` }) }
  let { prompt = '', model = '', width, height, reference_image } = req.body
  const mode = reference_image ? 'i2i' : 't2i'
  const reqId = Date.now()
  console.log(`[image:${reqId}] → generate | model=${model} ${width}x${height} mode=${mode} prompt="${prompt.slice(0, 60)}"`)

  // Convert a saved /output/<file> web path back to a base64 data URL so the
  // Python server can decode it.  This happens when lastImageUrl (from a prior
  // generation that was persisted to disk) is reused as a reference image.
  if (reference_image && reference_image.startsWith('/output/')) {
    try {
      const filePath = join(OUTPUT_DIR, reference_image.replace(/^\/output\//, ''))
      const buf = await readFile(filePath)
      const mime = detectMime(buf) ?? 'image/jpeg'
      reference_image = `data:${mime};base64,${buf.toString('base64')}`
      req.body = { ...req.body, reference_image }
    } catch (err) {
      console.warn(`[image:${reqId}] could not resolve reference_image path: ${err.message}`)
    }
  }

  // Image generation can take several minutes for large models.
  // Default timeout = 21 min (just over the Python 20-min INFERENCE_TIMEOUT so the
  // Python 504 always reaches the client before the bridge itself aborts).
  // Override with IMAGE_GEN_TIMEOUT_MS env var.
  const GEN_TIMEOUT_MS = parseInt(process.env.IMAGE_GEN_TIMEOUT_MS ?? '1260000', 10)
  const abort = new AbortController()
  const timer = setTimeout(() => abort.abort(), GEN_TIMEOUT_MS)

  const t0 = Date.now()
  try {
    const r = await httpPost(`${hfBase}/generate`, req.body, abort.signal)
    clearTimeout(timer)
    const data = r.json()
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1)
    if (r.status >= 400) {
      console.error(`[image:${reqId}] ✗ error | status=${r.status} elapsed=${elapsed}s error="${data.error}"`)
      return res.status(r.status).json(data)
    }
    console.log(`[image:${reqId}] ✓ done | mode=${data.mode ?? mode} elapsed=${data.elapsed ?? elapsed}s`)
    res.json(data)
  } catch (err) {
    clearTimeout(timer)
    if (err.name === 'AbortError') {
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1)
      console.error(`[image:${reqId}] ✗ timeout after ${elapsed}s (limit=${GEN_TIMEOUT_MS / 1000}s)`)
      return res.status(504).json({ error: `Image generation timed out after ${elapsed}s` })
    }
    console.error(`[image:${reqId}] ✗ proxy error:`, err.message)
    res.status(502).json({ error: `Image server unavailable: ${err.message}` })
  }
})

// Proxy all requests to the local HuggingFace server to avoid browser CORS restrictions
app.use('/api/hf', async (req, res) => {
  const hfBase = getHfBase(req)
  try { await waitForHf(hfBase) } catch (e) { return res.status(503).json({ error: e.message }) }
  const hfPath = req.originalUrl.replace(/^\/api\/hf/, '') || '/'
  const hfUrl = `${hfBase}${hfPath}`
  console.log(`[HF proxy] ${req.method} ${hfUrl}`, req.method !== 'GET' ? JSON.stringify(req.body).slice(0, 300) : '')
  try {
    const headers = { 'Content-Type': 'application/json' }
    if (req.headers['authorization']) headers['Authorization'] = req.headers['authorization']
    const fetchOpts = { method: req.method, headers }
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      fetchOpts.body = JSON.stringify(req.body)
    }
    const hfRes = await fetch(hfUrl, fetchOpts)
    const contentType = hfRes.headers.get('content-type') ?? ''
    const text = await hfRes.text()
    console.log(`[HF proxy] response ${hfRes.status}:`, text.slice(0, 300))
    if (contentType.includes('application/json') || contentType.includes('text/') || hfRes.status >= 400) {
      res.status(hfRes.status).set('Content-Type', contentType || 'application/json').send(text)
    } else {
      const buf = Buffer.from(text, 'binary')
      res.status(hfRes.status).set('Content-Type', contentType).send(buf)
    }
  } catch (err) {
    console.error(`[HF proxy] error:`, err.message)
    res.status(502).json({ error: `HuggingFace proxy error: ${err.message}` })
  }
})

app.post('/api/lmstudio/generate', async (req, res) => {
  const { prompt, model, width = 512, height = 512, referenceImageUrl = null } = req.body
  if (!prompt) return res.status(400).json({ error: 'prompt is required' })

  const messages = referenceImageUrl
    ? [{ role: 'user', content: [
        { type: 'image_url', image_url: { url: referenceImageUrl } },
        { type: 'text', text: prompt },
      ] }]
    : [{ role: 'user', content: prompt }]

  try {
    const lmRes = await fetch('http://localhost:1234/v1/images/generations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: model || 'default', prompt, messages, n: 1, size: `${width}x${height}` }),
    })

    if (!lmRes.ok) {
      const errText = await lmRes.text()
      console.error(`[LMStudio] error ${lmRes.status}:`, errText.slice(0, 300))
      return res.status(lmRes.status).json({ error: errText })
    }

    const data = await lmRes.json()
    const raw = data.data?.[0]?.url || data.data?.[0]?.b64_json
    if (!raw) return res.status(500).json({ error: 'LM Studio returned no image data' })

    // If it's already a URL, pass through; otherwise save base64 to output dir
    if (raw.startsWith('http')) return res.json({ imageUrl: raw })

    const b64 = raw.startsWith('data:') ? raw.split(',')[1] : raw
    const filename = `lmstudio-${Date.now()}.png`
    const outPath = join(OUTPUT_DIR, filename)
    await writeFile(outPath, Buffer.from(b64, 'base64'))
    res.json({ imageUrl: `/output/${filename}` })
  } catch (err) {
    console.error('[LMStudio] fetch error:', err.message)
    res.status(502).json({ error: `LM Studio bridge error: ${err.message}` })
  }
})

app.post('/api/ollama/generate', async (req, res) => {
  const { prompt, model = 'stable-diffusion', width = 512, height = 512, referenceImageUrl = null } = req.body
  if (!prompt) return res.status(400).json({ error: 'prompt is required' })

  const ollamaBase = getOllamaBase(req)
  const slug = prompt.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 60)
  const filename = `${slug}-${Date.now()}.png`
  const outPath = join(OUTPUT_DIR, filename)

  try {
    const body = {
      model,
      prompt,
      stream: false,
      width,
      height,
    }

    if (referenceImageUrl) {
      // Resolve a saved /output/<file> web path to real base64 before sending to Ollama.
      // Without this, the raw path string is sent as "base64" and Ollama's Go decoder
      // throws "illegal base64 data at input byte 4" on the second generation.
      let resolvedRef = referenceImageUrl
      if (referenceImageUrl.startsWith('/output/')) {
        try {
          const filePath = join(OUTPUT_DIR, referenceImageUrl.replace(/^\/output\//, ''))
          const buf = await readFile(filePath)
          resolvedRef = buf.toString('base64')
        } catch (err) {
          console.warn(`[ollama] could not resolve reference image path: ${err.message}`)
          resolvedRef = null
        }
      } else {
        resolvedRef = referenceImageUrl.includes(',') ? referenceImageUrl.split(',')[1] : referenceImageUrl
      }
      if (resolvedRef) body.images = [resolvedRef]
    }

    const ollamaRes = await fetch(`${ollamaBase}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })

    const rawText = await ollamaRes.text()
    if (!rawText || !rawText.trim()) {
      throw new Error(`Ollama returned an empty response (status ${ollamaRes.status})`)
    }

    // Ollama may stream NDJSON even with stream:false on some versions —
    // take only the last non-empty line so JSON.parse gets a complete object.
    const lastLine = rawText.trim().split('\n').filter(Boolean).at(-1)
    let data
    try {
      data = JSON.parse(lastLine)
    } catch {
      throw new Error(`Ollama returned invalid JSON: ${rawText.slice(0, 300)}`)
    }

    if (!ollamaRes.ok) throw new Error(data.error ?? `Ollama error: ${ollamaRes.status}`)

    if (!data.image) throw new Error('Ollama returned no image data')
    // console.log(`[Ollama] generation successful, saving image... ${data.image.slice(0, 30)}...`)
    const imageBuffer = Buffer.from(data.image, 'base64')
    await writeFile(outPath, imageBuffer)
    res.json({
      imageUrl: `/output/${filename}`,
      meta: {
        model,
        width,
        height,
        seed: null,
        steps: null,
        guidance_scale: null,
      },
    })
  } catch (err) {
    console.error('[ollama] generate error:', err.message)
    res.status(500).json({ error: err.message })
  }
})

function findOllama() {
  const candidates = [
    '/usr/local/bin/ollama',
    '/opt/homebrew/bin/ollama',
    '/opt/homebrew/sbin/ollama',
    '/usr/bin/ollama',
    '/Applications/Ollama.app/Contents/Resources/ollama',
  ]
  for (const p of candidates) {
    if (existsSync(p)) return p
  }
  return 'ollama'
}

function runOllama(model, prompt, outPath, width = 512, height = 512, _refImagePath = null) {
  // Ollama image-gen CLI does not support image input; reference image is already baked into the prompt by the LLM.
  return new Promise((resolve, reject) => {
    const args = ['run', model, prompt, '--width', String(width), '--height', String(height)]
    const ollamaBin = findOllama()
    // Augment PATH so the ollama binary is found in common install locations even when
    // launched from a Tauri sidecar (which gets a minimal macOS/Linux GUI environment).
    const spawnEnv = {
      ...process.env,
      PATH: `/usr/local/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/bin:/bin:/usr/sbin:/sbin:${process.env.PATH ?? ''}`,
    }
    const child = spawn(ollamaBin, args, { stdio: ['pipe', 'pipe', 'pipe'], env: spawnEnv })

    const stdoutChunks = []
    const stderrChunks = []

    child.stdout.on('data', chunk => stdoutChunks.push(chunk))
    child.stderr.on('data', chunk => stderrChunks.push(chunk))

    child.on('error', err => {
      if (err.code === 'ENOENT') {
        reject(new Error('ollama not found — make sure it is installed and on your PATH'))
      } else {
        reject(err)
      }
    })

    child.on('close', async (code) => {
      const stdout = Buffer.concat(stdoutChunks)
      const stderr = Buffer.concat(stderrChunks).toString('utf8').trim()

      // Case 1: stdout is raw binary image data
      const mime = detectMime(stdout)
      if (mime) {
        return resolve(`data:${mime};base64,${stdout.toString('base64')}`)
      }

      // Case 2: stdout contains a file path to the generated image
      const stdoutText = stdout.toString('utf8').trim()
      const pathMatch  = stdoutText.match(/([^\s]+\.(?:png|jpg|jpeg))/i)
      if (pathMatch) {
        try {
          const buf = await readFile(pathMatch[1])
          const fileMime = detectMime(buf) ?? 'image/png'
          return resolve(`data:${fileMime};base64,${buf.toString('base64')}`)
        } catch {
          // fall through to error
        }
      }

      // Case 3: look for a base64-ish string in stdout
      const b64Match = stdoutText.match(/[A-Za-z0-9+/]{100,}={0,2}/)
      if (b64Match) {
        return resolve(`data:image/png;base64,${b64Match[0]}`)
      }

      reject(new Error(
        `ollama exited with code ${code} and returned no image data.\n` +
        (stderr ? `stderr: ${stderr.slice(0, 300)}` : `stdout preview: "${stdoutText.slice(0, 300)}"`)
      ))
    })

    child.stdin.end()
  })
}

// Extract a human-readable prompt from an output filename by stripping the trailing timestamp/date suffix
function promptFromFilename(filename) {
  const base = filename.replace(/\.[^.]+$/, '')              // remove extension
  const slug = base
    .replace(/-\d{8}-\d{6}$/, '')   // date-time suffix (YYYYMMDD-HHMMSS)
    .replace(/-\d{10,}$/, '')        // unix-ms or unix-s timestamp suffix
  return slug.replace(/-/g, ' ')
}

app.get('/api/gallery', async (req, res) => {
  try {
    const files = await readdir(OUTPUT_DIR)
    const imageFiles = files.filter(f => /\.(png|jpe?g|webp)$/i.test(f))
    const items = await Promise.all(
      imageFiles.map(async (filename) => {
        const s = await statFile(join(OUTPUT_DIR, filename))
        return {
          filename,
          url: `/output/${filename}`,
          prompt: promptFromFilename(filename),
          timestamp: s.mtime.getTime(),
        }
      })
    )
    items.sort((a, b) => b.timestamp - a.timestamp)
    res.json(items)
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

app.post('/api/save-image', async (req, res) => {
  const { imageUrl, prompt = '' } = req.body
  if (!imageUrl?.startsWith('data:')) return res.status(400).json({ error: 'Expected a data: URL' })

  const mime = imageUrl.match(/^data:([^;]+);base64,/)?.[1] ?? 'image/png'
  const ext = mime === 'image/jpeg' ? 'jpg' : 'png'
  const slug = prompt.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 60)
  const filename = `${slug}-${Date.now()}.${ext}`
  const b64 = imageUrl.split(',')[1]
  const outPath = join(OUTPUT_DIR, filename)
  await writeFile(outPath, Buffer.from(b64, 'base64'))
  console.log(`[save-image] saved → output/${filename}`)
  res.json({ imageUrl: `/output/${filename}` })
})

const PORT = process.env.BRIDGE_PORT ?? process.env.PORT ?? 3001
mkdir(OUTPUT_DIR, { recursive: true })
  .then(() => app.listen(PORT, () => console.log(`Ollama bridge running on http://localhost:${PORT}`)))
  .catch(err => { console.error('Failed to create output dir:', err); process.exit(1) })
