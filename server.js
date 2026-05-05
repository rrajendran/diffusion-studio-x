import express from 'express'
import cors from 'cors'
import http from 'node:http'
import { spawn } from 'child_process'
import { writeFile, readFile, unlink, mkdir } from 'fs/promises'
import { tmpdir } from 'os'
import { join } from 'path'
import { fileURLToPath } from 'url'
import { dirname } from 'path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
const OUTPUT_DIR = join(__dirname, 'output')

// Ensure output directory exists
await mkdir(OUTPUT_DIR, { recursive: true })

const app = express()
app.use(cors())
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


// List locally cached HuggingFace models via `hf cache list`
app.get('/api/hf-cached-models', (req, res) => {
  const child = spawn('hf', ['cache', 'list'], { stdio: ['ignore', 'pipe', 'pipe'] })
  const chunks = []
  child.stdout.on('data', c => chunks.push(c))
  child.on('error', err => res.status(502).json({ error: `hf not found: ${err.message}` }))
  child.on('close', () => {
    const lines = Buffer.concat(chunks).toString('utf8').split('\n')
    const models = lines
      .filter(l => l.trimStart().startsWith('model/'))
      .map(l => l.trim().split(/\s+/)[0].replace(/^model\//, ''))
      .filter(Boolean)
    res.json(models)
  })
})

// Proxy image generation to the local diffusers image server (port 8001)
const IMAGE_SERVER_PORT = process.env.IMAGE_SERVER_PORT ?? 8001

app.get('/api/image/progress', (req, res) => {
  const proxyReq = http.request(
    { hostname: '127.0.0.1', port: IMAGE_SERVER_PORT, path: '/progress', method: 'GET' },
    (proxyRes) => {
      const chunks = []
      proxyRes.on('data', c => chunks.push(c))
      proxyRes.on('end', () => {
        try { res.json(JSON.parse(Buffer.concat(chunks).toString())) }
        catch { res.json({ step: 0, total: 0 }) }
      })
    }
  )
  proxyReq.on('error', () => res.json({ step: 0, total: 0 }))
  proxyReq.end()
})

app.get('/api/image/capabilities', (req, res) => {
  const options = {
    hostname: '127.0.0.1',
    port: IMAGE_SERVER_PORT,
    path: '/model-capabilities',
    method: 'GET',
  }
  const proxyReq = http.request(options, (proxyRes) => {
    const chunks = []
    proxyRes.on('data', c => chunks.push(c))
    proxyRes.on('end', () => {
      try {
        res.json(JSON.parse(Buffer.concat(chunks).toString()))
      } catch {
        res.status(502).json({ error: 'Image server: invalid JSON' })
      }
    })
  })
  proxyReq.on('error', (err) => res.status(502).json({ error: err.message }))
  proxyReq.end()
})

app.post('/api/image/generate', async (req, res) => {
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

  const body = JSON.stringify(req.body)
  const options = {
    hostname: '127.0.0.1',
    port: IMAGE_SERVER_PORT,
    path: '/generate',
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
  }
  const t0 = Date.now()
  const proxyReq = http.request(options, (proxyRes) => {
    const chunks = []
    proxyRes.on('data', c => chunks.push(c))
    proxyRes.on('end', () => {
      try {
        const data = JSON.parse(Buffer.concat(chunks).toString())
        const elapsed = ((Date.now() - t0) / 1000).toFixed(1)
        if (proxyRes.statusCode >= 400) {
          console.error(`[image:${reqId}] ✗ error | status=${proxyRes.statusCode} elapsed=${elapsed}s error="${data.error}"`)
          return res.status(proxyRes.statusCode).json(data)
        }
        console.log(`[image:${reqId}] ✓ done | mode=${data.mode ?? mode} elapsed=${data.elapsed ?? elapsed}s`)
        res.json(data)
      } catch {
        res.status(502).json({ error: 'Image server: invalid JSON response' })
      }
    })
  })
  proxyReq.on('error', (err) => {
    console.error(`[image:${reqId}] ✗ proxy error:`, err.message)
    res.status(502).json({ error: `Image server unavailable: ${err.message}` })
  })
  proxyReq.setTimeout(0)
  proxyReq.write(body)
  proxyReq.end()
})

// Proxy all requests to the local HuggingFace server to avoid browser CORS restrictions
app.use('/api/hf', async (req, res) => {
  const hfPath = req.originalUrl.replace(/^\/api\/hf/, '') || '/'
  const hfUrl = `http://localhost:${IMAGE_SERVER_PORT}${hfPath}`
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

  const filename = `ollama-${Date.now()}.png`
  const outPath = join(OUTPUT_DIR, filename)
  const tmpPath = join(tmpdir(), `ollama-${Date.now()}.png`)

  try {
    const imageData = await runOllama(model, prompt, tmpPath, width, height)
    
    // Extract base64 from data URL and save to output folder
    if (imageData.startsWith('data:')) {
      const base64Data = imageData.split(',')[1]
      await writeFile(outPath, Buffer.from(base64Data, 'base64'))
    } else {
      // Fallback: if it's a file path, read and save
      const buf = await readFile(imageData)
      await writeFile(outPath, buf)
    }
    
    res.json({ imageUrl: `/output/${filename}` })
  } catch (err) {
    res.status(500).json({ error: err.message })
  } finally {
    unlink(tmpPath).catch(() => {})
  }
})

function runOllama(model, prompt, outPath, width = 512, height = 512, _refImagePath = null) {
  // Ollama image-gen CLI does not support image input; reference image is already baked into the prompt by the LLM.
  return new Promise((resolve, reject) => {
    const args = ['run', model, prompt, '--width', String(width), '--height', String(height)]
    const child = spawn('ollama', args, { stdio: ['pipe', 'pipe', 'pipe'] })

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

const PORT = process.env.PORT ?? 3001
app.listen(PORT, () => console.log(`Ollama bridge running on http://localhost:${PORT}`))
