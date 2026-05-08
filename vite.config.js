import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    watch: {
      // Exclude Tauri build artifacts and sidecar binaries — the PyInstaller
      // bundle under src-tauri/binaries/_internal contains thousands of files
      // that trigger constant hot-reloads if not ignored.
      ignored: ['**/src-tauri/**'],
    },
  },
})
