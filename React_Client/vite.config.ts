import { defineConfig, type Plugin, type ViteDevServer, type PreviewServer } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'node:fs'
import path from 'node:path'

const CONFIG_PATH = path.resolve(__dirname, 'config.json')
const DEFAULT_API_URL = 'http://localhost:8002'

function serveConfigJs(server: ViteDevServer | PreviewServer) {
  server.middlewares.use('/config.js', (_req, res) => {
    let apiUrl = DEFAULT_API_URL
    try {
      apiUrl = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf-8')).apiUrl ?? apiUrl
    } catch {
      // config.json missing or invalid; fall back to default
    }
    res.setHeader('Content-Type', 'application/javascript')
    res.end(`window.__APP_CONFIG__ = ${JSON.stringify({ apiUrl })};`)
  })
}

// Serves /config.js from config.json so `npm run dev`/`preview` and the
// packaged exe (server/serve.py) read the API URL from the same place.
function runtimeConfigPlugin(): Plugin {
  return {
    name: 'runtime-config',
    configureServer: serveConfigJs,
    configurePreviewServer: serveConfigJs,
  }
}

export default defineConfig({
  plugins: [react(), runtimeConfigPlugin()],
})
