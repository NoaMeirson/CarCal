import type { AnalyzeResponse, Mode } from '../types/models'

declare global {
  interface Window {
    __APP_CONFIG__?: { apiUrl?: string }
  }
}

const API_BASE = window.__APP_CONFIG__?.apiUrl || 'http://localhost:8002'
const API_URL = `${API_BASE}/analyze`

export async function analyzeImage(file: File, mode: Mode): Promise<AnalyzeResponse> {
  const buffer = await file.arrayBuffer()
  const bytes = new Uint8Array(buffer)
  let binary = ''
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i])
  }
  const base64 = btoa(binary)

  const body = {
    requestId: crypto.randomUUID(),
    FileName: file.name,
    imageBase64: base64,
    mode,
  }

  const res = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `HTTP ${res.status}`)
  }

  return res.json()
}
