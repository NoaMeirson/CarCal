import type { AnalyzeResponse, Mode } from '../types/models'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8002/analyze'

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
