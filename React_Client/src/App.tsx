import { useState } from 'react'
import { ImageUploader } from './components/ImageUploader'
import { ModeSelector } from './components/ModeSelector'
import { ResultCanvas } from './components/ResultCanvas'
import { DetectionList } from './components/DetectionList'
import { useAnalyze } from './hooks/useAnalyze'
import type { Mode } from './types/models'

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [mode, setMode] = useState<Mode>('combined')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const { loading, result, error, run } = useAnalyze()

  function handleFile(f: File) {
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setSelectedId(null)
  }

  function handleSubmit() {
    if (file) run(file, mode)
  }

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="mx-auto max-w-5xl">
        <h1 className="mb-6 text-2xl font-bold text-gray-800">CarCal — Car Damage Analysis</h1>

        <div className="mb-4 flex flex-wrap items-end gap-4">
          <div className="flex-1">
            <ImageUploader onFile={handleFile} disabled={loading} />
          </div>
          <div className="flex flex-col gap-2">
            <ModeSelector value={mode} onChange={setMode} disabled={loading} />
            <button
              onClick={handleSubmit}
              disabled={!file || loading}
              className="rounded bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Analyzing…' : 'Analyze'}
            </button>
          </div>
        </div>

        {error && (
          <div className="mb-4 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {preview && (
          <div className="flex flex-col gap-4 lg:flex-row">
            <div className="flex-1">
              {result ? (
                <ResultCanvas
                  imageSrc={preview}
                  imageInfo={result.image ?? { width: 0, height: 0 }}
                  detections={result.detections}
                  selectedId={selectedId}
                  onSelect={setSelectedId}
                />
              ) : (
                <img src={preview} alt="Preview" className="w-full rounded-lg shadow" />
              )}
            </div>
            {result && result.detections.length > 0 && (
              <div className="w-full lg:w-72">
                <DetectionList
                  detections={result.detections}
                  selectedId={selectedId}
                  onSelect={setSelectedId}
                />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
