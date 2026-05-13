import { useEffect, useRef, useState } from 'react'
import { Navbar } from './components/Navbar'
import { HeroSection } from './components/HeroSection'
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
  const [heroVisible, setHeroVisible] = useState(true)
  const heroRef = useRef<HTMLDivElement>(null)
  const consoleRef = useRef<HTMLElement>(null)
  const { loading, result, error, run } = useAnalyze()

  // Navbar goes dark when hero section is in view
  useEffect(() => {
    const el = heroRef.current
    if (!el) return
    const observer = new IntersectionObserver(
      ([entry]) => setHeroVisible(entry.isIntersecting),
      { threshold: 0.15 },
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  function handleFile(f: File) {
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setSelectedId(null)
  }

  function handleSubmit() {
    if (file) run(file, mode)
  }

  function scrollToConsole() {
    consoleRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <div className="min-h-screen">
      <Navbar dark={heroVisible} />

      {/* Hero */}
      <div ref={heroRef}>
        <HeroSection onOpenConsole={scrollToConsole} />
      </div>

      {/* Console */}
      <section
        ref={consoleRef}
        id="console"
        className="min-h-screen bg-[#eaecf5] py-12"
      >
        <div className="mx-auto max-w-7xl px-8">
          <div className="flex gap-5">
            {/* ── Left column: 3 step cards ── */}
            <div className="flex w-[420px] shrink-0 flex-col gap-4">
              {/* Step 1 — Upload */}
              <StepCard
                number="1"
                title="Upload vehicle photo"
                subtitle="JPG or PNG up to ~10 MB. Clear daylight photos work best."
              >
                <ImageUploader
                  onFile={handleFile}
                  disabled={loading}
                  selectedFile={file}
                />
              </StepCard>

              {/* Step 2 — Choose analysis */}
              <StepCard
                number="2"
                title="Choose analysis"
                subtitle="Run damage, parts, or their intersection."
              >
                <ModeSelector
                  value={mode}
                  onChange={setMode}
                  disabled={loading}
                />
              </StepCard>

              {/* Step 3 — Run */}
              <StepCard number="3" title="Run">
                {error && (
                  <div className="mb-3 flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
                    <svg
                      className="mt-0.5 h-3.5 w-3.5 shrink-0 text-red-500"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path
                        fillRule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.28 7.22a.75.75 0 00-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 101.06 1.06L10 11.06l1.72 1.72a.75.75 0 101.06-1.06L11.06 10l1.72-1.72a.75.75 0 00-1.06-1.06L10 8.94 8.28 7.22z"
                        clipRule="evenodd"
                      />
                    </svg>
                    {error}
                  </div>
                )}
                <button
                  onClick={handleSubmit}
                  disabled={!file || loading}
                  className="flex w-full items-center justify-center gap-2 rounded-xl bg-[#4a78c8] py-3 text-sm font-semibold text-white transition-all hover:bg-[#3d6ab5] active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {loading ? (
                    <>
                      <svg
                        className="h-4 w-4 animate-spin"
                        viewBox="0 0 24 24"
                        fill="none"
                      >
                        <circle
                          className="opacity-25"
                          cx="12"
                          cy="12"
                          r="10"
                          stroke="currentColor"
                          strokeWidth="4"
                        />
                        <path
                          className="opacity-75"
                          fill="currentColor"
                          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                        />
                      </svg>
                      Analyzing…
                    </>
                  ) : (
                    <>
                      <svg
                        className="h-4 w-4"
                        viewBox="0 0 24 24"
                        fill="currentColor"
                      >
                        <path d="M8 5v14l11-7z" />
                      </svg>
                      Run analysis
                    </>
                  )}
                </button>
              </StepCard>
            </div>

            {/* ── Right column: Result panel ── */}
            <div className="flex-1">
              <div className="overflow-hidden rounded-2xl border border-[#d1d8e8] bg-white shadow-sm">
                {/* Card header */}
                <div className="border-b border-[#eaecf5] px-6 py-4">
                  <h2 className="text-base font-semibold text-slate-800">
                    Result
                  </h2>
                  <p className="mt-0.5 text-sm text-slate-400">
                    {result
                      ? `${result.detections.length} detection${result.detections.length !== 1 ? 's' : ''} found`
                      : 'Run an analysis to see overlays here.'}
                  </p>
                </div>

                {/* Card body */}
                {preview ? (
                  <>
                    <div className="bg-slate-900">
                      {result ? (
                        <ResultCanvas
                          imageSrc={preview}
                          imageInfo={result.image ?? { width: 0, height: 0 }}
                          detections={result.detections}
                          selectedId={selectedId}
                          onSelect={setSelectedId}
                        />
                      ) : (
                        <img
                          src={preview}
                          alt="Preview"
                          className="w-full object-contain"
                          style={{ maxHeight: '55vh' }}
                        />
                      )}
                    </div>
                    {result && result.detections.length > 0 && (
                      <div className="border-t border-[#eaecf5]">
                        <DetectionList
                          detections={result.detections}
                          selectedId={selectedId}
                          onSelect={setSelectedId}
                        />
                      </div>
                    )}
                  </>
                ) : (
                  <div className="flex min-h-[480px] flex-col items-center justify-center py-16 text-center">
                    <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-full bg-slate-100">
                      <svg
                        className="h-8 w-8 text-slate-300"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.4"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <rect x="3" y="3" width="18" height="18" rx="2" />
                        <circle cx="8.5" cy="8.5" r="1.5" />
                        <path d="M21 15l-5-5L5 21" />
                      </svg>
                    </div>
                    <p className="text-sm font-semibold text-slate-500">
                      No image yet
                    </p>
                    <p className="mt-2 max-w-[260px] text-xs leading-relaxed text-slate-400">
                      Upload a vehicle photo, pick an analysis mode, and run
                      CarCal to see damage and part overlays here.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-[#d1d8e8] bg-[#eaecf5] py-5">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-8">
          <span className="text-xs text-slate-400">
            © CarCal — AI inspection for insurers.
          </span>
          <span className="text-xs text-slate-400">
            v0.1 · Built on TanStack Start
          </span>
        </div>
      </footer>
    </div>
  )
}

function StepCard({
  number,
  title,
  subtitle,
  children,
}: {
  number: string
  title: string
  subtitle?: string
  children: React.ReactNode
}) {
  return (
    <div className="rounded-2xl border border-[#d1d8e8] bg-white p-5 shadow-sm">
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-slate-800">
          <span className="font-normal text-slate-400">{number}. </span>
          {title}
        </h3>
        {subtitle && (
          <p className="mt-0.5 text-xs text-slate-400">{subtitle}</p>
        )}
      </div>
      {children}
    </div>
  )
}
