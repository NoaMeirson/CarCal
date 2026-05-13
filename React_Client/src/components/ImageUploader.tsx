import { useRef, useState } from 'react'

interface Props {
  onFile: (file: File) => void
  disabled?: boolean
  selectedFile?: File | null
}

export function ImageUploader({ onFile, disabled, selectedFile }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  function handleFiles(files: FileList | null) {
    const file = files?.[0]
    if (file && file.type.startsWith('image/')) onFile(file)
  }

  return (
    <div
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={e => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={e => {
        e.preventDefault()
        setDragging(false)
        if (!disabled) handleFiles(e.dataTransfer.files)
      }}
      className={[
        'flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-8 py-10 text-center transition-all',
        dragging
          ? 'border-blue-400 bg-blue-50'
          : selectedFile
          ? 'border-slate-300 bg-slate-50 hover:border-blue-400'
          : 'border-slate-300 bg-white hover:border-blue-400 hover:bg-slate-50/60',
        disabled ? 'pointer-events-none opacity-50' : '',
      ].join(' ')}
    >
      {/* Upload icon */}
      <div
        className={[
          'mb-3 flex h-10 w-10 items-center justify-center rounded-full transition-colors',
          dragging ? 'bg-blue-100' : 'bg-slate-100',
        ].join(' ')}
      >
        <svg
          className={[
            'h-5 w-5 transition-colors',
            dragging ? 'text-blue-500' : 'text-slate-400',
          ].join(' ')}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
      </div>

      {selectedFile ? (
        <>
          <p className="text-sm font-semibold text-slate-700">{selectedFile.name}</p>
          <p className="mt-1 text-xs text-slate-400">Click or drop to replace</p>
        </>
      ) : (
        <>
          <p className="text-sm font-semibold text-slate-800">
            Drop image or click to browse
          </p>
          <p className="mt-1 text-xs text-slate-400">PNG, JPG, WEBP</p>
        </>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={e => handleFiles(e.target.files)}
      />
    </div>
  )
}
