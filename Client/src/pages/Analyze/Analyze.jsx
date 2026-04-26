import { useState, useRef } from 'react'
import GlassyCard from '../../components/GlassyCard/GlassyCard'
import Stepper from '../../components/Stepper/Stepper'
import Loader from '../../components/Loader/Loader'
import ModeCard from '../../components/ModeCard/ModeCard'
import ResultCard from '../../components/ResultCard/ResultCard'
import TrustSection from '../../components/TrustSection/TrustSection'
import './Analyze.css'

function Analyze() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [mode, setMode] = useState('full')
  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState(null)
  const [step, setStep] = useState(0)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef(null)

  const modes = [
    { value: 'full', label: 'Full Analysis', desc: 'Detects all car parts and damages', icon: '🧠' },
    { value: 'damage', label: 'Damage Only', desc: 'Detects only damages', icon: '💥' },
    { value: 'parts', label: 'Car Parts Only', desc: 'Detects only car parts', icon: '🚗' }
  ]

  const handleFile = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file)
      const reader = new FileReader()
      reader.onload = (e) => setPreview(e.target.result)
      reader.readAsDataURL(file)
      setStep(1)
    }
  }

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) handleFile(file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleAnalyze = async () => {
    if (!selectedFile) return
    setAnalyzing(true)
    setResult(null)
    setStep(2)
    // Simulate API call - in production, connect to actual backend
    setTimeout(() => {
      setResult({
        requestId: `req-${Date.now()}`,
        status: 'completed',
        detections: [
          { id: '1', type: 'part', label: 'Front Bumper', confidence: 0.95 },
          { id: '2', type: 'part', label: 'Hood', confidence: 0.92 },
          { id: '3', type: 'damage', label: 'Scratch', confidence: 0.88 }
        ],
        cost: '$1,200 (est.)',
        confidence: 0.93
      })
      setAnalyzing(false)
      setStep(3)
    }, 2000)
  }

  const handleClear = () => {
    setSelectedFile(null)
    setPreview(null)
    setResult(null)
    setStep(0)
  }

  return (
    <div className="analyze">
      <div className="page-header">
        <h2>Analyze Vehicle</h2>
        <p className="subtitle">Upload an image for AI-powered vehicle analysis</p>
      </div>
      <Stepper
        steps={[
          'Upload vehicle image',
          'Choose analysis mode',
          'Run analysis',
          'View results',
        ]}
        current={step}
      />
      <div className="analyze-content">
        <GlassyCard className="upload-card">
          <h3 className="card-title">
            <span className="card-icon">📤</span>
            Upload Vehicle Image
          </h3>
          <div 
            className={`upload-area-new ${isDragging ? 'dragging' : ''} ${preview ? 'has-preview' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            {preview ? (
              <div className="preview-container-new">
                <img src={preview} alt="Preview" className="preview-image-new" />
                <button className="clear-btn-new" onClick={handleClear}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                  </svg>
                </button>
                <div className="preview-badge">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="20 6 9 17 4 12"/>
                  </svg>
                  Image Ready
                </div>
              </div>
            ) : (
              <label className="upload-label-new">
                <input 
                  type="file" 
                  accept="image/*" 
                  onChange={handleFileChange} 
                  hidden 
                  ref={fileInputRef}
                />
                <div className="upload-icon-new">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                  </svg>
                </div>
                <span className="upload-text">Drag & drop your vehicle image here</span>
                <span className="upload-or">or</span>
                <span className="upload-browse">Browse Files</span>
                <span className="upload-hint">Supports PNG, JPG up to 10MB</span>
              </label>
            )}
          </div>
          
          <div className="mode-selector-new">
            <label className="mode-label-new">
              <span className="mode-label-icon">🎯</span>
              Select Analysis Mode
            </label>
            <div className="mode-cards-new">
              {modes.map(m => (
                <ModeCard
                  key={m.value}
                  icon={m.icon}
                  label={m.label}
                  description={m.desc}
                  active={mode === m.value}
                  onClick={() => { setMode(m.value); if (selectedFile) setStep(2); }}
                />
              ))}
            </div>
          </div>
          
          <div className="actions-new">
            <button 
              className="analyze-btn"
              onClick={handleAnalyze} 
              disabled={!selectedFile || analyzing}
            >
              {analyzing ? (
                <>
                  <span className="btn-spinner"></span>
                  Analyzing...
                </>
              ) : (
                <>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"/>
                    <polyline points="12 6 12 12 16 14"/>
                  </svg>
                  Start Analysis
                </>
              )}
            </button>
          </div>
        </GlassyCard>
        
        <div className="results-column">
          {analyzing && <Loader text="Analyzing image with AI..." />}
          {result && <ResultCard result={result} />}
        </div>
      </div>
      
      <TrustSection />
    </div>
  )
}

export default Analyze