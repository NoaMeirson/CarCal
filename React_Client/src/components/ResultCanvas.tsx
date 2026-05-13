import { useEffect, useRef, useState } from 'react'
import type { Detection, ImageInfo } from '../types/models'

interface Props {
  imageSrc: string
  imageInfo: ImageInfo
  detections: Detection[]
  selectedId: string | null
  onSelect: (id: string) => void
}

const PALETTE = {
  part:     { base: '#3b82f6', hover: '#1d4ed8' },
  damage:   { base: '#ef4444', hover: '#b91c1c' },
  selected: { base: '#f59e0b', hover: '#d97706' },
} as const

export function ResultCanvas({ imageSrc, detections, selectedId, onSelect }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)
  const [hoverId, setHoverId] = useState<string | null>(null)

  // Keep refs in sync so draw() always reads fresh values even in async callbacks
  const detectionsRef = useRef(detections)
  const selectedIdRef = useRef(selectedId)
  const hoverIdRef = useRef(hoverId)
  detectionsRef.current = detections
  selectedIdRef.current = selectedId
  hoverIdRef.current = hoverId

  function draw() {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dets = detectionsRef.current
    const selId = selectedIdRef.current
    const hovId = hoverIdRef.current

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, 0, 0)

    // Render in layers: normal → hovered → selected (topmost last)
    for (const det of dets) {
      if (det.id === selId || det.id === hovId) continue
      const pal = PALETTE[det.type as 'part' | 'damage'] ?? PALETTE.part
      drawPoly(ctx, det, pal.base, pal.base, '18', 1.5, false)
    }
    if (hovId && hovId !== selId) {
      const det = dets.find(d => d.id === hovId)
      if (det) {
        const pal = PALETTE[det.type as 'part' | 'damage'] ?? PALETTE.part
        drawPoly(ctx, det, pal.hover, pal.base, '45', 2.5, true)
      }
    }
    if (selId) {
      const det = dets.find(d => d.id === selId)
      if (det) {
        const isAlsoHovered = det.id === hovId
        drawPoly(ctx, det, PALETTE.selected.hover, PALETTE.selected.base, isAlsoHovered ? '55' : '38', 3, true)
      }
    }
  }

  function drawPoly(
    ctx: CanvasRenderingContext2D,
    det: Detection,
    strokeColor: string,
    fillColor: string,
    fillAlpha: string,
    lineWidth: number,
    highlight: boolean,
  ) {
    const pts = det.polygon.points
    if (pts.length < 2) return

    ctx.save()
    if (highlight) {
      ctx.shadowColor = strokeColor
      ctx.shadowBlur = 20
    }

    ctx.beginPath()
    ctx.moveTo(pts[0].x, pts[0].y)
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y)
    ctx.closePath()
    ctx.fillStyle = fillColor + fillAlpha
    ctx.fill()
    ctx.strokeStyle = strokeColor
    ctx.lineWidth = lineWidth
    ctx.stroke()
    ctx.restore()

    if (highlight) drawLabel(ctx, det, strokeColor)
  }

  function drawLabel(ctx: CanvasRenderingContext2D, det: Detection, color: string) {
    const pts = det.polygon.points
    const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length
    const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length

    const fontSize = Math.max(14, Math.round(ctx.canvas.width / 38))
    const pad = Math.round(fontSize * 0.45)
    ctx.font = `600 ${fontSize}px system-ui, -apple-system, sans-serif`
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'

    const text = det.label.charAt(0).toUpperCase() + det.label.slice(1)
    const tw = ctx.measureText(text).width
    const bx = cx - tw / 2 - pad
    const by = cy - fontSize / 2 - pad
    const bw = tw + pad * 2
    const bh = fontSize + pad * 2
    const r = 5

    // Rounded-rect background
    ctx.save()
    ctx.beginPath()
    ctx.moveTo(bx + r, by)
    ctx.lineTo(bx + bw - r, by)
    ctx.quadraticCurveTo(bx + bw, by, bx + bw, by + r)
    ctx.lineTo(bx + bw, by + bh - r)
    ctx.quadraticCurveTo(bx + bw, by + bh, bx + bw - r, by + bh)
    ctx.lineTo(bx + r, by + bh)
    ctx.quadraticCurveTo(bx, by + bh, bx, by + bh - r)
    ctx.lineTo(bx, by + r)
    ctx.quadraticCurveTo(bx, by, bx + r, by)
    ctx.closePath()
    ctx.fillStyle = color
    ctx.globalAlpha = 0.88
    ctx.fill()
    ctx.restore()

    ctx.globalAlpha = 1
    ctx.fillStyle = '#ffffff'
    ctx.fillText(text, cx, cy)
    ctx.textAlign = 'start'
    ctx.textBaseline = 'alphabetic'
  }

  // Load image; reset hover so stale polygon labels don't linger on a new image
  useEffect(() => {
    setHoverId(null)
    const img = new Image()
    img.src = imageSrc
    imgRef.current = img
    img.onload = () => {
      const canvas = canvasRef.current
      if (canvas) {
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight
      }
      draw()
    }
  }, [imageSrc])

  // Redraw when detections, selection, or hover changes
  useEffect(() => {
    if (imgRef.current?.complete && imgRef.current.naturalWidth > 0) draw()
  }, [detections, selectedId, hoverId])

  function getCanvasCoords(e: React.MouseEvent<HTMLCanvasElement>): [number, number] {
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    return [
      (e.clientX - rect.left) * (canvas.width / rect.width),
      (e.clientY - rect.top) * (canvas.height / rect.height),
    ]
  }

  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const [cx, cy] = getCanvasCoords(e)
    let found: string | null = null
    for (let i = detectionsRef.current.length - 1; i >= 0; i--) {
      if (pointInPolygon(cx, cy, detectionsRef.current[i].polygon.points)) {
        found = detectionsRef.current[i].id
        break
      }
    }
    setHoverId(prev => (prev === found ? prev : found))
  }

  function handleMouseLeave() {
    setHoverId(null)
  }

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const [cx, cy] = getCanvasCoords(e)
    for (let i = detectionsRef.current.length - 1; i >= 0; i--) {
      if (pointInPolygon(cx, cy, detectionsRef.current[i].polygon.points)) {
        onSelect(detectionsRef.current[i].id)
        return
      }
    }
  }

  return (
    <canvas
      ref={canvasRef}
      onClick={handleClick}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      className="w-full cursor-pointer rounded-2xl"
      style={{ maxHeight: '70vh', objectFit: 'contain' }}
    />
  )
}

function pointInPolygon(x: number, y: number, pts: { x: number; y: number }[]): boolean {
  let inside = false
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const xi = pts[i].x, yi = pts[i].y
    const xj = pts[j].x, yj = pts[j].y
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
      inside = !inside
    }
  }
  return inside
}
