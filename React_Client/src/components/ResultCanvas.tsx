import { useEffect, useRef } from 'react'
import type { Detection, ImageInfo } from '../types/models'

interface Props {
  imageSrc: string
  imageInfo: ImageInfo
  detections: Detection[]
  selectedId: string | null
  onSelect: (id: string) => void
}

const COLORS = {
  part: { stroke: '#3b82f6', fill: '#3b82f620' },
  damage: { stroke: '#ef4444', fill: '#ef444420' },
  selected: { stroke: '#f59e0b', fill: '#f59e0b30' },
}

export function ResultCanvas({ imageSrc, imageInfo, detections, selectedId, onSelect }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = new Image()
    img.src = imageSrc
    imgRef.current = img

    img.onload = () => {
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      draw(ctx, img)
    }
  }, [imageSrc, detections, selectedId])

  function draw(ctx: CanvasRenderingContext2D, img: HTMLImageElement) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    ctx.drawImage(img, 0, 0)

    for (const det of detections) {
      const isSelected = det.id === selectedId
      const color = isSelected ? COLORS.selected : COLORS[det.type as 'part' | 'damage'] ?? COLORS.part
      const pts = det.polygon.points
      if (pts.length < 2) continue

      ctx.beginPath()
      ctx.moveTo(pts[0].x, pts[0].y)
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y)
      ctx.closePath()
      ctx.strokeStyle = color.stroke
      ctx.lineWidth = isSelected ? 3 : 2
      ctx.fillStyle = color.fill
      ctx.fill()
      ctx.stroke()
    }
  }

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const cx = (e.clientX - rect.left) * scaleX
    const cy = (e.clientY - rect.top) * scaleY

    // Hit-test in reverse order so topmost polygon wins
    for (let i = detections.length - 1; i >= 0; i--) {
      const det = detections[i]
      if (pointInPolygon(cx, cy, det.polygon.points)) {
        onSelect(det.id)
        return
      }
    }
  }

  return (
    <canvas
      ref={canvasRef}
      onClick={handleClick}
      className="w-full cursor-pointer rounded-lg shadow"
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
