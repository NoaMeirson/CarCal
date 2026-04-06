export interface Point {
  x: number
  y: number
}

export interface Polygon {
  points: Point[]
}

export interface ImageInfo {
  width: number
  height: number
}

export interface MatchInfo {
  part: string
  score: number
  coveragePercent: number | null
}

export interface Detection {
  id: string
  type: 'part' | 'damage'
  label: string
  confidence: number
  polygon: Polygon
  matches: MatchInfo[] | null
}

export interface AnalyzeResponse {
  requestId: string
  FileName: string | null
  status: 'ok' | 'error'
  mode: string
  image: ImageInfo | null
  detections: Detection[]
  message: string | null
}

export type Mode = 'combined' | 'car_parts_only' | 'damage_only'
