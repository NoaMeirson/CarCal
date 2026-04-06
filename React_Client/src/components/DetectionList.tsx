import type { Detection } from '../types/models'

interface Props {
  detections: Detection[]
  selectedId: string | null
  onSelect: (id: string) => void
}

export function DetectionList({ detections, selectedId, onSelect }: Props) {
  if (detections.length === 0) return null

  const parts = detections.filter(d => d.type === 'part')
  const damages = detections.filter(d => d.type === 'damage')

  return (
    <div className="flex flex-col gap-4 overflow-y-auto">
      {damages.length > 0 && (
        <Section title="Damage" detections={damages} selectedId={selectedId} onSelect={onSelect} />
      )}
      {parts.length > 0 && (
        <Section title="Car Parts" detections={parts} selectedId={selectedId} onSelect={onSelect} />
      )}
    </div>
  )
}

function Section({
  title,
  detections,
  selectedId,
  onSelect,
}: {
  title: string
  detections: Detection[]
  selectedId: string | null
  onSelect: (id: string) => void
}) {
  return (
    <div>
      <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-gray-500">{title}</h3>
      <ul className="flex flex-col gap-1">
        {detections.map(det => (
          <li
            key={det.id}
            onClick={() => onSelect(det.id)}
            className={[
              'cursor-pointer rounded-lg border px-3 py-2 text-sm transition-colors',
              det.id === selectedId
                ? 'border-amber-400 bg-amber-50'
                : det.type === 'damage'
                ? 'border-red-200 bg-red-50 hover:bg-red-100'
                : 'border-blue-200 bg-blue-50 hover:bg-blue-100',
            ].join(' ')}
          >
            <div className="flex items-center justify-between">
              <span className="font-medium capitalize">{det.label}</span>
              <span className="text-xs text-gray-500">{(det.confidence * 100).toFixed(0)}%</span>
            </div>
            {det.matches && det.matches.length > 0 && (
              <ul className="mt-1 flex flex-col gap-0.5 border-t border-red-200 pt-1">
                {det.matches.map((m, i) => (
                  <li key={i} className="flex items-center justify-between text-xs text-gray-600">
                    <span className="capitalize">{m.part}</span>
                    <span>
                      {(m.score * 100).toFixed(0)}% overlap
                      {m.coveragePercent != null
                        ? ` · ${m.coveragePercent.toFixed(0)}% of part`
                        : ''}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}
