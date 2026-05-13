import type { Detection } from '../types/models'

interface Props {
  detections: Detection[]
  selectedId: string | null
  onSelect: (id: string) => void
}

export function DetectionList({ detections, selectedId, onSelect }: Props) {
  if (detections.length === 0) return null

  const damages = detections.filter(d => d.type === 'damage')
  const parts = detections.filter(d => d.type === 'part')

  return (
    <div className="max-h-64 overflow-y-auto p-5">
      <div className="flex flex-col gap-5">
        {damages.length > 0 && (
          <Section
            title="Damage"
            accent="red"
            detections={damages}
            selectedId={selectedId}
            onSelect={onSelect}
          />
        )}
        {parts.length > 0 && (
          <Section
            title="Car Parts"
            accent="blue"
            detections={parts}
            selectedId={selectedId}
            onSelect={onSelect}
          />
        )}
      </div>
    </div>
  )
}

function Section({
  title,
  accent,
  detections,
  selectedId,
  onSelect,
}: {
  title: string
  accent: 'red' | 'blue'
  detections: Detection[]
  selectedId: string | null
  onSelect: (id: string) => void
}) {
  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <span
          className={[
            'h-2 w-2 rounded-full',
            accent === 'red' ? 'bg-red-400' : 'bg-blue-400',
          ].join(' ')}
        />
        <h3 className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">
          {title}
        </h3>
        <span className="ml-auto rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-slate-500">
          {detections.length}
        </span>
      </div>

      <ul className="flex flex-col gap-1.5">
        {detections.map(det => {
          const isSelected = det.id === selectedId
          const pct = Math.round(det.confidence * 100)
          const badgeColor =
            pct >= 80
              ? 'text-emerald-700 bg-emerald-50 ring-1 ring-emerald-200'
              : pct >= 50
              ? 'text-amber-700 bg-amber-50 ring-1 ring-amber-200'
              : 'text-slate-500 bg-slate-100 ring-1 ring-slate-200'

          return (
            <li
              key={det.id}
              onClick={() => onSelect(det.id)}
              className={[
                'relative cursor-pointer overflow-hidden rounded-lg border pl-3 pr-3 py-2.5 text-sm transition-all',
                isSelected
                  ? 'border-amber-400 bg-amber-50'
                  : accent === 'red'
                  ? 'border-red-100 bg-red-50 hover:border-red-200'
                  : 'border-blue-100 bg-blue-50 hover:border-blue-200',
              ].join(' ')}
            >
              <span
                className={[
                  'absolute inset-y-0 left-0 w-1 rounded-l-lg',
                  isSelected
                    ? 'bg-amber-400'
                    : accent === 'red'
                    ? 'bg-red-400'
                    : 'bg-blue-400',
                ].join(' ')}
              />
              <div className="flex items-center justify-between gap-2">
                <span className="font-semibold capitalize text-slate-800">
                  {det.label}
                </span>
                <span className={['shrink-0 rounded-full px-2 py-0.5 text-xs font-semibold', badgeColor].join(' ')}>
                  {pct}%
                </span>
              </div>

              {det.matches && det.matches.length > 0 && (
                <ul className="mt-2 flex flex-col gap-1 border-t border-slate-200/70 pt-2">
                  {det.matches.map((m, i) => (
                    <li
                      key={i}
                      className="flex items-center justify-between gap-2 text-xs text-slate-500"
                    >
                      <span className="capitalize text-slate-600">{m.part}</span>
                      <span className="shrink-0 tabular-nums">
                        {Math.round(m.score * 100)}% overlap
                        {m.coveragePercent != null
                          ? ` · ${m.coveragePercent.toFixed(0)}% of part`
                          : ''}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </li>
          )
        })}
      </ul>
    </div>
  )
}
