import type { Mode } from '../types/models'

interface Props {
  value: Mode
  onChange: (mode: Mode) => void
  disabled?: boolean
}

const OPTIONS: {
  value: Mode
  label: string
  description: string
  icon: React.ReactNode
}[] = [
  {
    value: 'damage_only',
    label: 'Damage only',
    description: 'Locate impact areas',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="h-5 w-5">
        <circle cx="10" cy="10" r="7" />
        <circle cx="10" cy="10" r="3" />
        <line x1="10" y1="1" x2="10" y2="3" />
        <line x1="10" y1="17" x2="10" y2="19" />
        <line x1="1" y1="10" x2="3" y2="10" />
        <line x1="17" y1="10" x2="19" y2="10" />
      </svg>
    ),
  },
  {
    value: 'car_parts_only',
    label: 'Car parts only',
    description: 'Segment vehicle parts',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="h-5 w-5">
        <path d="M3 13v-2l2-4h10l2 4v2" />
        <path d="M5 7l1.5-3h7L15 7" />
        <circle cx="6.5" cy="13.5" r="1.5" />
        <circle cx="13.5" cy="13.5" r="1.5" />
      </svg>
    ),
  },
  {
    value: 'combined',
    label: 'Combined',
    description: 'Damage × part overlap',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="h-5 w-5">
        <rect x="2" y="7" width="11" height="9" rx="1.5" />
        <rect x="7" y="4" width="11" height="9" rx="1.5" />
      </svg>
    ),
  },
]

export function ModeSelector({ value, onChange, disabled }: Props) {
  return (
    <div className="grid grid-cols-3 gap-2.5">
      {OPTIONS.map(opt => {
        const selected = value === opt.value
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => !disabled && onChange(opt.value)}
            disabled={disabled}
            className={[
              'flex flex-col items-start gap-2.5 rounded-xl border p-3.5 text-left transition-all',
              selected
                ? 'border-blue-500 bg-[#172554]'
                : 'border-slate-200 bg-white hover:border-blue-300',
              disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer',
            ].join(' ')}
          >
            <div
              className={[
                'flex h-8 w-8 items-center justify-center rounded-lg',
                selected ? 'bg-blue-500/25' : 'bg-slate-100',
              ].join(' ')}
            >
              <span className={selected ? 'text-white' : 'text-slate-500'}>
                {opt.icon}
              </span>
            </div>
            <div>
              <p
                className={[
                  'text-xs font-semibold leading-tight',
                  selected ? 'text-white' : 'text-slate-800',
                ].join(' ')}
              >
                {opt.label}
              </p>
              <p
                className={[
                  'mt-0.5 text-[11px] leading-tight',
                  selected ? 'text-blue-200' : 'text-slate-400',
                ].join(' ')}
              >
                {opt.description}
              </p>
            </div>
          </button>
        )
      })}
    </div>
  )
}
