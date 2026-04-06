import type { Mode } from '../types/models'

interface Props {
  value: Mode
  onChange: (mode: Mode) => void
  disabled?: boolean
}

const OPTIONS: { value: Mode; label: string }[] = [
  { value: 'combined', label: 'Combined' },
  { value: 'car_parts_only', label: 'Car Parts Only' },
  { value: 'damage_only', label: 'Damage Only' },
]

export function ModeSelector({ value, onChange, disabled }: Props) {
  return (
    <div className="flex items-center gap-2">
      <label className="text-sm font-medium text-gray-700">Mode</label>
      <select
        value={value}
        onChange={e => onChange(e.target.value as Mode)}
        disabled={disabled}
        className="rounded border border-gray-300 px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
      >
        {OPTIONS.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  )
}
