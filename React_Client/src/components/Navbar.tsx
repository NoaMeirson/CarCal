interface Props {
  dark: boolean
}

export function Navbar({ dark }: Props) {
  return (
    <nav
      className={[
        'fixed top-0 z-50 w-full transition-all duration-500',
        dark
          ? ''
          : 'border-b border-[#d1d8e8] bg-[#eaecf5]/95 backdrop-blur-sm',
      ].join(' ')}
    >
      <div className="mx-auto flex max-w-7xl items-center px-8 py-4">
        <div className="flex items-center gap-3">
          <div
            className={[
              'flex h-8 w-8 items-center justify-center rounded-lg',
              dark ? 'bg-white/15' : 'bg-[#1e3a8a]',
            ].join(' ')}
          >
            <svg
              className="h-4 w-4 text-white"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>

          <span
            className={[
              'text-sm font-bold tracking-tight',
              dark ? 'text-white' : 'text-slate-900',
            ].join(' ')}
          >
            CarCal
          </span>

          <span
            className={[
              'rounded-full border px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-widest',
              dark
                ? 'border-white/20 text-white/50'
                : 'border-slate-300 text-slate-400',
            ].join(' ')}
          >
            Inspection Console
          </span>
        </div>

        <div className="ml-auto flex items-center gap-8">
          {(['Workflow', 'API', 'Contact'] as const).map(link => (
            <a
              key={link}
              href={link === 'API' ? '#console' : '#'}
              className={[
                'text-sm font-medium transition-colors',
                dark
                  ? 'text-white/70 hover:text-white'
                  : 'text-slate-500 hover:text-slate-900',
              ].join(' ')}
            >
              {link}
            </a>
          ))}
        </div>
      </div>
    </nav>
  )
}
