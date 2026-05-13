interface Props {
  onOpenConsole: () => void
}

export function HeroSection({ onOpenConsole }: Props) {
  return (
    <section
      className="relative flex min-h-screen items-center overflow-hidden"
      style={{
        background:
          'linear-gradient(140deg, #060e1f 0%, #0b1838 35%, #0e2255 65%, #112970 100%)',
      }}
    >
      {/* Radial glow */}
      <div className="pointer-events-none absolute inset-0">
        <div
          className="absolute inset-0"
          style={{
            background:
              'radial-gradient(ellipse at 72% 44%, rgba(37,99,235,0.38) 0%, transparent 60%)',
          }}
        />
        <div
          className="absolute"
          style={{
            right: '14%',
            top: '22%',
            width: 320,
            height: 320,
            background:
              'radial-gradient(circle, rgba(96,165,250,0.18) 0%, transparent 70%)',
          }}
        />
      </div>

      {/* Subtle dot grid */}
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.028]"
        style={{
          backgroundImage:
            'radial-gradient(circle, #fff 1px, transparent 1px)',
          backgroundSize: '32px 32px',
        }}
      />

      <div className="relative z-10 mx-auto grid max-w-7xl grid-cols-2 items-center gap-12 px-8 py-28">
        {/* Left — text */}
        <div>
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-blue-500/25 bg-blue-500/10 px-4 py-1.5">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-blue-400" />
            <span className="text-xs font-medium tracking-wide text-blue-300">
              AI Vehicle Inspection
            </span>
          </div>

          <h1 className="text-5xl font-extrabold leading-[1.12] tracking-tight text-white">
            Pinpoint damage.
            <br />
            Identify the part.
            <br />
            <span className="text-blue-400">In seconds.</span>
          </h1>

          <p className="mt-7 max-w-[430px] text-base leading-relaxed text-slate-400">
            CarCal combines damage detection and part segmentation so claim
            adjusters know exactly what was hit — not just that something was.
            Built for car insurers.
          </p>

          <div className="mt-10">
            <button
              onClick={onOpenConsole}
              className="inline-flex items-center gap-2.5 rounded-xl bg-blue-600 px-7 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-900/40 transition-all hover:bg-blue-500"
            >
              Open Inspection Console
              <svg
                className="h-4 w-4"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M12 5v14M5 12l7 7 7-7" />
              </svg>
            </button>
          </div>
        </div>

        {/* Right — car illustration */}
        <div className="flex items-center justify-center">
          <CarIllustration />
        </div>
      </div>
    </section>
  )
}

function CarIllustration() {
  return (
    <svg
      viewBox="0 0 560 300"
      className="w-full max-w-xl"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <radialGradient id="cg1" cx="50%" cy="70%" r="50%">
          <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.25" />
          <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
        </radialGradient>
        <filter id="cf1" x="-25%" y="-25%" width="150%" height="150%">
          <feGaussianBlur stdDeviation="4" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <filter id="cf2" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="9" />
        </filter>
      </defs>

      {/* Ground reflection */}
      <ellipse cx="280" cy="272" rx="225" ry="14" fill="rgba(37,99,235,0.1)" />

      {/* Body ambient glow */}
      <ellipse cx="280" cy="210" rx="210" ry="60" fill="url(#cg1)" />

      {/* Main body */}
      <path
        d="M 55 224 L 55 194 C 55 189 68 183 85 171 L 136 120 C 153 101 196 88 249 85 L 337 85 C 375 85 409 95 425 111 L 461 141 L 475 148 L 478 166 L 479 192 L 479 224 Z"
        fill="rgba(255,255,255,0.065)"
        stroke="rgba(147,197,253,0.55)"
        strokeWidth="1.5"
      />

      {/* Cabin */}
      <path
        d="M 149 171 L 189 124 C 203 108 241 94 276 91 L 339 91 C 373 91 403 102 417 114 L 447 144 L 149 144 Z"
        fill="rgba(147,197,253,0.07)"
        stroke="rgba(147,197,253,0.3)"
        strokeWidth="1"
      />

      {/* Front window */}
      <path
        d="M 193 140 L 223 107 C 235 95 259 93 279 91 L 279 140 Z"
        fill="rgba(147,197,253,0.14)"
        stroke="rgba(147,197,253,0.35)"
        strokeWidth="1"
      />

      {/* Rear window */}
      <path
        d="M 279 91 L 331 92 C 359 93 391 104 407 117 L 431 140 L 279 140 Z"
        fill="rgba(147,197,253,0.14)"
        stroke="rgba(147,197,253,0.35)"
        strokeWidth="1"
      />

      {/* Door pillar */}
      <line
        x1="279"
        y1="92"
        x2="279"
        y2="192"
        stroke="rgba(147,197,253,0.22)"
        strokeWidth="1"
      />

      {/* Body crease line */}
      <path
        d="M 86 194 Q 282 185 472 194"
        stroke="rgba(255,255,255,0.11)"
        strokeWidth="1.5"
        fill="none"
      />

      {/* Side mirror */}
      <path
        d="M 429 139 L 441 132 L 445 141 L 433 146 Z"
        fill="rgba(255,255,255,0.12)"
        stroke="rgba(147,197,253,0.3)"
        strokeWidth="1"
      />

      {/* Front wheel arch */}
      <path
        d="M 62 220 C 62 191 82 171 141 171 C 200 171 220 191 220 220"
        fill="rgba(5,12,26,0.7)"
        stroke="rgba(147,197,253,0.4)"
        strokeWidth="1.5"
      />

      {/* Rear wheel arch */}
      <path
        d="M 339 220 C 339 191 359 171 411 171 C 463 171 479 191 479 220"
        fill="rgba(5,12,26,0.7)"
        stroke="rgba(147,197,253,0.4)"
        strokeWidth="1.5"
      />

      {/* Front wheel */}
      <circle
        cx="141"
        cy="225"
        r="44"
        fill="rgba(6,12,28,0.95)"
        stroke="rgba(59,130,246,0.65)"
        strokeWidth="2.5"
        filter="url(#cf1)"
      />
      <circle
        cx="141"
        cy="225"
        r="31"
        fill="rgba(15,35,90,0.8)"
        stroke="rgba(96,165,250,0.38)"
        strokeWidth="1.5"
      />
      <circle
        cx="141"
        cy="225"
        r="12"
        fill="rgba(37,99,235,0.5)"
        stroke="rgba(147,197,253,0.6)"
        strokeWidth="1.5"
      />

      {/* Rear wheel */}
      <circle
        cx="411"
        cy="225"
        r="44"
        fill="rgba(6,12,28,0.95)"
        stroke="rgba(59,130,246,0.65)"
        strokeWidth="2.5"
        filter="url(#cf1)"
      />
      <circle
        cx="411"
        cy="225"
        r="31"
        fill="rgba(15,35,90,0.8)"
        stroke="rgba(96,165,250,0.38)"
        strokeWidth="1.5"
      />
      <circle
        cx="411"
        cy="225"
        r="12"
        fill="rgba(37,99,235,0.5)"
        stroke="rgba(147,197,253,0.6)"
        strokeWidth="1.5"
      />

      {/* Headlight */}
      <path
        d="M 463 162 L 478 155 L 480 170 L 466 173 Z"
        fill="rgba(255,255,255,0.9)"
        filter="url(#cf1)"
      />
      <path
        d="M 459 173 L 480 170 L 481 177 L 460 180 Z"
        fill="rgba(255,220,120,0.6)"
        filter="url(#cf1)"
      />

      {/* Headlight beam */}
      <path
        d="M 480 161 L 558 147 L 558 181 L 480 172 Z"
        fill="rgba(255,255,200,0.03)"
      />

      {/* Taillight */}
      <rect
        x="55"
        y="185"
        width="5"
        height="23"
        rx="2"
        fill="rgba(239,68,68,0.78)"
        filter="url(#cf1)"
      />
      <rect
        x="55"
        y="185"
        width="5"
        height="23"
        rx="2"
        fill="rgba(239,68,68,0.25)"
        filter="url(#cf2)"
      />

      {/* Ground line */}
      <line
        x1="55"
        y1="225"
        x2="479"
        y2="225"
        stroke="rgba(147,197,253,0.12)"
        strokeWidth="1"
      />
    </svg>
  )
}
