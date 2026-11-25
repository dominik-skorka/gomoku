/** Small stroke icons (24×24 grid, currentColor), sized by the caller. */
const base = {
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  focusable: false,
};

/** White flag on a pole: the universal "I give up". */
export function WhiteFlagIcon({ className = "size-4" }: { className?: string }) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M5 21V4" />
      <path d="M5 4h11.5l-2.5 4 2.5 4H5" fill="#fff" stroke="currentColor" />
    </svg>
  );
}

export function FirstIcon({ className = "size-5" }: { className?: string }) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M6 5v14M18 6l-8 6 8 6z" />
    </svg>
  );
}

export function PrevIcon({ className = "size-5" }: { className?: string }) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M15 6l-6 6 6 6" />
    </svg>
  );
}

export function NextIcon({ className = "size-5" }: { className?: string }) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M9 6l6 6-6 6" />
    </svg>
  );
}

export function LastIcon({ className = "size-5" }: { className?: string }) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M18 5v14M6 6l8 6-8 6z" />
    </svg>
  );
}
