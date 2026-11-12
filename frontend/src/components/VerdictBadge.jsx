const STYLES = {
  Pass: ['bg-emerald-50 text-emerald-700 ring-emerald-600/20', 'bg-emerald-500'],
  Review: ['bg-amber-50 text-amber-800 ring-amber-600/20', 'bg-amber-500'],
  Reject: ['bg-rose-50 text-rose-700 ring-rose-600/20', 'bg-rose-500'],
}

const DESCRIPTIONS = {
  Pass: 'Suitable for AI diagnostic use',
  Review: 'Borderline quality – review recommended',
  Reject: 'Not suitable for AI diagnostic use',
}

export default function VerdictBadge({ verdict, size = 'md' }) {
  if (!verdict) return <span className="text-sm text-slate-400">—</span>
  const [box, dot] = STYLES[verdict] ?? ['bg-slate-50 text-slate-700 ring-slate-600/20', 'bg-slate-400']
  const sizing = size === 'lg' ? 'px-3.5 py-1 text-sm' : 'px-2.5 py-0.5 text-xs'
  return (
    <span
      title={DESCRIPTIONS[verdict]}
      data-verdict={verdict}
      className={`inline-flex items-center gap-1.5 rounded-full font-semibold uppercase tracking-wider ring-1 ring-inset ${sizing} ${box}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} aria-hidden="true" />
      {verdict}
    </span>
  )
}
