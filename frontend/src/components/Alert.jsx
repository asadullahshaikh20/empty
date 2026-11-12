import Icon from './Icon'

const STYLES = {
  error: ['border-rose-200 bg-rose-50 text-rose-800', 'reject', 'text-rose-500'],
  warning: ['border-amber-200 bg-amber-50 text-amber-900', 'warning', 'text-amber-500'],
  info: ['border-indigo-200 bg-indigo-50 text-indigo-900', 'info', 'text-indigo-500'],
}

export default function Alert({ kind = 'error', children, onDismiss }) {
  if (!children) return null
  const [box, icon, iconColor] = STYLES[kind]
  return (
    <div
      role={kind === 'error' ? 'alert' : 'status'}
      className={`flex animate-fade-in items-start gap-3 rounded-xl border px-4 py-3 text-sm ${box}`}
    >
      <Icon name={icon} className={`mt-0.5 h-5 w-5 shrink-0 ${iconColor}`} />
      <div className="flex-1">{children}</div>
      {onDismiss && (
        <button type="button" onClick={onDismiss} aria-label="Dismiss" className="rounded p-0.5 opacity-60 hover:opacity-100">
          <Icon name="close" className="h-4 w-4" />
        </button>
      )}
    </div>
  )
}
