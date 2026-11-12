import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { errorMessage } from '../api/client'
import { assessmentsApi } from '../api/endpoints'
import Alert from '../components/Alert'
import AuthImage from '../components/AuthImage'
import Icon from '../components/Icon'
import Spinner from '../components/Spinner'
import VerdictBadge from '../components/VerdictBadge'
import { MODALITIES, MODALITY_LABEL, VERDICTS } from '../lib/constants'
import { formatDateTime } from '../lib/validation'

const PAGE_SIZE = 10
const emptyFilters = () => ({ verdict: '', modality: '', date_from: '', date_to: '', q: '' })

function pct(n, total) {
  return total ? `${Math.round((n / total) * 100)}%` : '–'
}

function scoreBar(score) {
  if (score >= 75) return 'bg-emerald-500'
  if (score >= 50) return 'bg-amber-500'
  return 'bg-rose-500'
}

function Chip({ active, onClick, children }) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onClick}
      className={`rounded-full px-3 py-1 text-xs font-semibold transition ${
        active ? 'bg-indigo-600 text-white shadow-sm' : 'bg-white text-slate-600 ring-1 ring-slate-200 hover:ring-slate-300'
      }`}
    >
      {children}
    </button>
  )
}

function pageNumbers(page, pages) {
  const start = Math.max(1, Math.min(page - 2, pages - 4))
  return Array.from({ length: Math.min(5, pages) }, (_, i) => start + i)
}

/** Assessment History screen (SDS 9.2 screen 6, UC-07, FR-RPT-05/06). */
export default function HistoryPage() {
  const navigate = useNavigate()
  const [filters, setFilters] = useState(emptyFilters)
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const [data, setData] = useState(null)
  const [stats, setStats] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    assessmentsApi.stats().then((s) => !cancelled && setStats(s)).catch(() => {})
    return () => {
      cancelled = true
    }
  }, [])

  // debounce the file-name search
  useEffect(() => {
    const t = setTimeout(() => {
      setPage(1)
      setFilters((f) => (f.q === search.trim() ? f : { ...f, q: search.trim() }))
    }, 300)
    return () => clearTimeout(t)
  }, [search])

  useEffect(() => {
    let cancelled = false
    setError('')
    const params = { page, page_size: PAGE_SIZE }
    for (const [k, v] of Object.entries(filters)) if (v) params[k] = v
    assessmentsApi
      .history(params)
      .then((d) => !cancelled && setData(d))
      .catch((err) => !cancelled && setError(errorMessage(err, 'Could not load history.')))
    return () => {
      cancelled = true
    }
  }, [filters, page])

  const update = (field, value) => {
    setPage(1)
    setFilters((f) => ({ ...f, [field]: value }))
  }
  const activeCount = Object.values(filters).filter(Boolean).length
  const clearAll = () => {
    setSearch('')
    setPage(1)
    setFilters(emptyFilters())
  }
  const open = (id) => navigate(`/results/${id}`)

  const kpis = [
    ['Total scans', stats?.total ?? '–', 'text-slate-900'],
    ['Pass rate', stats ? pct(stats.Pass, stats.total) : '–', 'text-emerald-600'],
    ['Need review', stats?.Review ?? '–', 'text-amber-600'],
    ['Rejection rate', stats ? pct(stats.Reject, stats.total) : '–', 'text-rose-600'],
  ]

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <h1 className="page-title">Assessment history</h1>
          <p className="mt-1 text-sm text-slate-600">Every image you have assessed, newest first.</p>
        </div>
        <Link to="/upload" className="btn-primary ml-auto">
          <Icon name="plus" className="h-4 w-4" />
          New assessment
        </Link>
      </div>

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {kpis.map(([label, value, color]) => (
          <div key={label} className="card p-5">
            <p className="card-title">{label}</p>
            <p className={`mt-2 text-3xl font-bold tabular-nums ${color}`}>{value}</p>
          </div>
        ))}
      </div>

      <div className="card space-y-4 p-5">
        <div className="flex flex-wrap items-center gap-3">
          <div className="relative min-w-[14rem] flex-1">
            <Icon name="search" className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <input
              type="search"
              aria-label="Search by file name"
              placeholder="Search by file name…"
              className="input pl-9"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <label className="flex items-center gap-2 text-xs text-slate-500">
            From
            <input type="date" aria-label="From date" className="input w-auto py-2" value={filters.date_from}
              max={filters.date_to || undefined} onChange={(e) => update('date_from', e.target.value)} />
          </label>
          <label className="flex items-center gap-2 text-xs text-slate-500">
            To
            <input type="date" aria-label="To date" className="input w-auto py-2" value={filters.date_to}
              min={filters.date_from || undefined} onChange={(e) => update('date_to', e.target.value)} />
          </label>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="mr-1 text-xs font-semibold uppercase tracking-wider text-slate-400">Verdict</span>
          <Chip active={!filters.verdict} onClick={() => update('verdict', '')}>All</Chip>
          {VERDICTS.map((v) => (
            <Chip key={v} active={filters.verdict === v} onClick={() => update('verdict', v)}>{v}</Chip>
          ))}
          <span className="ml-4 mr-1 text-xs font-semibold uppercase tracking-wider text-slate-400">Modality</span>
          <Chip active={!filters.modality} onClick={() => update('modality', '')}>All</Chip>
          {MODALITIES.map((m) => (
            <Chip key={m.value} active={filters.modality === m.value} onClick={() => update('modality', m.value)}>{m.label}</Chip>
          ))}
          {activeCount > 0 && (
            <button type="button" onClick={clearAll} className="ml-auto text-xs font-semibold text-indigo-700 hover:underline">
              Clear all ({activeCount})
            </button>
          )}
        </div>
      </div>

      <Alert>{error}</Alert>
      {!data && !error && <Spinner label="Loading history…" />}
      {data && data.items.length === 0 && (
        <div className="card flex flex-col items-center py-12 text-center">
          <span className="flex h-14 w-14 items-center justify-center rounded-full bg-slate-100 text-slate-400">
            <Icon name="history" className="h-7 w-7" />
          </span>
          <p className="mt-4 font-semibold text-slate-800">{activeCount ? 'No assessments match these filters' : 'No assessments yet'}</p>
          <p className="mt-1 text-sm text-slate-500">{activeCount ? 'Try clearing the filters.' : 'Upload an image to see its quality report here.'}</p>
          {!activeCount && <Link to="/upload" className="btn-primary mt-5">Upload an image</Link>}
        </div>
      )}
      {data && data.items.length > 0 && (
        <div className="card overflow-hidden p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs uppercase tracking-wider text-slate-500">
                  <th className="px-5 py-3 font-semibold">Scan</th>
                  <th className="px-5 py-3 font-semibold">Modality</th>
                  <th className="px-5 py-3 font-semibold">AI-Readiness</th>
                  <th className="px-5 py-3 font-semibold">Verdict</th>
                  <th className="px-5 py-3 font-semibold">Assessed</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((a) => (
                  <tr
                    key={a.id}
                    tabIndex={0}
                    role="link"
                    aria-label={`Open assessment of ${a.original_filename}`}
                    onClick={() => open(a.id)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        open(a.id)
                      }
                    }}
                    className="cursor-pointer border-b border-slate-100 transition last:border-0 hover:bg-indigo-50/40 focus:bg-indigo-50/60 focus:outline-none"
                  >
                    <td className="px-5 py-3">
                      <div className="flex items-center gap-3">
                        <AuthImage imageId={a.image_id} kind="thumbnail" alt="" className="h-11 w-11 rounded-lg bg-black object-contain" />
                        <div className="min-w-0">
                          <p className="max-w-[16rem] truncate font-semibold text-slate-800">{a.original_filename}</p>
                          <p className="font-mono text-xs text-slate-400">{a.id.slice(0, 8)}</p>
                        </div>
                      </div>
                    </td>
                    <td className="px-5 py-3">
                      {a.modality ? (
                        <span className="rounded-md bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700">{MODALITY_LABEL[a.modality]}</span>
                      ) : '—'}
                    </td>
                    <td className="px-5 py-3">
                      {a.ai_readiness_score == null ? '—' : (
                        <div className="flex items-center gap-2.5">
                          <span className="w-7 font-semibold tabular-nums text-slate-800">{Math.round(a.ai_readiness_score)}</span>
                          <div className="h-1.5 w-20 overflow-hidden rounded-full bg-slate-200">
                            <div className={`h-full rounded-full ${scoreBar(a.ai_readiness_score)}`} style={{ width: `${a.ai_readiness_score}%` }} />
                          </div>
                        </div>
                      )}
                    </td>
                    <td className="px-5 py-3">
                      {a.status === 'completed' ? <VerdictBadge verdict={a.verdict} /> : <span className="text-xs font-medium text-rose-700">{a.status}</span>}
                    </td>
                    <td className="whitespace-nowrap px-5 py-3 text-slate-500">{formatDateTime(a.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex flex-wrap items-center gap-3 border-t border-slate-100 px-5 py-3 text-sm">
            <span className="text-slate-500">
              Showing {(data.page - 1) * data.page_size + 1}–{(data.page - 1) * data.page_size + data.items.length} of {data.total} results
            </span>
            {data.pages > 1 && (
              <nav className="ml-auto flex items-center gap-1" aria-label="Pagination">
                <button type="button" className="btn-secondary px-3 py-1.5" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
                  Previous
                </button>
                {pageNumbers(page, data.pages).map((n) => (
                  <button
                    key={n}
                    type="button"
                    aria-current={n === page ? 'page' : undefined}
                    onClick={() => setPage(n)}
                    className={`h-8 w-8 rounded-lg text-sm font-medium ${n === page ? 'bg-indigo-600 text-white' : 'text-slate-600 hover:bg-slate-100'}`}
                  >
                    {n}
                  </button>
                ))}
                <button type="button" className="btn-secondary px-3 py-1.5" disabled={page >= data.pages} onClick={() => setPage((p) => p + 1)}>
                  Next
                </button>
              </nav>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
