import { useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { errorMessage } from '../api/client'
import { assessmentsApi } from '../api/endpoints'
import Alert from '../components/Alert'
import AuthImage from '../components/AuthImage'
import HeadRadar from '../components/HeadRadar'
import Icon from '../components/Icon'
import ReadinessGauge from '../components/ReadinessGauge'
import Spinner from '../components/Spinner'
import { MODALITY_LABEL, MODALITY_SOURCE_LABEL } from '../lib/constants'
import { formatBytes, formatDateTime } from '../lib/validation'

const VERDICT_UI = {
  Pass: {
    icon: 'check',
    panel: 'border-emerald-200 from-emerald-50',
    pill: 'bg-emerald-600',
    tone: 'text-emerald-700',
    label: 'Passed diagnostic quality',
    text: 'The image meets the quality requirements and can be forwarded to a diagnostic AI model.',
  },
  Review: {
    icon: 'warning',
    panel: 'border-amber-200 from-amber-50',
    pill: 'bg-amber-500',
    tone: 'text-amber-700',
    label: 'Review recommended',
    text: 'Image quality is borderline. A manual review is recommended before AI diagnostic use.',
  },
  Reject: {
    icon: 'reject',
    panel: 'border-rose-200 from-rose-50',
    pill: 'bg-rose-600',
    tone: 'text-rose-700',
    label: 'Rejected',
    text: 'The image is not reliable for AI diagnosis. Re-acquisition is recommended.',
  },
}

const HEAD_INFO = {
  technical_quality: 'Blur, noise, contrast and exposure. Poor technical quality hides fine detail such as small lesions.',
  anatomical_completeness: 'Whether the full region of interest is in the field of view. Cut-off anatomy can hide findings.',
  artifact: 'Motion, ring and banding artifacts. Higher means fewer artifacts; artifacts can mimic or mask pathology.',
  orientation: 'Rotation and positioning. Misoriented images can mislead models trained on standard views.',
  diagnostic_suitability: 'The model’s overall judgement of whether the image is fit for AI diagnosis.',
}

function tone(score) {
  if (score >= 67) return { bar: 'bg-emerald-500', text: 'text-emerald-700' }
  if (score >= 34) return { bar: 'bg-amber-500', text: 'text-amber-700' }
  return { bar: 'bg-rose-500', text: 'text-rose-700' }
}

function ImageViewer({ imageId, alt }) {
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const drag = useRef(null)
  const setZ = (z) => {
    const next = Math.min(4, Math.max(1, Math.round(z * 4) / 4))
    setZoom(next)
    if (next === 1) setPan({ x: 0, y: 0 })
  }
  return (
    <div>
      <div
        className="relative aspect-square overflow-hidden rounded-xl bg-black"
        style={{ cursor: zoom > 1 ? 'grab' : 'zoom-in', touchAction: 'none' }}
        onDoubleClick={() => setZ(zoom > 1 ? 1 : 2)}
        onPointerDown={(e) => {
          if (zoom === 1) return
          drag.current = { x: e.clientX - pan.x, y: e.clientY - pan.y }
          e.currentTarget.setPointerCapture(e.pointerId)
        }}
        onPointerMove={(e) => drag.current && setPan({ x: e.clientX - drag.current.x, y: e.clientY - drag.current.y })}
        onPointerUp={() => {
          drag.current = null
        }}
      >
        <div
          className="h-full w-full"
          style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, transition: drag.current ? 'none' : 'transform 150ms' }}
        >
          <AuthImage imageId={imageId} kind="preview" alt={alt} className="h-full w-full select-none object-contain" />
        </div>
      </div>
      <div className="mt-2 flex items-center justify-center gap-1 text-slate-600">
        <button type="button" className="rounded-lg p-1.5 hover:bg-slate-100 disabled:opacity-40" onClick={() => setZ(zoom - 0.5)} disabled={zoom <= 1} aria-label="Zoom out">
          <Icon name="minus" className="h-4 w-4" />
        </button>
        <span className="w-12 text-center text-xs tabular-nums">{Math.round(zoom * 100)}%</span>
        <button type="button" className="rounded-lg p-1.5 hover:bg-slate-100 disabled:opacity-40" onClick={() => setZ(zoom + 0.5)} disabled={zoom >= 4} aria-label="Zoom in">
          <Icon name="plus" className="h-4 w-4" />
        </button>
        <button type="button" className="rounded-lg p-1.5 hover:bg-slate-100 disabled:opacity-40" onClick={() => setZ(1)} disabled={zoom === 1} aria-label="Reset zoom">
          <Icon name="reset" className="h-4 w-4" />
        </button>
      </div>
    </div>
  )
}

function Warnings({ data }) {
  const warnings = data.warnings ?? []
  if (!warnings.length) return null
  const severe = data.out_of_distribution
  return (
    <details className="group rounded-2xl border border-amber-200 bg-amber-50 px-5 py-4 text-sm text-amber-900" open={severe}>
      <summary className="flex cursor-pointer list-none items-center gap-3">
        <Icon name="warning" className="h-5 w-5 shrink-0 text-amber-600" />
        <span className="font-semibold">{severe ? 'Possible out-of-distribution image' : 'Low-confidence result'}</span>
        <span className={`rounded-full px-2 py-0.5 text-xs font-semibold uppercase tracking-wider ${severe ? 'bg-rose-100 text-rose-700' : 'bg-amber-100 text-amber-800'}`}>
          {severe ? 'High' : 'Medium'}
        </span>
        <Icon name="chevronDown" className="ml-auto h-4 w-4 transition group-open:rotate-180" />
      </summary>
      <div className="mt-3 grid gap-4 pl-8 md:grid-cols-2">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-amber-700">Details</p>
          <ul className="mt-1 list-inside list-disc space-y-1 text-xs">
            {warnings.map((w, i) => <li key={`${w}-${i}`}>{w}</li>)}
          </ul>
        </div>
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-amber-700">Recommended steps</p>
          <ol className="mt-1 list-inside list-decimal space-y-1 text-xs">
            <li>Confirm the image is an X-ray, CT, MRI or retinal fundus image.</li>
            <li>Have a radiologist or technician check the image quality manually.</li>
            <li>If quality is doubtful, re-acquire the image before AI diagnosis.</li>
          </ol>
        </div>
      </div>
    </details>
  )
}

/** Assessment Results screen (SDS 9.2 screen 3, UC-03 / UC-06). */
export default function ResultsPage() {
  const { id } = useParams()
  const [data, setData] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    setData(null)
    setError('')
    assessmentsApi
      .get(id)
      .then((d) => !cancelled && setData(d))
      .catch((err) => !cancelled && setError(errorMessage(err, 'Could not load the assessment.')))
    return () => {
      cancelled = true
    }
  }, [id])

  if (error) return <Alert>{error}</Alert>
  if (!data) return <Spinner label="Loading assessment…" />

  const { image } = data
  const ui = VERDICT_UI[data.verdict]
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end gap-3">
        <div className="min-w-0">
          <p className="font-mono text-xs text-slate-400">Assessment {data.id.slice(0, 8)}</p>
          <h1 className="page-title">{image.original_filename}</h1>
          <p className="mt-1 text-sm text-slate-500">Assessed {formatDateTime(data.completed_at || data.created_at)}</p>
        </div>
        <div className="ml-auto flex flex-wrap gap-2">
          <Link to="/history" className="btn-secondary">
            <Icon name="history" className="h-4 w-4" />
            View in history
          </Link>
          <Link to="/upload" className="btn-primary">
            <Icon name="plus" className="h-4 w-4" />
            Assess another scan
          </Link>
        </div>
      </div>

      {data.status !== 'completed' ? (
        <Alert>
          Assessment {data.status === 'error' ? 'failed' : 'is still processing'}
          {data.error_message ? `: ${data.error_message}` : '.'}
        </Alert>
      ) : (
        <>
          <div className="grid gap-6 lg:grid-cols-3">
            <section className={`flex flex-col items-center gap-6 rounded-2xl border bg-gradient-to-br to-white p-6 shadow-sm sm:flex-row sm:gap-10 lg:col-span-2 ${ui.panel}`}>
              <ReadinessGauge score={data.ai_readiness_score} verdict={data.verdict} size={180} />
              <div className="text-center sm:text-left">
                <span className={`inline-flex items-center gap-2 rounded-full px-3.5 py-1.5 text-xs font-bold uppercase tracking-wider text-white ${ui.pill}`}>
                  <Icon name={ui.icon} className="h-4 w-4" />
                  {ui.label}
                </span>
                <p className="mt-4 max-w-md text-base text-slate-700">{ui.text}</p>
                {data.artifact_present && (
                  <p className="mt-3 inline-flex items-center gap-1.5 rounded-full bg-white px-3 py-1 text-xs font-medium text-amber-800 ring-1 ring-amber-200">
                    <Icon name="warning" className="h-3.5 w-3.5" />
                    Acquisition artifact detected
                  </p>
                )}
              </div>
            </section>

            <section className="card p-4">
              <ImageViewer imageId={image.id} alt={`Uploaded image ${image.original_filename}`} />
              <dl className="mt-3 grid grid-cols-[auto,1fr] gap-x-4 gap-y-1.5 border-t border-slate-100 px-1 pt-3 text-xs">
                <dt className="text-slate-500">Modality</dt>
                <dd className="font-medium text-slate-800">
                  {MODALITY_LABEL[data.modality] ?? 'Unknown'}{' '}
                  <span className="font-normal text-slate-400">({MODALITY_SOURCE_LABEL[data.modality_source] ?? 'n/a'})</span>
                </dd>
                <dt className="text-slate-500">Format</dt>
                <dd className="text-slate-800">{image.file_format} · {image.width}×{image.height} · {formatBytes(image.size_bytes)}</dd>
                <dt className="text-slate-500">Model</dt>
                <dd className="font-mono text-slate-800">{data.model_version} · {data.inference_ms} ms</dd>
              </dl>
            </section>
          </div>

          <Warnings data={data} />

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
            {data.heads.map((h) => {
              const t = tone(h.score)
              return (
                <section key={h.key} className="card p-5">
                  <div className="flex items-start justify-between gap-2">
                    <h2 className="text-sm font-semibold text-slate-800">{h.name}</h2>
                    <span title={HEAD_INFO[h.key]} aria-label={HEAD_INFO[h.key]} className="cursor-help text-slate-400 hover:text-indigo-600">
                      <Icon name="info" className="h-4 w-4" />
                    </span>
                  </div>
                  <p className={`mt-3 text-3xl font-bold tabular-nums ${t.text}`}>{Math.round(h.score)}</p>
                  <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
                    <div className={`h-full rounded-full ${t.bar}`} style={{ width: `${Math.max(0, Math.min(100, h.score))}%` }} />
                  </div>
                  <p className="mt-2 text-xs font-medium text-slate-700">{h.label}</p>
                  <p className="text-xs text-slate-400">
                    Confidence {(h.confidence * 100).toFixed(0)}% · {h.confidence_level}
                  </p>
                </section>
              )
            })}
          </div>

          <section className="card">
            <h2 className="card-title">Quality profile</h2>
            <HeadRadar heads={data.heads} />
            <p className="text-center text-xs text-slate-400">Higher is better on every axis (0–100).</p>
          </section>
        </>
      )}
    </div>
  )
}
