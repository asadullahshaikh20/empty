import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { errorMessage } from '../api/client'
import { assessmentsApi, imagesApi } from '../api/endpoints'
import Alert from '../components/Alert'
import Icon from '../components/Icon'
import { ACCEPTED_EXTENSIONS, MAX_UPLOAD_MB, MODALITIES } from '../lib/constants'
import { formatBytes, validateFile } from '../lib/validation'

const MODALITY_OPTIONS = [{ value: '', label: 'Auto-detect', hint: 'Let the model decide' }, ...MODALITIES]
const FORMATS = ['DICOM', 'PNG', 'JPEG']
const CHECKS = [
  ['Technical quality', 'Blur, noise, contrast and exposure'],
  ['Anatomical completeness', 'Is the whole region of interest in view?'],
  ['Artifact detection', 'Motion, ring and banding artifacts'],
  ['Orientation', 'Rotation and positioning errors'],
  ['Diagnostic suitability', 'Overall fitness for AI diagnosis'],
]
const VERDICT_LEGEND = [
  ['Pass', 'bg-emerald-500', 'Ready for AI diagnosis'],
  ['Review', 'bg-amber-500', 'Borderline – check manually'],
  ['Reject', 'bg-rose-500', 'Re-acquire the image'],
]

function ProgressSteps({ phase, progress }) {
  const steps = [
    ['Upload and validate image', phase === 'uploading' ? `${progress}%` : null],
    ['Run five-head quality model', null],
  ]
  const active = phase === 'uploading' ? 0 : 1
  return (
    <ol className="space-y-3" aria-label="Assessment progress">
      {steps.map(([label, extra], i) => {
        const done = i < active
        const current = i === active
        return (
          <li key={label} className="flex items-center gap-3 text-sm">
            {done ? (
              <Icon name="check" className="h-5 w-5 text-emerald-500" />
            ) : current ? (
              <span className="h-5 w-5 animate-spin rounded-full border-2 border-indigo-200 border-t-indigo-600" />
            ) : (
              <span className="h-5 w-5 rounded-full border-2 border-slate-200" />
            )}
            <span className={current ? 'font-semibold text-slate-900' : done ? 'text-slate-600' : 'text-slate-400'}>{label}</span>
            {current && extra && <span className="ml-auto tabular-nums text-xs text-slate-500">{extra}</span>}
          </li>
        )
      })}
    </ol>
  )
}

/** Upload screen (SDS 9.2 screen 2, UC-02): drag-and-drop, validation, progress, then assessment. */
export default function UploadPage() {
  const navigate = useNavigate()
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [dims, setDims] = useState(null)
  const [modality, setModality] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const [phase, setPhase] = useState('idle') // idle | uploading | assessing
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')

  // Local preview for PNG/JPEG (browsers cannot render DICOM)
  useEffect(() => {
    setDims(null)
    if (!file || !/\.(png|jpe?g)$/i.test(file.name) || typeof URL.createObjectURL !== 'function') {
      setPreviewUrl(null)
      return undefined
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const busy = phase !== 'idle'
  const openPicker = () => !busy && inputRef.current?.click()

  function choose(selected) {
    setError('')
    const problem = validateFile(selected)
    if (problem) {
      setFile(null)
      setError(problem)
      return
    }
    setFile(selected)
  }

  function onDrop(e) {
    e.preventDefault()
    setDragOver(false)
    if (busy) return
    if (e.dataTransfer.files.length > 1) {
      setError('Please drop a single file.')
      return
    }
    choose(e.dataTransfer.files[0])
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!file) return
    setError('')
    setProgress(0)
    setPhase('uploading')
    let image
    try {
      image = await imagesApi.upload(file, modality, setProgress)
    } catch (err) {
      setError(errorMessage(err, 'Upload failed.'))
      setPhase('idle')
      return
    }
    setPhase('assessing')
    try {
      const result = await assessmentsApi.assess(image.id)
      navigate(`/results/${result.id}`)
    } catch (err) {
      setError(`The image was uploaded, but the quality assessment failed: ${errorMessage(err)}`)
      setPhase('idle')
      setFile(null)
    }
  }

  return (
    <div>
      <h1 className="page-title">New quality assessment</h1>
      <p className="mt-1 max-w-2xl text-sm text-slate-600">
        Upload an X-ray, CT, MRI or retinal fundus image. Five quality dimensions are scored and the image is
        classified as ready, borderline or unsuitable for AI diagnosis.
      </p>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <form onSubmit={handleSubmit} className="card space-y-6 lg:col-span-2">
          <Alert onDismiss={busy ? undefined : () => setError('')}>{error}</Alert>

          <input
            ref={inputRef}
            data-testid="file-input"
            type="file"
            className="hidden"
            accept={ACCEPTED_EXTENSIONS.join(',')}
            onChange={(e) => {
              choose(e.target.files?.[0])
              e.target.value = ''
            }}
          />

          {!file ? (
            <div
              role="button"
              tabIndex={0}
              aria-label="Choose or drop an image file"
              onClick={openPicker}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault() // stop Space from scrolling the page
                  openPicker()
                }
              }}
              onDragOver={(e) => {
                e.preventDefault()
                setDragOver(true)
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
              className={`group flex min-h-[16rem] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-all ${
                dragOver ? 'scale-[1.01] border-indigo-500 bg-indigo-50' : 'border-slate-300 bg-slate-50/60 hover:border-indigo-400 hover:bg-indigo-50/40'
              }`}
            >
              <span className="flex h-16 w-16 items-center justify-center rounded-2xl bg-white text-indigo-600 shadow-sm ring-1 ring-slate-200 transition group-hover:-translate-y-0.5">
                <Icon name="upload" className="h-8 w-8" />
              </span>
              <p className="mt-5 text-sm text-slate-700">
                <span className="font-semibold text-indigo-700">Click to browse</span> or drag and drop a scan
              </p>
              <div className="mt-3 flex gap-2">
                {FORMATS.map((f) => (
                  <span key={f} className="rounded-md bg-white px-2 py-0.5 text-xs font-semibold tracking-wider text-slate-600 ring-1 ring-slate-200">
                    {f}
                  </span>
                ))}
              </div>
              <p className="mt-2 text-xs text-slate-500">Up to {MAX_UPLOAD_MB} MB · at least 128×128 px</p>
            </div>
          ) : (
            <div className="flex flex-col gap-4 rounded-2xl border border-slate-200 bg-slate-50/60 p-4 sm:flex-row sm:items-center">
              <div className="relative h-36 w-36 shrink-0 overflow-hidden rounded-xl bg-black">
                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="Selected image preview"
                    className="h-full w-full object-contain"
                    onLoad={(e) => setDims([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])}
                  />
                ) : (
                  <span className="flex h-full w-full items-center justify-center text-slate-400">
                    <Icon name="file" className="h-12 w-12" />
                  </span>
                )}
                {busy && <span className="absolute inset-x-0 h-0.5 animate-scan bg-indigo-400 shadow-[0_0_12px_2px_rgba(99,102,241,0.8)]" />}
              </div>
              <div className="min-w-0 flex-1">
                <p className="break-all text-sm font-semibold text-slate-900">{file.name}</p>
                <div className="mt-2 flex flex-wrap gap-2 text-xs">
                  <span className="rounded-md bg-white px-2 py-0.5 font-medium text-slate-600 ring-1 ring-slate-200">{formatBytes(file.size)}</span>
                  {dims && (
                    <span className="rounded-md bg-white px-2 py-0.5 font-medium text-slate-600 ring-1 ring-slate-200">
                      {dims[0]}×{dims[1]} px
                    </span>
                  )}
                  {!previewUrl && (
                    <span className="rounded-md bg-white px-2 py-0.5 font-medium text-slate-600 ring-1 ring-slate-200">DICOM – preview after upload</span>
                  )}
                </div>
                {busy ? (
                  <div className="mt-4">
                    <ProgressSteps phase={phase} progress={progress} />
                    {phase === 'uploading' && (
                      <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-slate-200" role="progressbar"
                        aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100} aria-label="Upload progress">
                        <div className="h-full rounded-full bg-indigo-600 transition-all" style={{ width: `${progress}%` }} />
                      </div>
                    )}
                  </div>
                ) : (
                  <button type="button" onClick={openPicker} className="btn-secondary mt-4 px-3 py-1.5 text-xs">
                    Change image
                  </button>
                )}
              </div>
            </div>
          )}

          <fieldset>
            <legend className="label">Imaging modality</legend>
            <div role="radiogroup" aria-label="Imaging modality" className="grid grid-cols-2 gap-2 sm:grid-cols-5">
              {MODALITY_OPTIONS.map((m) => {
                const selected = modality === m.value
                return (
                  <button
                    key={m.value || 'auto'}
                    type="button"
                    role="radio"
                    aria-checked={selected}
                    disabled={busy}
                    onClick={() => setModality(m.value)}
                    className={`rounded-xl border px-3 py-2.5 text-left text-sm transition disabled:opacity-50 ${
                      selected ? 'border-indigo-500 bg-indigo-50 text-indigo-800 ring-1 ring-indigo-500' : 'border-slate-200 bg-white text-slate-700 hover:border-slate-300'
                    }`}
                  >
                    <span className="block font-semibold">{m.label}</span>
                    {m.hint && <span className="block text-xs text-slate-500">{m.hint}</span>}
                  </button>
                )
              })}
            </div>
          </fieldset>

          <button type="submit" className="btn-primary w-full py-3" disabled={!file || busy}>
            {busy && <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />}
            {phase === 'idle' ? 'Upload and assess' : phase === 'uploading' ? 'Uploading…' : 'Assessing…'}
          </button>
        </form>

        <aside className="space-y-6">
          <div className="card">
            <h2 className="card-title">What gets checked</h2>
            <ul className="mt-4 space-y-3.5">
              {CHECKS.map(([name, text]) => (
                <li key={name} className="flex gap-3">
                  <Icon name="check" className="mt-0.5 h-5 w-5 shrink-0 text-indigo-500" />
                  <div>
                    <p className="text-sm font-medium text-slate-800">{name}</p>
                    <p className="text-xs text-slate-500">{text}</p>
                  </div>
                </li>
              ))}
            </ul>
          </div>
          <div className="card">
            <h2 className="card-title">Verdicts</h2>
            <ul className="mt-4 space-y-3">
              {VERDICT_LEGEND.map(([name, dot, text]) => (
                <li key={name} className="flex items-center gap-3 text-sm">
                  <span className={`h-2.5 w-2.5 rounded-full ${dot}`} />
                  <span className="w-14 font-semibold text-slate-800">{name}</span>
                  <span className="text-slate-500">{text}</span>
                </li>
              ))}
            </ul>
          </div>
        </aside>
      </div>
    </div>
  )
}
