import Icon from '../components/Icon'
import { APP_TITLE, DISCLAIMER, PROJECT_CODE } from '../lib/constants'

const POINTS = [
  ['layers', 'Five quality checks', 'Technical quality, anatomical completeness, artifacts, orientation and diagnostic suitability.'],
  ['check', 'AI-Readiness Score', 'One 0–100 score with a clear Pass, Review or Reject verdict for every image.'],
  ['file', 'X-ray, CT, MRI and retinal', 'Upload DICOM, PNG or JPEG images; the modality is detected automatically.'],
]

export default function AuthShell({ title, subtitle, children }) {
  return (
    <div className="grid min-h-screen lg:grid-cols-2">
      <div className="relative hidden flex-col justify-between overflow-hidden bg-gradient-to-br from-indigo-800 via-indigo-900 to-slate-900 p-12 text-white lg:flex">
        <div className="pointer-events-none absolute -right-24 -top-24 h-96 w-96 rounded-full bg-indigo-500/20 blur-3xl" />
        <div className="flex items-center gap-3">
          <img src="/favicon.svg" alt="" className="h-10 w-10 rounded-lg ring-1 ring-white/30" />
          <span className="text-sm font-medium text-indigo-100">{PROJECT_CODE} · Sukkur IBA University</span>
        </div>
        <div className="relative">
          <h1 className="max-w-md text-4xl font-bold leading-tight tracking-tight">{APP_TITLE}</h1>
          <p className="mt-4 max-w-md text-indigo-100">
            A quality gate that checks medical images before they reach a diagnostic AI model.
          </p>
          <ul className="mt-10 space-y-6">
            {POINTS.map(([icon, heading, text]) => (
              <li key={heading} className="flex gap-4">
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-white/10 ring-1 ring-white/20">
                  <Icon name={icon} />
                </span>
                <div>
                  <p className="font-semibold">{heading}</p>
                  <p className="max-w-sm text-sm text-indigo-100/90">{text}</p>
                </div>
              </li>
            ))}
          </ul>
        </div>
        <p className="text-xs text-indigo-200/80">{DISCLAIMER}</p>
      </div>

      <div className="flex flex-col justify-center bg-white px-6 py-10 sm:px-12">
        <div className="mx-auto w-full max-w-sm">
          <div className="mb-8 flex items-center gap-3 lg:hidden">
            <img src="/favicon.svg" alt="" className="h-10 w-10" />
            <p className="text-sm font-semibold leading-tight text-slate-900">{APP_TITLE}</p>
          </div>
          <h2 className="text-2xl font-bold tracking-tight text-slate-900">{title}</h2>
          {subtitle && <p className="mt-2 text-sm text-slate-500">{subtitle}</p>}
          <div className="mt-8">{children}</div>
          <p className="mt-10 text-xs text-slate-400 lg:hidden">{DISCLAIMER}</p>
        </div>
      </div>
    </div>
  )
}
