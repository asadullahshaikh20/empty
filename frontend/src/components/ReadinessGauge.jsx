import { useEffect, useState } from 'react'

const COLORS = { Pass: '#10B981', Review: '#F59E0B', Reject: '#EF4444' }

/** Circular AI-Readiness Score gauge, 0–100 (SDS 9.2 screen 3, FR-RPT-04). */
export default function ReadinessGauge({ score, verdict, size = 180 }) {
  const value = Math.max(0, Math.min(100, Number.isFinite(score) ? score : 0))
  const stroke = 14
  const radius = (size - stroke) / 2
  const circumference = 2 * Math.PI * radius
  // start empty and animate to the value after the first paint
  const [shown, setShown] = useState(0)
  useEffect(() => setShown(value), [value])
  const offset = circumference * (1 - shown / 100)
  const color = COLORS[verdict] ?? '#4F46E5'
  const rounded = Math.round(value)
  return (
    <figure className="flex flex-col items-center">
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        role="meter"
        aria-valuenow={rounded}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuetext={`${rounded} out of 100${verdict ? `, verdict ${verdict}` : ''}`}
        aria-label="AI-Readiness Score"
      >
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="#e2e8f0" strokeWidth={stroke} />
        <circle
          data-testid="gauge-arc"
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: 'stroke-dashoffset 800ms ease-out' }}
        />
        <text x="50%" y="48%" textAnchor="middle" dominantBaseline="middle" fontSize={size / 4.2} fontWeight="700" fill="#0f172a">
          {rounded}
        </text>
        <text x="50%" y="66%" textAnchor="middle" fontSize={size / 14} fill="#64748b">
          / 100
        </text>
      </svg>
      <figcaption className="mt-1 text-xs font-semibold uppercase tracking-wider text-slate-500">AI-Readiness Score</figcaption>
    </figure>
  )
}
