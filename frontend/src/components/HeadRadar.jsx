import { useMemo } from 'react'
import { PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer, Tooltip } from 'recharts'

const SHORT = {
  technical_quality: 'Technical',
  anatomical_completeness: 'Anatomical',
  artifact: 'Artifact-free',
  orientation: 'Orientation',
  diagnostic_suitability: 'Suitability',
}

/** Five-head quality profile as a radar chart (SDS 9.2 screen 3). */
export default function HeadRadar({ heads }) {
  const data = useMemo(
    () => heads.map((h) => ({ head: SHORT[h.key] ?? h.name, score: Math.round(h.score) })),
    [heads],
  )
  return (
    <div className="h-72 w-full" aria-label="Quality head scores radar chart">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={data} outerRadius="72%">
          <PolarGrid stroke="#e2e8f0" />
          <PolarAngleAxis dataKey="head" tick={{ fontSize: 12, fill: '#334155', fontWeight: 500 }} />
          {/* scale drawn between two spokes so it does not overlap a label */}
          <PolarRadiusAxis domain={[0, 100]} tickCount={5} tick={{ fontSize: 10, fill: '#94a3b8' }} angle={54} axisLine={false} />
          <Tooltip formatter={(v) => [`${v} / 100`, 'Score']} />
          <Radar dataKey="score" stroke="#4F46E5" strokeWidth={2} fill="#6366F1" fillOpacity={0.25} isAnimationActive={false} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  )
}
