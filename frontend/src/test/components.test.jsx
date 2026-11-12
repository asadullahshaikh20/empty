import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ReadinessGauge from '../components/ReadinessGauge'
import VerdictBadge from '../components/VerdictBadge'

describe('VerdictBadge', () => {
  it.each([
    ['Pass', 'bg-emerald-50'],
    ['Review', 'bg-amber-50'],
    ['Reject', 'bg-rose-50'],
  ])('renders %s with its colour', (verdict, cls) => {
    render(<VerdictBadge verdict={verdict} />)
    const badge = screen.getByText(verdict)
    expect(badge).toHaveClass(cls)
    expect(badge).toHaveAttribute('data-verdict', verdict)
  })

  it('renders a dash without a verdict', () => {
    render(<VerdictBadge verdict={null} />)
    expect(screen.getByText('—')).toBeInTheDocument()
  })
})

describe('ReadinessGauge', () => {
  it('shows the rounded score with an accessible label', () => {
    render(<ReadinessGauge score={81.6} verdict="Pass" />)
    const meter = screen.getByRole('meter', { name: 'AI-Readiness Score' })
    expect(meter).toHaveAttribute('aria-valuenow', '82')
    expect(meter).toHaveAttribute('aria-valuetext', '82 out of 100, verdict Pass')
    expect(screen.getByText('82')).toBeInTheDocument()
    expect(screen.getByTestId('gauge-arc')).toHaveAttribute('stroke', '#10B981')
  })

  it('clamps out-of-range and missing values', () => {
    const { rerender } = render(<ReadinessGauge score={140} verdict="Review" />)
    expect(screen.getByText('100')).toBeInTheDocument()
    expect(screen.getByTestId('gauge-arc')).toHaveAttribute('stroke-dashoffset', '0')
    rerender(<ReadinessGauge score={-5} verdict="Reject" />)
    expect(screen.getByText('0')).toBeInTheDocument()
    rerender(<ReadinessGauge score={undefined} />)
    expect(screen.getByText('0')).toBeInTheDocument()
  })
})
