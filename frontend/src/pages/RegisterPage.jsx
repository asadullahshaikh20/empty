import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { errorMessage } from '../api/client'
import { useAuth } from '../auth/context'
import Alert from '../components/Alert'
import { passwordProblems } from '../lib/validation'
import AuthShell from './AuthShell'

export default function RegisterPage() {
  const { register } = useAuth()
  const navigate = useNavigate()
  const [form, setForm] = useState({ name: '', email: '', username: '', password: '', confirm: '', role: 'radiologist' })
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const set = (field) => (e) => setForm((f) => ({ ...f, [field]: e.target.value }))
  const problems = form.password ? passwordProblems(form.password) : []
  const mismatch = form.confirm && form.password !== form.confirm

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    if (problems.length) return setError(`Password needs ${problems.join(', ')}.`)
    if (form.password !== form.confirm) return setError('Passwords do not match.')
    setBusy(true)
    try {
      await register({
        name: form.name.trim(),
        email: form.email.trim(),
        username: form.username.trim(),
        password: form.password,
        role: form.role,
      })
      navigate('/upload', { replace: true })
    } catch (err) {
      setError(errorMessage(err, 'Registration failed.'))
    } finally {
      setBusy(false)
    }
  }

  const complete = form.name && form.email && form.username && form.password && form.confirm
  return (
    <AuthShell title="Create an account" subtitle="Administrator accounts are created by the system administrator.">
      <form onSubmit={handleSubmit} className="space-y-4" noValidate>
        <Alert>{error}</Alert>
        <div>
          <label htmlFor="name" className="label">Full name</label>
          <input id="name" className="input" autoComplete="name" value={form.name} onChange={set('name')} />
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label htmlFor="email" className="label">Email</label>
            <input id="email" type="email" className="input" autoComplete="email" value={form.email} onChange={set('email')} />
          </div>
          <div>
            <label htmlFor="username" className="label">Username</label>
            <input id="username" className="input" autoComplete="username" value={form.username} onChange={set('username')} />
          </div>
        </div>
        <div>
          <label htmlFor="role" className="label">Role</label>
          <select id="role" className="input" value={form.role} onChange={set('role')}>
            <option value="radiologist">Radiologist / Technician</option>
            <option value="researcher">Researcher</option>
          </select>
        </div>
        <div>
          <label htmlFor="password" className="label">Password</label>
          <input id="password" type="password" className="input" autoComplete="new-password" value={form.password}
            onChange={set('password')} aria-describedby="pw-help" />
          <p id="pw-help" className={`mt-1 text-xs ${problems.length ? 'text-amber-700' : 'text-slate-500'}`}>
            {problems.length ? `Needs ${problems.join(', ')}.` : 'At least 8 characters with an uppercase letter and a digit.'}
          </p>
        </div>
        <div>
          <label htmlFor="confirm" className="label">Confirm password</label>
          <input id="confirm" type="password" className="input" autoComplete="new-password" value={form.confirm}
            onChange={set('confirm')} />
          {mismatch && <p className="mt-1 text-xs text-red-700">Passwords do not match.</p>}
        </div>
        <button type="submit" className="btn-primary w-full" disabled={busy || !complete}>
          {busy ? 'Creating account…' : 'Create account'}
        </button>
      </form>
      <p className="mt-4 text-center text-sm text-slate-600">
        Already registered? <Link to="/login" className="font-medium text-indigo-700 hover:underline">Sign in</Link>
      </p>
    </AuthShell>
  )
}
