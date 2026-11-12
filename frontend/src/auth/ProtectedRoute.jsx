import { Navigate, Outlet, useLocation } from 'react-router-dom'
import Spinner from '../components/Spinner'
import { useAuth } from './context'

export default function ProtectedRoute() {
  const { status } = useAuth()
  const location = useLocation()
  if (status === 'loading') return <Spinner label="Restoring session…" />
  if (status !== 'authenticated') return <Navigate to="/login" replace state={{ from: location }} />
  return <Outlet />
}

export function GuestOnlyRoute() {
  const { status } = useAuth()
  if (status === 'loading') return <Spinner label="Loading…" />
  if (status === 'authenticated') return <Navigate to="/upload" replace />
  return <Outlet />
}
