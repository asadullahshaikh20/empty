import { Link, Navigate, Route, Routes } from 'react-router-dom'
import ProtectedRoute, { GuestOnlyRoute } from './auth/ProtectedRoute'
import Layout from './components/Layout'
import HistoryPage from './pages/HistoryPage'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import ResultsPage from './pages/ResultsPage'
import UploadPage from './pages/UploadPage'

function NotFound() {
  return (
    <div className="card text-center">
      <p className="text-lg font-semibold">Page not found</p>
      <Link to="/upload" className="mt-2 inline-block text-indigo-700 hover:underline">Go to upload</Link>
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route element={<GuestOnlyRoute />}>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
      </Route>
      <Route element={<ProtectedRoute />}>
        <Route element={<Layout />}>
          <Route path="/" element={<Navigate to="/upload" replace />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/results/:id" element={<ResultsPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Route>
    </Routes>
  )
}
