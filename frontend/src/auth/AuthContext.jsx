import { useCallback, useEffect, useMemo, useState } from 'react'
import { refreshSession, setAccessToken, setOnAuthLost } from '../api/client'
import { authApi } from '../api/endpoints'
import { AuthContext } from './context'

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [status, setStatus] = useState('loading') // loading | authenticated | anonymous

  const applySession = useCallback((data) => {
    setAccessToken(data.access_token)
    setUser(data.user)
    setStatus('authenticated')
  }, [])

  const clearSession = useCallback(() => {
    setAccessToken(null)
    setUser(null)
    setStatus('anonymous')
  }, [])

  // Restore the session from the refresh cookie on page load.
  useEffect(() => {
    let cancelled = false
    refreshSession()
      .then((data) => !cancelled && applySession(data))
      .catch(() => !cancelled && clearSession())
    return () => {
      cancelled = true
    }
  }, [applySession, clearSession])

  useEffect(() => setOnAuthLost(clearSession), [clearSession])

  const login = useCallback(
    async (email, password) => applySession(await authApi.login(email, password)),
    [applySession],
  )
  const register = useCallback(async (payload) => applySession(await authApi.register(payload)), [applySession])
  const logout = useCallback(async () => {
    try {
      await authApi.logout()
    } finally {
      clearSession()
    }
  }, [clearSession])

  const value = useMemo(() => ({ user, status, login, register, logout }), [user, status, login, register, logout])
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
