import axios from 'axios'

// Access token lives in memory only (SDS 3.1). The refresh token is an httpOnly cookie.
let accessToken = null
let onAuthLost = () => {}
let refreshPromise = null

export function setAccessToken(token) {
  accessToken = token
}

export function getAccessToken() {
  return accessToken
}

export function setOnAuthLost(handler) {
  onAuthLost = handler
}

export const api = axios.create({ baseURL: '/api', withCredentials: true })

api.interceptors.request.use((config) => {
  if (accessToken) {
    config.headers = config.headers ?? {}
    config.headers.Authorization = `Bearer ${accessToken}`
  }
  return config
})

/** Obtain a new access token from the refresh cookie. Concurrent callers share one request. */
export function refreshSession() {
  if (!refreshPromise) {
    refreshPromise = axios
      .post('/api/auth/refresh', null, { withCredentials: true })
      .then((res) => {
        setAccessToken(res.data.access_token)
        return res.data
      })
      .finally(() => {
        refreshPromise = null
      })
  }
  return refreshPromise
}

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const original = error.config
    const status = error.response?.status
    const isAuthCall = original?.url?.startsWith('/auth/')
    if (status === 401 && original && !original._retried && !isAuthCall) {
      original._retried = true
      try {
        await refreshSession()
        return api(original)
      } catch {
        setAccessToken(null)
        onAuthLost()
      }
    }
    return Promise.reject(error)
  },
)

/** Human readable message from an Axios error (FastAPI detail formats included). */
export function errorMessage(error, fallback = 'Something went wrong. Please try again.') {
  if (!error?.response) {
    return error?.message === 'Network Error' ? 'Cannot reach the server. Is the backend running?' : fallback
  }
  const detail = error.response.data?.detail
  if (typeof detail === 'string') return detail
  if (detail && typeof detail === 'object' && !Array.isArray(detail) && detail.message) return detail.message
  if (Array.isArray(detail) && detail.length) {
    return detail
      .map((d) => {
        const field = Array.isArray(d.loc) ? d.loc[d.loc.length - 1] : ''
        const msg = String(d.msg || '').replace(/^Value error, /, '')
        return field && field !== 'body' ? `${field}: ${msg}` : msg
      })
      .join('; ')
  }
  return fallback
}
