import { api } from './client'

export const authApi = {
  login: (email, password) => api.post('/auth/login', { email, password }).then((r) => r.data),
  register: (payload) => api.post('/auth/register', payload).then((r) => r.data),
  logout: () => api.post('/auth/logout'),
  me: () => api.get('/auth/me').then((r) => r.data),
}

export const imagesApi = {
  upload: (file, modality, onProgress) => {
    const form = new FormData()
    form.append('file', file)
    if (modality) form.append('modality', modality)
    return api
      .post('/images/upload', form, {
        onUploadProgress: (e) => {
          if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
        },
      })
      .then((r) => r.data)
  },
  blob: (imageId, kind) => api.get(`/images/${imageId}/${kind}`, { responseType: 'blob' }).then((r) => r.data),
  remove: (imageId) => api.delete(`/images/${imageId}`),
}

export const assessmentsApi = {
  assess: (imageId) => api.post(`/assess/${imageId}`).then((r) => r.data),
  get: (id) => api.get(`/assessments/${id}`).then((r) => r.data),
  history: (params) => api.get('/assessments', { params }).then((r) => r.data),
  stats: () => api.get('/assessments/stats').then((r) => r.data),
}
