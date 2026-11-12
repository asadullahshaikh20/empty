import { ACCEPTED_EXTENSIONS, MAX_UPLOAD_MB } from './constants'

/** Client-side file check before upload (SDS 3.1). Returns an error string or null. */
export function validateFile(file) {
  if (!file) return 'Please choose a file.'
  const name = (file.name || '').toLowerCase()
  const dot = name.lastIndexOf('.')
  const ext = dot >= 0 ? name.slice(dot) : ''
  if (!ACCEPTED_EXTENSIONS.includes(ext)) {
    return 'Unsupported file type. Please upload a DICOM (.dcm), PNG or JPEG image.'
  }
  if (file.size === 0) return 'The file is empty.'
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return `The file is larger than ${MAX_UPLOAD_MB} MB.`
  return null
}

/** Mirrors the backend password policy (FR-AUTH-02). Returns a list of unmet rules. */
export function passwordProblems(password) {
  const problems = []
  if (password.length < 8) problems.push('at least 8 characters')
  if (!/[A-Z]/.test(password)) problems.push('one uppercase letter')
  if (!/[0-9]/.test(password)) problems.push('one digit')
  if (new TextEncoder().encode(password).length > 72) problems.push('at most 72 bytes')
  return problems
}

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function formatDateTime(iso) {
  if (!iso) return ''
  return new Date(iso).toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}
