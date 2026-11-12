import { useEffect, useState } from 'react'
import { imagesApi } from '../api/endpoints'

/** Image served by an authenticated endpoint: fetched as a blob with the bearer token. */
export default function AuthImage({ imageId, kind = 'preview', alt, className = '' }) {
  const [src, setSrc] = useState(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let url = null
    let cancelled = false
    setFailed(false)
    setSrc(null)
    imagesApi
      .blob(imageId, kind)
      .then((blob) => {
        if (cancelled) return
        url = URL.createObjectURL(blob)
        setSrc(url)
      })
      .catch(() => !cancelled && setFailed(true))
    return () => {
      cancelled = true
      if (url) URL.revokeObjectURL(url)
    }
  }, [imageId, kind])

  if (failed) {
    return <div className={`flex items-center justify-center bg-slate-100 text-xs text-slate-400 ${className}`}>No preview</div>
  }
  if (!src) return <div className={`animate-pulse bg-slate-100 ${className}`} aria-label="Loading image" />
  return <img src={src} alt={alt} className={className} draggable={false} />
}
