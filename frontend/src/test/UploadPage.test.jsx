import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import UploadPage from '../pages/UploadPage'

vi.mock('../api/endpoints', () => ({
  imagesApi: { upload: vi.fn() },
  assessmentsApi: { assess: vi.fn() },
}))
const { imagesApi, assessmentsApi } = await import('../api/endpoints')

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/upload']} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes>
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/results/:id" element={<p>results page</p>} />
      </Routes>
    </MemoryRouter>,
  )
}

const pngFile = () => new File([new Uint8Array([137, 80, 78, 71])], 'chest.png', { type: 'image/png' })

describe('UploadPage', () => {
  beforeEach(() => vi.clearAllMocks())

  it('blocks unsupported files on the client (TC-06)', async () => {
    renderPage()
    const input = screen.getByTestId('file-input')
    fireEvent.change(input, { target: { files: [new File(['hello'], 'notes.txt', { type: 'text/plain' })] } })
    expect(await screen.findByRole('alert')).toHaveTextContent('Unsupported file type')
    expect(screen.getByRole('button', { name: 'Upload and assess' })).toBeDisabled()
    expect(imagesApi.upload).not.toHaveBeenCalled()
  })

  it('uploads, assesses and redirects to the results (TC-05)', async () => {
    imagesApi.upload.mockResolvedValue({ id: 'img-1' })
    assessmentsApi.assess.mockResolvedValue({ id: 'res-1' })
    renderPage()
    fireEvent.change(screen.getByTestId('file-input'), { target: { files: [pngFile()] } })
    expect(screen.getByText('chest.png')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('radio', { name: /X-ray/ }))
    expect(screen.getByRole('radio', { name: /X-ray/ })).toHaveAttribute('aria-checked', 'true')
    await userEvent.click(screen.getByRole('button', { name: 'Upload and assess' }))
    expect(await screen.findByText('results page')).toBeInTheDocument()
    expect(imagesApi.upload).toHaveBeenCalledWith(expect.any(File), 'xray', expect.any(Function))
    expect(assessmentsApi.assess).toHaveBeenCalledWith('img-1')
  })

  it('shows the server error when the assessment fails', async () => {
    imagesApi.upload.mockResolvedValue({ id: 'img-1' })
    assessmentsApi.assess.mockRejectedValue({ response: { status: 503, data: { detail: 'No active model is registered.' } } })
    renderPage()
    fireEvent.change(screen.getByTestId('file-input'), { target: { files: [pngFile()] } })
    await userEvent.click(screen.getByRole('button', { name: 'Upload and assess' }))
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('No active model is registered.'))
  })
})
