// Display name of the application. The project has no product name yet: change it here only.
export const APP_TITLE = 'AI-Based Medical Image Quality and Reliability Assessment'
export const APP_SHORT_TITLE = 'Medical Image Quality Assessment'
export const PROJECT_CODE = 'FYP 23F-34'

export const MODALITIES = [
  { value: 'xray', label: 'X-ray', hint: 'Radiograph' },
  { value: 'ct', label: 'CT', hint: 'Computed tomography' },
  { value: 'mri', label: 'MRI', hint: 'Magnetic resonance' },
  { value: 'retinal', label: 'Retinal fundus', hint: 'Fundus photograph' },
]

export const MODALITY_LABEL = Object.fromEntries(MODALITIES.map((m) => [m.value, m.label]))

export const MODALITY_SOURCE_LABEL = {
  user: 'selected by user',
  dicom: 'from DICOM metadata',
  model: 'detected by model',
}

export const VERDICTS = ['Pass', 'Review', 'Reject']

export const MAX_UPLOAD_MB = 50
export const ACCEPTED_EXTENSIONS = ['.dcm', '.dicom', '.png', '.jpg', '.jpeg']

export const DISCLAIMER =
  'Academic prototype (FYP 23F-34). Not FDA-approved or CE-marked. Not for clinical diagnosis.'
