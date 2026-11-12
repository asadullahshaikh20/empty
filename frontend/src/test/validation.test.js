import { describe, expect, it } from 'vitest'
import { errorMessage } from '../api/client'
import { formatBytes, passwordProblems, validateFile } from '../lib/validation'

const file = (name, size) => ({ name, size })

describe('validateFile', () => {
  it('accepts DICOM, PNG and JPEG up to 50 MB', () => {
    for (const name of ['a.dcm', 'b.DICOM', 'c.png', 'd.JPG', 'e.jpeg']) {
      expect(validateFile(file(name, 1000))).toBeNull()
    }
    expect(validateFile(file('big.png', 50 * 1024 * 1024))).toBeNull()
  })

  it('rejects other types, empty and oversized files (TC-06)', () => {
    expect(validateFile(file('notes.txt', 10))).toMatch(/Unsupported file type/)
    expect(validateFile(file('noextension', 10))).toMatch(/Unsupported file type/)
    expect(validateFile(file('a.png', 0))).toMatch(/empty/)
    expect(validateFile(file('a.png', 50 * 1024 * 1024 + 1))).toMatch(/larger than 50 MB/)
    expect(validateFile(null)).toMatch(/choose a file/)
  })
})

describe('passwordProblems', () => {
  it('mirrors the backend policy', () => {
    expect(passwordProblems('Secret123')).toEqual([])
    expect(passwordProblems('short1A')).toContain('at least 8 characters')
    expect(passwordProblems('alllower1')).toContain('one uppercase letter')
    expect(passwordProblems('NoDigitsHere')).toContain('one digit')
    expect(passwordProblems('A1' + 'x'.repeat(80))).toContain('at most 72 bytes')
  })
})

describe('formatBytes', () => {
  it('formats sizes', () => {
    expect(formatBytes(500)).toBe('500 B')
    expect(formatBytes(2048)).toBe('2.0 KB')
    expect(formatBytes(5 * 1024 * 1024)).toBe('5.0 MB')
  })
})

describe('errorMessage', () => {
  it('handles FastAPI error shapes', () => {
    expect(errorMessage({ response: { data: { detail: 'Email already registered' } } })).toBe('Email already registered')
    expect(errorMessage({ response: { data: { detail: { message: 'Quality assessment failed' } } } })).toBe(
      'Quality assessment failed',
    )
    const validation = { response: { data: { detail: [{ loc: ['body', 'password'], msg: 'Value error, Password must contain one digit' }] } } }
    expect(errorMessage(validation)).toBe('password: Password must contain one digit')
    expect(errorMessage({ message: 'Network Error' })).toMatch(/Cannot reach the server/)
    expect(errorMessage({ response: { data: {} } }, 'fallback')).toBe('fallback')
  })
})
