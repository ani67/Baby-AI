import { describe, it, expect } from 'vitest'
import { api } from '../lib/api'
import { BASE_URL } from '../lib/constants'

describe('api module', () => {
  it('exports all required API methods', () => {
    expect(typeof api.start).toBe('function')
    expect(typeof api.pause).toBe('function')
    expect(typeof api.resume).toBe('function')
    expect(typeof api.step).toBe('function')
    expect(typeof api.reset).toBe('function')
    expect(typeof api.status).toBe('function')
    expect(typeof api.snapshot).toBe('function')
    expect(typeof api.setStage).toBe('function')
    expect(typeof api.setSpeed).toBe('function')
    expect(typeof api.chat).toBe('function')
    expect(typeof api.uploadImage).toBe('function')
  })

  it('BASE_URL points to localhost:8000', () => {
    expect(BASE_URL).toBe('http://localhost:8000')
  })
})
