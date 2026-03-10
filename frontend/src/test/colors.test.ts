import { describe, it, expect } from 'vitest'
import { clusterColor, CLUSTER_COLORS } from '../lib/colors'

describe('colors', () => {
  it('returns correct color for each cluster type', () => {
    expect(clusterColor('integration')).toBe('#3b7fff')
    expect(clusterColor('transformation')).toBe('#9b59b6')
    expect(clusterColor('arbitration')).toBe('#e67e22')
    expect(clusterColor('routing')).toBe('#1abc9c')
    expect(clusterColor('dormant')).toBe('#2a2a2a')
  })

  it('returns fallback for unknown cluster type', () => {
    expect(clusterColor('unknown')).toBe('#888888')
  })

  it('has 5 cluster colors defined', () => {
    expect(Object.keys(CLUSTER_COLORS)).toHaveLength(5)
  })
})
