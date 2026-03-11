import { useState } from 'react'
import { useLoopStore } from '../store/loopStore'
import { useLoopControl } from '../hooks/useLoopControl'
import { api } from '../lib/api'

export function Controls() {
  const status = useLoopStore(s => s.status)
  const { start, pause, resume, step, reset, setStage, setSpeed } =
    useLoopControl()
  const [showResetModal, setShowResetModal] = useState(false)

  const isRunning = status?.state === 'running'
  const isPaused = status?.state === 'paused' || status?.state === 'idle'

  return (
    <div className="controls">
      {/* Playback */}
      <div className="control-group">
        {isRunning ? (
          <button onClick={pause} className="btn-primary">⏸ PAUSE</button>
        ) : (
          <button
            onClick={isPaused ? (status?.state === 'idle' ? start : resume) : start}
            className="btn-primary accent"
          >
            ▶ {status?.state === 'idle' ? 'START' : 'RESUME'}
          </button>
        )}
        <button
          onClick={step}
          disabled={isRunning}
          className="btn-secondary"
        >
          STEP
        </button>
        <button onClick={() => setShowResetModal(true)} className="btn-ghost">
          RESET
        </button>
      </div>

      {showResetModal && (
        <ResetModal
          onConfirm={async (notes) => {
            await reset(notes)
            setShowResetModal(false)
          }}
          onCancel={() => setShowResetModal(false)}
        />
      )}

      {/* Speed */}
      <div className="control-group">
        <label>SPEED</label>
        <input
          type="range"
          min={0}
          max={2000}
          step={100}
          defaultValue={500}
          onChange={e => setSpeed(Number(e.target.value))}
        />
      </div>

      {/* Stage */}
      <div className="control-group">
        <label>STAGE</label>
        <div className="stage-buttons">
          {[0, 1, 2, 3, 4].map(s => (
            <button
              key={s}
              onClick={() => setStage(s)}
              className={`stage-btn ${status?.stage === s ? 'active' : ''}`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* Image URL paste */}
      <ImageUrlInput />
    </div>
  )
}

const RESET_FIELDS = [
  { key: 'architecture_state', label: 'Architecture State', placeholder: 'Describe the current graph structure, cluster count, layer depth...' },
  { key: 'signal_quality', label: 'Signal Quality', placeholder: 'How was the positive/negative signal? Threshold behavior, saturation...' },
  { key: 'why_reset', label: 'Why Reset', placeholder: 'What went wrong or what do you want to try differently...' },
  { key: 'what_was_learned', label: 'What Was Learned', placeholder: 'Key observations, patterns noticed, hypotheses confirmed/rejected...' },
] as const

interface ResetNotes {
  architecture_state: string
  signal_quality: string
  why_reset: string
  what_was_learned: string
}

function ResetModal({ onConfirm, onCancel }: {
  onConfirm: (notes: ResetNotes) => Promise<void>
  onCancel: () => void
}) {
  const [notes, setNotes] = useState<ResetNotes>({
    architecture_state: '',
    signal_quality: '',
    why_reset: '',
    what_was_learned: '',
  })
  const [submitting, setSubmitting] = useState(false)

  const allValid = RESET_FIELDS.every(f => notes[f.key].trim().length >= 10)

  const handleConfirm = async () => {
    setSubmitting(true)
    try {
      await onConfirm(notes)
    } catch {
      setSubmitting(false)
    }
  }

  return (
    <div className="reset-modal-overlay" onClick={onCancel}>
      <div className="reset-modal" onClick={e => e.stopPropagation()}>
        <h3>Reset Experiment</h3>
        <p className="reset-modal-desc">Document your observations before resetting. All fields require at least 10 characters.</p>

        {RESET_FIELDS.map(({ key, label, placeholder }) => (
          <div key={key} className="reset-field">
            <label>{label}</label>
            <textarea
              value={notes[key]}
              onChange={e => setNotes(prev => ({ ...prev, [key]: e.target.value }))}
              placeholder={placeholder}
              rows={3}
            />
            {notes[key].length > 0 && notes[key].trim().length < 10 && (
              <span className="reset-field-hint">{10 - notes[key].trim().length} more chars needed</span>
            )}
          </div>
        ))}

        <div className="reset-modal-actions">
          <button onClick={onCancel} className="btn-ghost" disabled={submitting}>Cancel</button>
          <button
            onClick={handleConfirm}
            className="btn-primary"
            disabled={!allValid || submitting}
          >
            {submitting ? 'Resetting...' : 'Confirm Reset'}
          </button>
        </div>
      </div>
    </div>
  )
}

function ImageUrlInput() {
  const [text, setText] = useState('')
  const [status, setStatus] = useState<'idle' | 'loading' | string>('idle')

  const submit = async () => {
    const urls = text.split('\n').map(s => s.trim()).filter(Boolean)
    if (urls.length === 0) return
    setStatus('loading')
    try {
      if (urls.length === 1) {
        await api.uploadImageUrl(urls[0])
        setStatus('1 added')
      } else {
        const res = await api.uploadImagesBulk(urls)
        console.log('[bulk upload]', res)
        setStatus(`${res.added}/${res.total} added`)
      }
      setText('')
      setTimeout(() => setStatus('idle'), 3000)
    } catch {
      setStatus('error')
      setTimeout(() => setStatus('idle'), 2000)
    }
  }

  const urls = text.split('\n').filter(s => s.trim())
  const label = status === 'loading' ? '...'
    : status === 'idle' ? (urls.length > 1 ? `ADD ${urls.length}` : 'ADD')
    : status

  return (
    <div className="image-url-input">
      <textarea
        placeholder="paste image URLs (one per line)"
        value={text}
        onChange={e => setText(e.target.value)}
        onKeyDown={e => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            submit()
          }
        }}
        rows={Math.min(text.split('\n').length || 1, 5)}
        style={{ width: 240, resize: 'vertical', minHeight: 22, maxHeight: 120 }}
      />
      <button
        onClick={submit}
        disabled={!text.trim() || status === 'loading'}
        className="btn-secondary"
      >
        {label}
      </button>
    </div>
  )
}
