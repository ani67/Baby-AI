import { useEffect, useRef, useState } from 'react'
import { useDialogueStore, DialogueEntry as DialogueEntryType } from '../store/dialogueStore'
import { BASE_URL } from '../lib/constants'

export function DialogueFeed() {
  const entries = useDialogueStore(s => s.entries)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [entries.length])

  const positive = entries.filter(e => e.is_positive).length
  const negative = entries.length - positive

  return (
    <div className="dialogue-feed">
      <div className="feed-header">
        <span>DIALOGUE</span>
        <span className="step-counter">
          {entries.length} exchanges
          {entries.length > 0 && (
            <> &nbsp;·&nbsp; <span style={{ color: 'var(--positive)' }}>✓{positive}</span> <span style={{ color: 'var(--negative)' }}>✗{negative}</span></>
          )}
        </span>
      </div>
      <div className="feed-scroll">
        {entries.map((entry, i) => (
          <DialogueEntryRow key={`${entry.step}-${i}`} entry={entry} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

function DialogueEntryRow({ entry }: { entry: DialogueEntryType }) {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    requestAnimationFrame(() => setVisible(true))
  }, [])

  return (
    <div
      className={`dialogue-entry ${visible ? 'visible' : ''} ${
        entry.is_positive ? 'positive' : 'negative'
      }`}
    >
      <div className="entry-meta">
        <span className="step">#{entry.step}</span>
        <span className="curiosity">
          ◆ {(entry.curiosity_score * 100).toFixed(0)}
        </span>
        <span className={`signal ${entry.is_positive ? 'pos' : 'neg'}`}>
          {entry.is_positive ? '✓' : '✗'}
        </span>
      </div>
      {entry.image_url && (
        <img
          src={`${BASE_URL}${entry.image_url}`}
          alt=""
          className="dialogue-thumb"
        />
      )}
      <div className="question">
        <span className="label">Q</span>
        {entry.question}
      </div>
      <div className="answer">
        <span className="label">T</span>
        {entry.answer}
      </div>
      {entry.model_answer && (
        <div className="model-answer">
          <span className="label">M</span>{' '}
          {entry.model_answer}
        </div>
      )}
    </div>
  )
}
