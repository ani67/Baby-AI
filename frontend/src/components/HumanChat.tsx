import { useState, useRef } from 'react'
import { api } from '../lib/api'

interface Message {
  role: 'human' | 'model'
  text: string
}

export function HumanChat() {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [waiting, setWaiting] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const send = async () => {
    const text = input.trim()
    if (!text || waiting) return
    setInput('')
    setMessages(prev => [...prev, { role: 'human', text }])
    setWaiting(true)
    try {
      const { message } = await api.chat(text)
      setMessages(prev => [...prev, { role: 'model', text: message }])
    } catch {
      setMessages(prev => [
        ...prev,
        { role: 'model', text: '[no response]' },
      ])
    } finally {
      setWaiting(false)
      inputRef.current?.focus()
    }
  }

  return (
    <div className="human-chat">
      <div className="chat-header">TALK TO IT</div>
      <div className="chat-messages">
        {messages.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role}`}>
            <span className="role">{m.role === 'human' ? 'YOU' : 'IT'}</span>
            {m.text}
          </div>
        ))}
        {waiting && (
          <div className="chat-msg model thinking">
            <span className="role">IT</span>
            <span className="dots">···</span>
          </div>
        )}
      </div>
      <div className="chat-input-row">
        <input
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && send()}
          placeholder="say something"
          disabled={waiting}
        />
        <button onClick={send} disabled={waiting || !input.trim()}>
          →
        </button>
      </div>
    </div>
  )
}
