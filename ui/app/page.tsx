"use client";

import { useState, useRef, useEffect, useCallback } from "react";

const API = "http://localhost:8000";

// ---------- types ----------

interface Message {
  role: "user" | "ai";
  text: string;
  elapsedMs?: number;
  prompt?: string;
}

interface Status {
  narrative: string;
  episodes_stored: number;
  facts_stored: number;
  training_queue_size: number;
  uptime_seconds: number;
}

interface SleepStatus {
  state: "awake" | "sleeping" | "waking";
  sleeping_since: string | null;
  elapsed_seconds: number | null;
  last_sleep_duration: number | null;
}

interface Session {
  id: string;
  title: string;
  created_at: number;
  message_count: number;
}

// ---------- Sleep Bar ----------

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function SleepBar({
  sleepStatus,
  fading,
}: {
  sleepStatus: SleepStatus;
  fading: boolean;
}) {
  const [elapsed, setElapsed] = useState(sleepStatus.elapsed_seconds ?? 0);

  useEffect(() => {
    if (sleepStatus.state !== "sleeping") return;
    setElapsed(sleepStatus.elapsed_seconds ?? 0);
    const t0 = Date.now();
    const base = sleepStatus.elapsed_seconds ?? 0;
    const interval = setInterval(() => {
      setElapsed(base + (Date.now() - t0) / 1000);
    }, 1000);
    return () => clearInterval(interval);
  }, [sleepStatus.state, sleepStatus.elapsed_seconds]);

  return (
    <div
      className={`flex items-center justify-between px-4 py-2 border-b border-[#222] ${fading ? "sleep-bar-exit" : ""}`}
    >
      <div className="flex items-center gap-2">
        <span className="sleep-dot inline-block w-1.5 h-1.5 rounded-full bg-[#7b8fb8]" />
        <span className="text-xs text-[#667] tracking-wide">sleeping</span>
      </div>
      <span className="font-mono text-xs text-[#556]">
        {formatElapsed(elapsed)}
      </span>
    </div>
  );
}

// ---------- Sidebar ----------

function Sidebar({
  sessions,
  currentSessionId,
  status,
  sleeping,
  statusLoading,
  onNewChat,
  onSwitchSession,
  onDeleteSession,
  onRefresh,
  onSleep,
  pollSleep,
}: {
  sessions: Session[];
  currentSessionId: string | null;
  status: Status | null;
  sleeping: boolean;
  statusLoading: boolean;
  onNewChat: () => void;
  onSwitchSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
  onRefresh: () => void;
  onSleep: () => void;
  pollSleep: () => void;
}) {
  return (
    <div className="flex flex-col h-full border-l border-[#222]">
      {/* Conversations section */}
      <div className="flex flex-col flex-1 min-h-0 p-4">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs tracking-widest uppercase text-[#555]">
            Chats
          </span>
          <button
            onClick={onNewChat}
            className="text-xs text-[#555] hover:text-[var(--accent)] transition-colors cursor-pointer"
          >
            + new
          </button>
        </div>

        <div className="flex-1 overflow-y-auto flex flex-col gap-0.5">
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`group flex items-center justify-between px-2 py-1.5 rounded cursor-pointer transition-colors ${
                s.id === currentSessionId
                  ? "bg-[#1a1a2e] text-[var(--foreground)]"
                  : "text-[#777] hover:bg-[#161616] hover:text-[#aaa]"
              }`}
              onClick={() => onSwitchSession(s.id)}
            >
              <span className="text-xs truncate flex-1 mr-2">{s.title}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteSession(s.id);
                }}
                className="text-[10px] text-[#333] opacity-0 group-hover:opacity-100 hover:text-[#a55] transition-all cursor-pointer flex-shrink-0"
              >
                ×
              </button>
            </div>
          ))}
          {sessions.length === 0 && (
            <div className="text-xs text-[#333]">no conversations yet</div>
          )}
        </div>
      </div>

      {/* System section */}
      <div className="flex flex-col gap-3 p-4 border-t border-[#222]">
        <div className="flex items-center justify-between">
          <span className="text-xs tracking-widest uppercase text-[#555]">
            System
          </span>
          <div className="flex items-center gap-3">
            <button
              onClick={onSleep}
              disabled={sleeping}
              className={`text-xs transition-colors cursor-pointer ${
                sleeping
                  ? "text-[#333] cursor-default"
                  : "text-[#555] hover:text-[var(--accent)]"
              }`}
            >
              {sleeping ? "sleeping" : "sleep"}
            </button>
            <button
              onClick={onRefresh}
              className="text-xs text-[#555] hover:text-[var(--accent)] transition-colors cursor-pointer"
            >
              {statusLoading ? "..." : "refresh"}
            </button>
          </div>
        </div>

        {status ? (
          <>
            <div className="font-mono text-xs leading-relaxed text-[#999]">
              {status.narrative}
            </div>
            <div className="flex flex-col gap-1.5 font-mono text-xs">
              <StatRow label="episodes" value={status.episodes_stored} />
              <StatRow label="facts" value={status.facts_stored} />
              <StatRow label="train queue" value={status.training_queue_size} />
              <StatRow
                label="uptime"
                value={`${Math.floor(status.uptime_seconds)}s`}
              />
            </div>
          </>
        ) : (
          <div className="text-xs text-[#555]">loading...</div>
        )}
      </div>
    </div>
  );
}

function StatRow({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="flex justify-between">
      <span className="text-[#555]">{label}</span>
      <span className="text-[var(--foreground)]">{value}</span>
    </div>
  );
}

// ---------- Correction Input ----------

function CorrectionInput({
  prompt,
  onSubmit,
  onCancel,
}: {
  prompt: string;
  onSubmit: (correction: string) => void;
  onCancel: () => void;
}) {
  const [value, setValue] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  return (
    <div className="mt-2 flex gap-2">
      <input
        ref={inputRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && value.trim()) {
            onSubmit(value.trim());
          }
          if (e.key === "Escape") {
            onCancel();
          }
        }}
        placeholder="correct answer..."
        className="flex-1 bg-[#1a1a1a] border border-[#333] px-2 py-1 text-xs font-mono text-[var(--foreground)] outline-none focus:border-[var(--accent)]"
      />
      <button
        onClick={() => value.trim() && onSubmit(value.trim())}
        className="text-xs text-[var(--accent)] hover:underline cursor-pointer"
      >
        send
      </button>
      <button
        onClick={onCancel}
        className="text-xs text-[#555] hover:text-[var(--foreground)] cursor-pointer"
      >
        cancel
      </button>
    </div>
  );
}

// ---------- Message Bubble ----------

function MessageBubble({ msg }: { msg: Message }) {
  const [showCorrection, setShowCorrection] = useState(false);
  const [corrected, setCorrected] = useState(false);
  const isUser = msg.role === "user";

  const handleSubmitCorrection = async (correction: string) => {
    try {
      await fetch(`${API}/correct`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: msg.prompt || "", correction }),
      });
      setCorrected(true);
      setShowCorrection(false);
    } catch {
      // silently fail
    }
  };

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[75%] group ${isUser ? "text-right" : "text-left"}`}
      >
        <div
          className={`inline-block px-3 py-2 text-sm whitespace-pre-wrap ${
            isUser
              ? "bg-[#1a1a2e] text-[var(--foreground)]"
              : "bg-[var(--surface)] font-mono text-[#c8c8c8]"
          }`}
        >
          {msg.text}
        </div>

        {!isUser && (
          <div className="flex items-center gap-3 mt-0.5 min-h-[18px]">
            {msg.elapsedMs !== undefined && (
              <span className="text-[10px] text-[#444]">
                {msg.elapsedMs}ms
              </span>
            )}
            {!corrected && !showCorrection && (
              <button
                onClick={() => setShowCorrection(true)}
                className="text-[10px] text-[#333] opacity-0 group-hover:opacity-100 hover:text-[var(--accent)] transition-all cursor-pointer"
              >
                correct this
              </button>
            )}
            {corrected && (
              <span className="text-[10px] text-[#4a6] italic">corrected</span>
            )}
          </div>
        )}

        {showCorrection && (
          <CorrectionInput
            prompt={msg.prompt || ""}
            onSubmit={handleSubmitCorrection}
            onCancel={() => setShowCorrection(false)}
          />
        )}
      </div>
    </div>
  );
}

// ---------- helpers ----------

function historyToMessages(
  data: { prompt: string; response: string }[],
): Message[] {
  const msgs: Message[] = [];
  for (const entry of data) {
    msgs.push({ role: "user", text: entry.prompt });
    msgs.push({ role: "ai", text: entry.response, prompt: entry.prompt });
  }
  return msgs;
}

// ---------- Main Page ----------

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<Status | null>(null);
  const [statusLoading, setStatusLoading] = useState(false);
  const [statusOpen, setStatusOpen] = useState(true);
  const [sleepStatus, setSleepStatus] = useState<SleepStatus | null>(null);
  const [sleepBarFading, setSleepBarFading] = useState(false);
  const [queuedMessage, setQueuedMessage] = useState<string | null>(null);
  const [showQueued, setShowQueued] = useState(false);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const prevSleepState = useRef<string>("awake");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const sleeping = sleepStatus?.state === "sleeping";

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  // ---------- load sessions + history on mount ----------

  const fetchSessions = useCallback(async () => {
    try {
      const res = await fetch(`${API}/sessions`);
      const data = await res.json();
      setSessions(data.sessions);
      setCurrentSessionId(data.current);
    } catch {
      // backend not ready
    }
  }, []);

  const loadCurrentHistory = useCallback(async () => {
    try {
      const res = await fetch(`${API}/history`);
      const data: { prompt: string; response: string }[] = await res.json();
      setMessages(historyToMessages(data));
    } catch {
      // backend not ready
    }
  }, []);

  useEffect(() => {
    fetchSessions();
    loadCurrentHistory();
  }, [fetchSessions, loadCurrentHistory]);

  // ---------- session actions ----------

  const handleNewChat = async () => {
    try {
      const res = await fetch(`${API}/sessions/new`, { method: "POST" });
      const data = await res.json();
      setCurrentSessionId(data.id);
      setMessages([]);
      await fetchSessions();
      textareaRef.current?.focus();
    } catch {}
  };

  const handleSwitchSession = async (id: string) => {
    if (id === currentSessionId) return;
    try {
      await fetch(`${API}/sessions/${id}/switch`, { method: "POST" });
      setCurrentSessionId(id);
      await loadCurrentHistory();
      await fetchSessions();
    } catch {}
  };

  const handleDeleteSession = async (id: string) => {
    try {
      const res = await fetch(`${API}/sessions/${id}`, { method: "DELETE" });
      const data = await res.json();
      if (data.ok) {
        setCurrentSessionId(data.current);
        await fetchSessions();
        await loadCurrentHistory();
      }
    } catch {}
  };

  // ---------- system status polling ----------

  const fetchStatus = useCallback(async () => {
    setStatusLoading(true);
    try {
      const res = await fetch(`${API}/status`);
      const data = await res.json();
      setStatus(data);
    } catch {
      // backend not ready
    }
    setStatusLoading(false);
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  // ---------- sleep status polling ----------

  const pollSleep = useCallback(async () => {
    try {
      const res = await fetch(`${API}/sleep-status`);
      const data: SleepStatus = await res.json();
      setSleepStatus(data);
      return data;
    } catch {
      return null;
    }
  }, []);

  useEffect(() => {
    pollSleep();
    const interval = setInterval(pollSleep, sleeping ? 1000 : 5000);
    return () => clearInterval(interval);
  }, [pollSleep, sleeping]);

  // ---------- wake transition: fade bar + send queued ----------

  const fetchReply = useCallback(
    async (prompt: string) => {
      setLoading(true);
      try {
        const res = await fetch(`${API}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt }),
        });
        const data = await res.json();
        setMessages((prev) => [
          ...prev,
          {
            role: "ai",
            text: data.response,
            elapsedMs: data.elapsed_ms,
            prompt,
          },
        ]);
        // Refresh sessions to update title/count
        fetchSessions();
      } catch {
        setMessages((prev) => [
          ...prev,
          { role: "ai", text: "[error: backend unreachable]", prompt },
        ]);
      }
      setLoading(false);
    },
    [fetchSessions],
  );

  const sendPrompt = useCallback(
    async (prompt: string) => {
      setMessages((prev) => [...prev, { role: "user", text: prompt }]);
      fetchReply(prompt);
    },
    [fetchReply],
  );

  useEffect(() => {
    if (!sleepStatus) return;
    const prev = prevSleepState.current;
    const curr = sleepStatus.state;

    if (prev === "sleeping" && curr === "awake") {
      setSleepBarFading(true);
      setTimeout(() => setSleepBarFading(false), 600);

      if (queuedMessage) {
        const msg = queuedMessage;
        setQueuedMessage(null);
        setShowQueued(false);
        fetchReply(msg);
      }
    }

    prevSleepState.current = curr;
  }, [sleepStatus, queuedMessage, fetchReply]);

  // ---------- send / queue ----------

  const sendMessage = async () => {
    const prompt = input.trim();
    if (!prompt || loading) return;

    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    if (sleeping) {
      setQueuedMessage(prompt);
      setMessages((prev) => [...prev, { role: "user", text: prompt }]);
      setShowQueued(true);
      return;
    }

    sendPrompt(prompt);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 150) + "px";
  };

  return (
    <div className="flex h-screen">
      {/* Chat area */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#222]">
          <span className="text-xs tracking-widest uppercase text-[#555]">
            Kaida Reed
          </span>
          <button
            onClick={() => setStatusOpen(!statusOpen)}
            className="text-xs text-[#555] hover:text-[var(--accent)] transition-colors cursor-pointer"
          >
            {statusOpen ? "hide panel" : "show panel"}
          </button>
        </div>

        {/* Sleep bar */}
        {sleepStatus && (sleeping || sleepBarFading) && (
          <SleepBar sleepStatus={sleepStatus} fading={sleepBarFading} />
        )}

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-4">
          <div className="max-w-3xl mx-auto flex flex-col gap-3">
            {messages.length === 0 && (
              <div className="text-center text-[#333] text-sm mt-20">
                start typing below
              </div>
            )}
            {messages.map((msg, i) => (
              <MessageBubble key={i} msg={msg} />
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-[var(--surface)] px-3 py-2 font-mono text-sm">
                  <span className="cursor-blink">_</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input */}
        <div className="border-t border-[#222] px-4 py-3">
          <div className="max-w-3xl mx-auto">
            {showQueued && (
              <div className="text-[11px] text-[#556] mb-1.5">
                message queued, will send when awake
              </div>
            )}
            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder={
                sleeping
                  ? "model is sleeping, will respond when done"
                  : "message..."
              }
              rows={1}
              className={`w-full text-sm px-3 py-2 outline-none resize-none border transition-colors ${
                sleeping
                  ? "bg-[#0f0f0f] text-[#666] border-[#1a1a1a] placeholder-[#333]"
                  : "bg-[var(--surface)] text-[var(--foreground)] border-[#222] focus:border-[#444] placeholder-[#444]"
              }`}
            />
          </div>
        </div>
      </div>

      {/* Right panel: conversations + system */}
      {statusOpen && (
        <div className="w-64 flex-shrink-0 hidden md:block">
          <Sidebar
            sessions={sessions}
            currentSessionId={currentSessionId}
            status={status}
            sleeping={sleeping}
            statusLoading={statusLoading}
            onNewChat={handleNewChat}
            onSwitchSession={handleSwitchSession}
            onDeleteSession={handleDeleteSession}
            onRefresh={fetchStatus}
            onSleep={async () => {
              try {
                await fetch(`${API}/sleep`, { method: "POST" });
                setTimeout(pollSleep, 200);
              } catch {}
            }}
            pollSleep={pollSleep}
          />
        </div>
      )}
    </div>
  );
}
