# SPEC: Frontend
*Component 8 of 9 — The window into the growing mind*

---

## What it is

A React + Vite single-page application.
One screen, four panels, permanently visible.
No navigation. No pages. No modals.

The human watches the model learn in real time.
They can talk to it, control its speed, drop images in,
and watch the graph restructure itself.

The aesthetic is **dark lab / deep space** — not "AI dashboard."
Think: a scientist observing something alive, through glass.
Black background. Dim grid. Nodes are bioluminescent, not neon.
Edges are gossamer threads of light, not thick pipes.
Growth events should feel like watching cells divide.

---

## Location in the project

```
project/
  frontend/
    index.html
    vite.config.ts
    package.json
    src/
      main.tsx
      App.tsx
      components/
        LatentSpace.tsx       ← Three.js 3D visualizer
        DialogueFeed.tsx      ← Q&A stream
        HumanChat.tsx         ← Chat with the model
        Controls.tsx          ← Start/Pause/Step/Reset + sliders
        StatusBar.tsx         ← Step count, stage, graph stats
      hooks/
        useWebSocket.ts       ← WS connection + message dispatch
        useGraphState.ts      ← Local graph state, applies diffs
        useLoopControl.ts     ← Start/pause/step/reset API calls
      store/
        graphStore.ts         ← Zustand store for graph state
        loopStore.ts          ← Zustand store for loop status
        dialogueStore.ts      ← Zustand store for dialogue history
      lib/
        api.ts                ← typed wrappers for all HTTP calls
        colors.ts             ← cluster type → color mapping
        constants.ts          ← thresholds, animation durations
      styles/
        global.css
```

---

## Layout

```
┌──────────────────────────────────────────────────────────────┐
│  StatusBar                                         [Controls] │
├────────────────────────────────┬─────────────────────────────┤
│                                │                             │
│                                │  DialogueFeed               │
│   LatentSpace                  │  (scrolling Q&A stream)     │
│   (Three.js, fills the panel)  │                             │
│                                ├─────────────────────────────┤
│                                │                             │
│                                │  HumanChat                  │
│                                │  (talk to the model)        │
└────────────────────────────────┴─────────────────────────────┘
```

Left panel: 60% width, full height minus StatusBar.
Right panel: 40% width, split 60/40 between DialogueFeed and HumanChat.
StatusBar: full width, fixed height ~40px.
Controls: slide-out drawer from the right, or persistent panel above right column.

---

## Technology

```
React 18           UI framework
Vite 5             build + dev server
Three.js (r128)    3D latent space visualization
@react-three/fiber React bindings for Three.js (r8)
@react-three/drei  helper hooks for r3f (OrbitControls, etc.)
Zustand            lightweight state management
Tailwind CSS       utility styling
```

---

## Global style

```css
/* global.css */

:root {
  --bg:           #080808;
  --surface:      #0e0e0e;
  --surface-mid:  #141414;
  --border:       #1e1e1e;
  --text:         #c8c8c8;
  --text-dim:     #666;
  --text-bright:  #f0f0f0;
  --accent:       #4effa0;   /* bioluminescent green — the "alive" color */
  --accent-dim:   #1a3d2a;
  --positive:     #4effa0;
  --negative:     #ff6b4e;
  --warning:      #ffcc44;

  /* Cluster type colors — muted, not saturated */
  --cluster-integration:    #3b7fff;
  --cluster-transformation: #9b59b6;
  --cluster-arbitration:    #e67e22;
  --cluster-routing:        #1abc9c;
  --cluster-dormant:        #2a2a2a;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

html, body, #root {
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: var(--bg);
  color: var(--text);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  line-height: 1.5;
}
```

**Font: IBM Plex Mono.** Scientific. Legible at small sizes.
Appropriate for a system you are observing, not operating.
Load from Google Fonts or bundle locally.

---

---

# LatentSpace

---

## What it shows

The Baby Model's internal graph rendered in 3D.

Each **node** is a small sphere. Position = UMAP projection of its
512-dim weight vector. Color = cluster type. Size = activation level.

Each **edge** between clusters is a thin line connecting the
centroid positions of the two clusters, not individual nodes.
Cluster centroids = mean of member node positions.

**Active nodes** pulse when they fire (scale up briefly, then return).
**New edges** draw in from opacity 0 to their full opacity over 800ms.
**Pruned edges** fade out over 600ms, then are removed.
**BUD** events: parent node cluster briefly glows white, then two
child clusters expand from its position.
**INSERT** events: a new cluster materializes (opacity 0 → full)
between two existing ones.

The camera orbits automatically when idle.
The human can grab and rotate.

---

## Three.js scene setup

```tsx
// LatentSpace.tsx

import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'

export function LatentSpace() {
  return (
    <div className="latent-space-panel">
      <Canvas
        camera={{ position: [0, 0, 4], fov: 50 }}
        gl={{ antialias: true, alpha: false }}
        style={{ background: '#080808' }}
      >
        {/* Ambient + subtle point lights */}
        <ambientLight intensity={0.1} />
        <pointLight position={[2, 2, 2]} intensity={0.3} color="#4effa0" />

        {/* Dim grid plane at y = -1 */}
        <GridPlane />

        {/* All clusters and their nodes */}
        <GraphRenderer />

        {/* Auto-orbit when idle */}
        <OrbitControls
          enablePan={false}
          enableZoom={true}
          autoRotate={true}
          autoRotateSpeed={0.3}
          makeDefault
        />
      </Canvas>
    </div>
  )
}
```

---

## GraphRenderer

```tsx
function GraphRenderer() {
  const clusters = useGraphStore(s => s.clusters)
  const nodes    = useGraphStore(s => s.nodes)
  const edges    = useGraphStore(s => s.edges)
  const activations = useGraphStore(s => s.activations)

  return (
    <group>
      {/* Edges between cluster centroids */}
      {edges.map(edge => (
        <ClusterEdge
          key={`${edge.from}-${edge.to}`}
          edge={edge}
          fromPos={clusterCentroid(clusters, nodes, edge.from)}
          toPos={clusterCentroid(clusters, nodes, edge.to)}
        />
      ))}

      {/* Node spheres, grouped by cluster */}
      {clusters.map(cluster => (
        <ClusterGroup
          key={cluster.id}
          cluster={cluster}
          nodes={nodes.filter(n => n.cluster === cluster.id)}
          activation={activations[cluster.id] ?? 0}
        />
      ))}
    </group>
  )
}
```

---

## Individual node sphere

```tsx
function NodeSphere({ node, clusterType, activation }: NodeSphereProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const baseScale = 0.015 + node.activation_mean * 0.01
  const color = CLUSTER_COLORS[clusterType] ?? '#888'

  // Pulse on activation
  useFrame((_, delta) => {
    if (!meshRef.current) return
    const target = node.id in activeSet
      ? baseScale * 2.5
      : baseScale
    meshRef.current.scale.lerp(
      new THREE.Vector3(target, target, target),
      delta * 8
    )
  })

  return (
    <mesh
      ref={meshRef}
      position={node.pos}
      scale={[baseScale, baseScale, baseScale]}
    >
      <sphereGeometry args={[1, 8, 8]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={0.4 + activation * 0.6}
        roughness={0.6}
        metalness={0.2}
        transparent
        opacity={node.alive ? 0.9 : 0.15}
      />
    </mesh>
  )
}
```

---

## Edge line

```tsx
function ClusterEdge({ edge, fromPos, toPos }: EdgeProps) {
  const points = [
    new THREE.Vector3(...fromPos),
    new THREE.Vector3(...toPos)
  ]
  const lineRef = useRef<THREE.Line>(null)

  // Fade in newly formed edges
  const opacity = useSpring(0, {   // start at 0, animate to target
    target: edgeOpacity(edge.strength),
    stiffness: 80, damping: 20
  })

  return (
    <line ref={lineRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[new Float32Array(points.flatMap(p => p.toArray())), 3]}
        />
      </bufferGeometry>
      <lineBasicMaterial
        color="#4effa0"
        transparent
        opacity={opacity}
        linewidth={1}    // note: linewidth >1 requires LineSegments2
      />
    </line>
  )
}

function edgeOpacity(strength: number): number {
  if (strength < 0.2) return 0.1
  if (strength < 0.5) return 0.3
  if (strength < 0.8) return 0.6
  return 1.0
}
```

---

## GridPlane

```tsx
function GridPlane() {
  return (
    <gridHelper
      args={[10, 20, '#1a1a1a', '#111']}
      position={[0, -1.5, 0]}
      rotation={[0, 0, 0]}
    />
  )
}
```

---

## Growth event animations

These are triggered by the WebSocket `growth_events` array.
Each event type gets its own brief animation, then state is reconciled.

```tsx
function useGrowthAnimations(growthEvents: GrowthEvent[]) {
  // For each event, push a timed animation state
  // Components read from this to trigger their own animations

  useEffect(() => {
    growthEvents.forEach(event => {
      switch (event.type) {
        case 'BUD':
          triggerBudAnimation(event.from)    // flash parent, expand children
          break
        case 'CONNECT':
          triggerConnectAnimation(event.from, event.to)   // draw edge in
          break
        case 'PRUNE':
          triggerPruneAnimation(event.from, event.to)     // fade edge out
          break
        case 'INSERT':
          triggerInsertAnimation(event.new_cluster)       // materialize
          break
        case 'EXTEND':
          triggerExtendAnimation(event.new_cluster)       // rise from top
          break
        case 'DORMANT':
          triggerDormantAnimation(event.cluster)          // dim and shrink
          break
      }
    })
  }, [growthEvents])
}
```

---

---

# DialogueFeed

---

## What it shows

The live stream of questions the model asks and answers the teacher gives.
Most recent at the bottom. Auto-scrolls.
Entries flash in — not instant appear.

```tsx
interface DialogueEntry {
  step: number
  question: string
  answer: string
  curiosity_score: number
  is_positive: boolean
  stage: number
  timestamp: number
}

export function DialogueFeed() {
  const entries = useDialogueStore(s => s.entries)
  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom on new entry
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [entries.length])

  return (
    <div className="dialogue-feed">
      <div className="feed-header">
        <span>DIALOGUE</span>
        <span className="step-counter">{entries.length} exchanges</span>
      </div>
      <div className="feed-scroll">
        {entries.map(entry => (
          <DialogueEntry key={entry.step} entry={entry} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
```

## Individual entry

```tsx
function DialogueEntry({ entry }: { entry: DialogueEntry }) {
  const [visible, setVisible] = useState(false)

  // Fade in on mount
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
      <div className="question">
        <span className="label">Q</span>
        {entry.question}
      </div>
      <div className="answer">
        <span className="label">A</span>
        {entry.answer}
      </div>
    </div>
  )
}
```

CSS for the entry:

```css
.dialogue-entry {
  padding: 10px 12px;
  border-left: 2px solid var(--border);
  margin-bottom: 2px;
  opacity: 0;
  transform: translateY(4px);
  transition: opacity 300ms, transform 300ms;
}
.dialogue-entry.visible {
  opacity: 1;
  transform: translateY(0);
}
.dialogue-entry.negative {
  border-left-color: var(--negative);
}
.dialogue-entry.positive {
  border-left-color: var(--accent-dim);
}
.entry-meta {
  display: flex;
  gap: 10px;
  color: var(--text-dim);
  font-size: 10px;
  margin-bottom: 4px;
}
.question .label,
.answer .label {
  display: inline-block;
  width: 14px;
  color: var(--text-dim);
  margin-right: 6px;
}
.answer {
  color: var(--text-dim);
  margin-top: 2px;
}
```

Keep the last 200 entries in the store. Older entries are evicted.
The dialogue feed is not a history viewer — it's a live window.

---

---

# HumanChat

---

## What it shows

A minimal chat interface to talk to the Baby Model.
The model's responses will be broken and strange early on.
That's the point — you can watch language emerge.

```tsx
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
        { role: 'model', text: '[no response]' }
      ])
    } finally {
      setWaiting(false)
      inputRef.current?.focus()
    }
  }

  return (
    <div className="human-chat">
      <div className="chat-header">
        TALK TO IT
      </div>
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
```

```css
.chat-msg {
  padding: 6px 0;
  display: flex;
  gap: 10px;
}
.chat-msg .role {
  color: var(--text-dim);
  font-size: 10px;
  min-width: 28px;
  padding-top: 1px;
}
.chat-msg.model {
  color: var(--accent);
}
.chat-msg.model.thinking .dots {
  animation: blink 1.2s infinite;
}
@keyframes blink {
  0%, 100% { opacity: 0.2; }
  50% { opacity: 1; }
}
.chat-input-row {
  display: flex;
  gap: 6px;
  padding: 8px 0 4px;
  border-top: 1px solid var(--border);
}
.chat-input-row input {
  flex: 1;
  background: transparent;
  border: none;
  border-bottom: 1px solid var(--border);
  color: var(--text-bright);
  font-family: inherit;
  font-size: 12px;
  outline: none;
  padding: 4px 0;
}
.chat-input-row button {
  background: none;
  border: 1px solid var(--accent-dim);
  color: var(--accent);
  font-family: inherit;
  padding: 4px 10px;
  cursor: pointer;
}
.chat-input-row button:disabled {
  opacity: 0.3;
  cursor: default;
}
```

---

---

# Controls

---

## What it shows

The loop control interface. Compact. Always visible.
Not a sidebar — a tight cluster of controls above the right column.

```tsx
export function Controls() {
  const status = useLoopStore(s => s.status)
  const { start, pause, resume, step, reset, setStage, setSpeed } =
    useLoopControl()

  const isRunning = status?.state === 'running'
  const isPaused  = status?.state === 'paused' || status?.state === 'idle'

  return (
    <div className="controls">
      {/* Playback */}
      <div className="control-group">
        {isRunning ? (
          <button onClick={pause} className="btn-primary">⏸ PAUSE</button>
        ) : (
          <button onClick={isPaused ? resume : start} className="btn-primary accent">
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
        <button
          onClick={reset}
          className="btn-ghost"
        >
          RESET
        </button>
      </div>

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

      {/* Image drop */}
      <ImageDropZone />
    </div>
  )
}
```

---

## ImageDropZone

```tsx
function ImageDropZone() {
  const [dragging, setDragging] = useState(false)
  const [label, setLabel] = useState('')

  const onDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (!file || !file.type.startsWith('image/')) return
    await api.uploadImage(file, label || undefined)
    setLabel('')
  }

  return (
    <div
      className={`image-dropzone ${dragging ? 'dragging' : ''}`}
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <span>DROP IMAGE</span>
      <input
        placeholder="label (optional)"
        value={label}
        onChange={e => setLabel(e.target.value)}
        onClick={e => e.stopPropagation()}
      />
    </div>
  )
}
```

---

---

# StatusBar

---

```tsx
export function StatusBar() {
  const status = useLoopStore(s => s.status)
  const graphStats = status?.graph_summary ?? {}

  const stateColor = {
    idle:     'var(--text-dim)',
    running:  'var(--accent)',
    paused:   'var(--warning)',
    error:    'var(--negative)'
  }[status?.state ?? 'idle']

  return (
    <div className="status-bar">
      <span style={{ color: stateColor }}>
        ● {(status?.state ?? 'IDLE').toUpperCase()}
      </span>
      <span>STEP {status?.step ?? 0}</span>
      <span>STAGE {status?.stage ?? 0}</span>
      <span>CLUSTERS {graphStats.cluster_count ?? 0}</span>
      <span>NODES {graphStats.node_count ?? 0}</span>
      <span>EDGES {graphStats.edge_count ?? 0}</span>
      <span>LAYERS {graphStats.layer_count ?? 0}</span>
      {graphStats.dormant_count > 0 && (
        <span style={{ color: 'var(--text-dim)' }}>
          {graphStats.dormant_count} dormant
        </span>
      )}
      {status?.state === 'error' && (
        <span style={{ color: 'var(--negative)' }}>
          {status.error_message}
        </span>
      )}
    </div>
  )
}
```

```css
.status-bar {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 0 16px;
  height: 40px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
  font-size: 11px;
  letter-spacing: 0.05em;
  color: var(--text-dim);
}
```

---

---

# WebSocket connection (useWebSocket.ts)

---

```typescript
export function useWebSocket(url: string) {
  const applyDiff     = useGraphStore(s => s.applyDiff)
  const setSnapshot   = useGraphStore(s => s.setSnapshot)
  const setActivations = useGraphStore(s => s.setActivations)
  const addDialogue   = useDialogueStore(s => s.add)
  const triggerGrowth = useGraphStore(s => s.triggerGrowthEvents)

  useEffect(() => {
    let ws: WebSocket
    let reconnectTimer: ReturnType<typeof setTimeout>

    function connect() {
      ws = new WebSocket(url)

      ws.onopen = () => {
        console.log('WS connected')
      }

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data)

        if (msg.type === 'snapshot') {
          setSnapshot(msg)
        }

        if (msg.type === 'step') {
          applyDiff(msg.graph_diff, msg.positions_stale)
          setActivations(msg.activations)
          triggerGrowthEvents(msg.growth_events)
          addDialogue({
            step: msg.step,
            ...msg.dialogue,
            stage: msg.stage,
            timestamp: Date.now()
          })
        }
      }

      ws.onerror = () => {}

      ws.onclose = () => {
        // Reconnect after 2s
        reconnectTimer = setTimeout(connect, 2000)
      }
    }

    connect()

    return () => {
      clearTimeout(reconnectTimer)
      ws?.close()
    }
  }, [url])
}
```

---

# Graph store (graphStore.ts)

---

```typescript
import { create } from 'zustand'

interface GraphState {
  nodes: Node[]
  clusters: Cluster[]
  edges: Edge[]
  activations: Record<string, number>
  growthQueue: GrowthEvent[]

  setSnapshot: (snapshot: SnapshotMessage) => void
  applyDiff: (diff: GraphDiff, positionsStale: boolean) => void
  setActivations: (activations: Record<string, number>) => void
  triggerGrowthEvents: (events: GrowthEvent[]) => void
}

export const useGraphStore = create<GraphState>((set, get) => ({
  nodes: [],
  clusters: [],
  edges: [],
  activations: {},
  growthQueue: [],

  setSnapshot: (msg) => set({
    nodes: msg.nodes,
    clusters: msg.clusters,
    edges: msg.edges
  }),

  applyDiff: (diff, positionsStale) => set(state => {
    let nodes    = [...state.nodes]
    let clusters = [...state.clusters]
    let edges    = [...state.edges]

    // Add
    nodes    = [...nodes, ...diff.nodes_added]
    clusters = [...clusters, ...diff.clusters_added]
    edges    = [...edges, ...diff.edges_added]

    // Remove
    const removedNodeIds    = new Set(diff.nodes_removed.map(n => n.id))
    const removedClusterIds = new Set(diff.clusters_removed.map(c => c.id))
    const removedEdgeKeys   = new Set(
      diff.edges_removed.map(e => `${e.from}-${e.to}`)
    )
    nodes    = nodes.filter(n => !removedNodeIds.has(n.id))
    clusters = clusters.filter(c => !removedClusterIds.has(c.id))
    edges    = edges.filter(e => !removedEdgeKeys.has(`${e.from}-${e.to}`))

    // Update clusters
    const updatedClusters = new Map(diff.clusters_updated.map(c => [c.id, c]))
    clusters = clusters.map(c => updatedClusters.has(c.id)
      ? { ...c, ...updatedClusters.get(c.id) }
      : c
    )

    // Update edge strengths
    const updatedEdges = new Map(
      diff.edges_updated.map(e => [`${e.from}-${e.to}`, e])
    )
    edges = edges.map(e => {
      const key = `${e.from}-${e.to}`
      return updatedEdges.has(key)
        ? { ...e, strength: updatedEdges.get(key)!.strength }
        : e
    })

    // Update node positions only if not stale
    if (!positionsStale) {
      const updatedNodes = new Map(diff.nodes_added.map(n => [n.id, n]))
      nodes = nodes.map(n => updatedNodes.has(n.id)
        ? { ...n, pos: updatedNodes.get(n.id)!.pos }
        : n
      )
    }

    return { nodes, clusters, edges }
  }),

  setActivations: (activations) => set({ activations }),

  triggerGrowthEvents: (events) => set(state => ({
    growthQueue: [...state.growthQueue, ...events]
  }))
}))
```

---

# api.ts

---

```typescript
const BASE = 'http://localhost:8000'

export const api = {
  start:       () => fetch(`${BASE}/start`, { method: 'POST' }).then(r => r.json()),
  pause:       () => fetch(`${BASE}/pause`, { method: 'POST' }).then(r => r.json()),
  resume:      () => fetch(`${BASE}/resume`, { method: 'POST' }).then(r => r.json()),
  step:        () => fetch(`${BASE}/step`, { method: 'POST' }).then(r => r.json()),
  reset:       () => fetch(`${BASE}/reset`, { method: 'POST' }).then(r => r.json()),
  status:      () => fetch(`${BASE}/status`).then(r => r.json()),
  snapshot:    () => fetch(`${BASE}/snapshot`).then(r => r.json()),

  setStage: (stage: number) =>
    fetch(`${BASE}/stage`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stage })
    }).then(r => r.json()),

  setSpeed: (delay_ms: number) =>
    fetch(`${BASE}/speed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ delay_ms })
    }).then(r => r.json()),

  chat: (message: string) =>
    fetch(`${BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    }).then(r => r.json()),

  uploadImage: (file: File, label?: string) => {
    const form = new FormData()
    form.append('file', file)
    if (label) form.append('label', label)
    return fetch(`${BASE}/image`, { method: 'POST', body: form }).then(r => r.json())
  }
}
```

---

# package.json

```json
{
  "name": "dev-ai-frontend",
  "version": "0.0.1",
  "private": true,
  "scripts": {
    "dev":   "vite --host",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react":              "^18.2.0",
    "react-dom":          "^18.2.0",
    "@react-three/fiber": "^8.15.0",
    "@react-three/drei":  "^9.88.0",
    "three":              "^0.157.0",
    "zustand":            "^4.4.0"
  },
  "devDependencies": {
    "@types/react":       "^18.2.0",
    "@types/three":       "^0.157.0",
    "typescript":         "^5.2.0",
    "vite":               "^5.0.0",
    "@vitejs/plugin-react": "^4.0.0",
    "tailwindcss":        "^3.3.0",
    "autoprefixer":       "^10.4.0"
  }
}
```

---

# vite.config.ts

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true
  }
})
```

---

## Hard parts

**Three.js r128 is pinned for this project.**
The system prompt specifies r128.
`@react-three/fiber` v8 targets Three.js r139+ — version mismatch.
Use `@react-three/fiber@7` with Three.js r128, or upgrade Three.js
to the version r3f v8 expects. Check `peerDependencies` in r3f's
package.json and pin Three.js to match.
Mismatched versions cause silent render failures (blank canvas,
no error in console).

**`OrbitControls` disables the auto-orbit when the user touches the canvas.**
Set a timeout (5s of inactivity) before re-enabling autoRotate.
The `@react-three/drei` `OrbitControls` component accepts `autoRotate`
as a prop — toggle it via state after the inactivity timeout.

**UMAP position jumps when graph structure changes.**
When a BUD or INSERT happens, UMAP reruns and all positions shift.
The frontend receives a `snapshot` message (not a `step` message)
after UMAP reruns. On `snapshot`, do a hard position reset —
teleport all nodes to new positions, then tween for 500ms.
Don't try to continuously tween through the jump — it looks wrong.

**WebSocket reconnect and state sync.**
When the browser tab goes to sleep and wakes up, the WebSocket
disconnects and reconnects. On reconnect, the backend sends a full
snapshot. The `setSnapshot` action in `graphStore` does a full
state replace — this is correct, not a diff. The frontend
re-renders from the snapshot as if it just loaded.

**Dialogue store memory.**
At max speed the loop generates ~3 dialogue entries per second.
Over an hour that's ~10,800 entries. With 200-entry cap the store
stays bounded, but the DOM renders 200 items in the feed.
Virtualize the list if scrolling performance degrades (react-virtual
or @tanstack/react-virtual).

**`line` element in Three.js r128.**
`<line>` in JSX refers to the SVG `<line>` element, not Three.js Line.
Use `<primitive object={new THREE.Line(geometry, material)} />`
or extend the r3f catalog with `extend({ Line: THREE.Line })`.
The spec uses `<line>` for clarity — the actual implementation
must use the Three.js primitive correctly.

---

## M1-specific notes

Vite dev server on M1 is fast — cold start < 500ms.
Hot module replacement is near-instant.
No M1-specific configuration required for the frontend.

Three.js uses the browser's WebGL context — on M1 this is Metal-backed.
The scene is intentionally simple (small spheres, basic lines) so
GPU load is negligible even at 500+ nodes.

If the node count grows above ~2000, Three.js instanced meshes
(`InstancedMesh`) will be necessary for performance. At 2000 nodes
with individual `Mesh` objects, the scene may drop below 60fps.
The initial design uses individual meshes — switch to `InstancedMesh`
if performance degrades. This is an optimization, not a day-one concern.
