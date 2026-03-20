import { useRef, useMemo, useState, useCallback, useEffect } from 'react'
import { Canvas, useFrame, useThree, extend } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { useGraphStore, GraphNode, GraphEdge, GraphCluster } from '../store/graphStore'
import { clusterColor } from '../lib/colors'
import { ANIMATION } from '../lib/constants'
import { clusterCentroid } from '../hooks/useGraphState'
import { useClusterLabels } from '../hooks/useClusterLabels'
import { useClusterTree, TreeNode, type TreeStatus } from '../hooks/useClusterTree'
import { api } from '../lib/api'

extend({ Line_: THREE.Line })

// ── Shared geometry — single GPU upload, reused by all spheres ──
const SHARED_SPHERE = new THREE.SphereGeometry(1, 6, 6)
const SHARED_SPHERE_LOWPOLY = new THREE.SphereGeometry(1, 4, 4)

// ── Edge culling helpers ──
const MAX_UMAP_EDGES = 500
const MAX_LIVE_EDGES = 300

function topEdges(edges: GraphEdge[], max: number): GraphEdge[] {
  if (edges.length <= max) return edges
  // Find p75 threshold, then take top N
  const sorted = [...edges].sort((a, b) => b.strength - a.strength)
  return sorted.slice(0, max)
}

// ── Types ──

type VizMode = 'umap' | 'live' | 'tree'

interface HoverInfo {
  clusterId: string
  clusterType: string
  screenX: number
  screenY: number
}

// ── Main component ──

export function LatentSpace() {
  const [autoRotate, setAutoRotate] = useState(true)
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>()
  const [hover, setHover] = useState<HoverInfo | null>(null)
  const [mode, setMode] = useState<VizMode>('umap')
  const clusterLabels = useClusterLabels()
  const { tree: treeData, status: treeStatus } = useClusterTree()

  const handleInteraction = useCallback(() => {
    setAutoRotate(false)
    clearTimeout(timeoutRef.current)
    timeoutRef.current = setTimeout(() => {
      setAutoRotate(true)
    }, ANIMATION.AUTO_ROTATE_RESUME_MS)
  }, [])

  useEffect(() => {
    return () => clearTimeout(timeoutRef.current)
  }, [])

  const [showLegend, setShowLegend] = useState(true)

  return (
    <div className="latent-space-panel" onPointerDown={handleInteraction}>
      <ModeToggle mode={mode} setMode={setMode} />

      <Canvas
        camera={{ position: [0, 0, 4], fov: 50 }}
        gl={{ antialias: true, alpha: false }}
        style={{ background: '#080808' }}
      >
        <ambientLight intensity={0.1} />
        <pointLight position={[2, 2, 2]} intensity={0.3} color="#4effa0" />

        <GridPlane />

        {mode === 'tree' && treeStatus === 'ok' ? (
          <TreeRenderer treeData={treeData} onHover={setHover} />
        ) : mode === 'live' ? (
          <LiveRenderer onHover={setHover} />
        ) : mode === 'umap' ? (
          <UmapRenderer onHover={setHover} />
        ) : null}

        <OrbitControls
          enablePan={false}
          enableZoom={true}
          autoRotate={autoRotate}
          autoRotateSpeed={0.3}
          makeDefault
        />
      </Canvas>

      {hover && (
        <ClusterTooltip
          hover={hover}
          labels={clusterLabels[hover.clusterId] ?? []}
        />
      )}

      {mode === 'tree' && treeStatus !== 'ok' && (
        <div className="tree-fallback">
          {treeStatus === 'loading' ? 'Loading tree data...' : 'Tree data unavailable'}
        </div>
      )}

      {showLegend && <Legend onClose={() => setShowLegend(false)} />}
      {!showLegend && (
        <button className="legend-toggle" onClick={() => setShowLegend(true)}>?</button>
      )}
    </div>
  )
}

// ── Mode Toggle ──

function ModeToggle({ mode, setMode }: { mode: VizMode; setMode: (m: VizMode) => void }) {
  return (
    <div className="viz-mode-toggle">
      {(['umap', 'live', 'tree'] as VizMode[]).map(m => (
        <button
          key={m}
          className={`viz-mode-btn ${mode === m ? 'active' : ''}`}
          onClick={() => setMode(m)}
        >
          {m.toUpperCase()}
        </button>
      ))}
    </div>
  )
}

// ── Tooltip ──

function ClusterTooltip({ hover, labels }: { hover: HoverInfo; labels: string[] }) {
  return (
    <div
      className="cluster-tooltip"
      style={{ left: hover.screenX + 14, top: hover.screenY - 10 }}
    >
      <div className="cluster-tooltip-id">
        {hover.clusterId}
        <span className="cluster-tooltip-type">{hover.clusterType}</span>
      </div>
      {labels.length > 0 ? (
        <div className="cluster-tooltip-labels">
          {labels.map(w => <span key={w} className="cluster-tooltip-tag">{w}</span>)}
        </div>
      ) : (
        <div className="cluster-tooltip-empty">no labels yet</div>
      )}
    </div>
  )
}

// ── Legend ──

const LEGEND_COLORS = [
  { label: 'Integration', color: '#3b7fff' },
  { label: 'Transformation', color: '#9b59b6' },
  { label: 'Arbitration', color: '#e67e22' },
  { label: 'Routing', color: '#1abc9c' },
  { label: 'Dormant', color: '#2a2a2a' },
]

function Legend({ onClose }: { onClose: () => void }) {
  return (
    <div className="graph-legend">
      <div className="legend-header">
        <span>LEGEND</span>
        <button className="legend-close" onClick={onClose}>&times;</button>
      </div>
      <div className="legend-section">
        <div className="legend-title">Cluster types</div>
        {LEGEND_COLORS.map(({ label, color }) => (
          <div key={label} className="legend-row">
            <span className="legend-swatch" style={{ background: color }} />
            <span>{label}</span>
          </div>
        ))}
      </div>
      <div className="legend-section">
        <div className="legend-title">Nodes</div>
        <div className="legend-desc">Size = activation strength</div>
        <div className="legend-desc">Pulse = currently active</div>
      </div>
      <div className="legend-section">
        <div className="legend-title">Edges</div>
        <div className="legend-row">
          <span className="legend-line" style={{ opacity: 1 }} />
          <span>Strong connection</span>
        </div>
        <div className="legend-row">
          <span className="legend-line" style={{ opacity: 0.25 }} />
          <span>Weak connection</span>
        </div>
      </div>
    </div>
  )
}

function GridPlane() {
  return <gridHelper args={[10, 20, '#1a1a1a', '#111111']} position={[0, -1.5, 0]} />
}

// ═══════════════════════════════════════════════════════════════════════
//  Force simulation (singleton, outside React)
// ═══════════════════════════════════════════════════════════════════════

interface SimBody { x: number; y: number; z: number; vx: number; vy: number; vz: number }
interface CofiringSpring { a: string; b: string; strength: number; targetStrength: number }

class ForceSimulation {
  bodies: Map<string, SimBody> = new Map()
  springs: Map<string, CofiringSpring> = new Map()
  damping = 0.85
  maxVelocity = 2
  repulsionStrength = 0.15
  attractionStrength = 0.8
  springLerpSpeed = 0.0033
  private _hasFetched = false

  syncClusters(clusterIds: string[]) {
    const idSet = new Set(clusterIds)
    for (const cid of this.bodies.keys()) { if (!idSet.has(cid)) this.bodies.delete(cid) }
    for (const cid of clusterIds) {
      if (!this.bodies.has(cid)) {
        const partner = this._bestPartner(cid)
        if (partner) {
          const p = this.bodies.get(partner)!
          this.bodies.set(cid, { x: p.x + (Math.random() - 0.5) * 0.3, y: p.y + (Math.random() - 0.5) * 0.3, z: p.z + (Math.random() - 0.5) * 0.3, vx: 0, vy: 0, vz: 0 })
        } else {
          const a = Math.random() * Math.PI * 2, r = 0.5 + Math.random() * 0.5
          this.bodies.set(cid, { x: Math.cos(a) * r, y: (Math.random() - 0.5) * 0.4, z: Math.sin(a) * r, vx: 0, vy: 0, vz: 0 })
        }
      }
    }
  }

  async fetchCofiring() {
    try {
      const data = await api.clusterCofiring()
      const pairs = data.pairs ?? []
      const newKeys = new Set<string>()
      for (const p of pairs) {
        const key = p.a < p.b ? `${p.a}|${p.b}` : `${p.b}|${p.a}`
        newKeys.add(key)
        const existing = this.springs.get(key)
        if (existing) { existing.targetStrength = p.strength }
        else { this.springs.set(key, { a: p.a < p.b ? p.a : p.b, b: p.a < p.b ? p.b : p.a, strength: this._hasFetched ? 0 : p.strength, targetStrength: p.strength }) }
      }
      for (const key of this.springs.keys()) { if (!newKeys.has(key)) this.springs.delete(key) }
      this._hasFetched = true
    } catch {}
  }

  step(dt: number) {
    const ids = [...this.bodies.keys()]
    const n = ids.length
    if (n === 0) return
    const boundaryR = Math.max(1.5, Math.sqrt(n) * 0.5)

    for (const spring of this.springs.values()) {
      spring.strength += (spring.targetStrength - spring.strength) * this.springLerpSpeed
    }

    // Flat arrays for force accumulation — avoids Map overhead
    const fx = new Float32Array(n)
    const fy = new Float32Array(n)
    const fz = new Float32Array(n)

    // Pre-extract positions into flat arrays
    const px = new Float32Array(n)
    const py = new Float32Array(n)
    const pz = new Float32Array(n)
    for (let i = 0; i < n; i++) {
      const b = this.bodies.get(ids[i])!
      px[i] = b.x; py[i] = b.y; pz[i] = b.z
    }

    // Repulsion: nearby pairs only (skip > 2.0 apart)
    const CUTOFF_SQ = 4.0
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = px[i] - px[j]
        const dy = py[i] - py[j]
        const dz = pz[i] - pz[j]
        const distSq = dx * dx + dy * dy + dz * dz
        if (distSq > CUTOFF_SQ) continue
        const distSqSafe = distSq + 0.001
        const dist = Math.sqrt(distSqSafe)
        const force = this.repulsionStrength / distSqSafe
        const ffx = (dx / dist) * force
        const ffy = (dy / dist) * force
        const ffz = (dz / dist) * force
        fx[i] += ffx; fy[i] += ffy; fz[i] += ffz
        fx[j] -= ffx; fy[j] -= ffy; fz[j] -= ffz
      }
    }

    // Attraction: co-firing springs (use index lookup)
    const idIndex = new Map<string, number>()
    for (let i = 0; i < n; i++) idIndex.set(ids[i], i)

    for (const spring of this.springs.values()) {
      const ai = idIndex.get(spring.a)
      const bi = idIndex.get(spring.b)
      if (ai === undefined || bi === undefined) continue
      const dx = px[bi] - px[ai]
      const dy = py[bi] - py[ai]
      const dz = pz[bi] - pz[ai]
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz + 0.001)
      const force = this.attractionStrength * spring.strength * dist * 0.1
      const ffx = (dx / dist) * force
      const ffy = (dy / dist) * force
      const ffz = (dz / dist) * force
      fx[ai] += ffx; fy[ai] += ffy; fz[ai] += ffz
      fx[bi] -= ffx; fy[bi] -= ffy; fz[bi] -= ffz
    }

    // Boundary + integrate
    for (let i = 0; i < n; i++) {
      const body = this.bodies.get(ids[i])!
      const dist = Math.sqrt(px[i] * px[i] + py[i] * py[i] + pz[i] * pz[i])
      if (dist > boundaryR) {
        const push = (dist - boundaryR) * 0.3
        fx[i] -= (px[i] / dist) * push
        fy[i] -= (py[i] / dist) * push
        fz[i] -= (pz[i] / dist) * push
      }
      body.vx = (body.vx + fx[i] * dt) * this.damping
      body.vy = (body.vy + fy[i] * dt) * this.damping
      body.vz = (body.vz + fz[i] * dt) * this.damping
      const speed = Math.sqrt(body.vx * body.vx + body.vy * body.vy + body.vz * body.vz)
      if (speed > this.maxVelocity) { const s = this.maxVelocity / speed; body.vx *= s; body.vy *= s; body.vz *= s }
      body.x += body.vx * dt; body.y += body.vy * dt; body.z += body.vz * dt
    }
  }

  getPosition(id: string): [number, number, number] {
    const b = this.bodies.get(id)
    return b ? [b.x, b.y, b.z] : [0, 0, 0]
  }

  private _bestPartner(cid: string): string | null {
    let best: string | null = null, bestStr = 0
    for (const spring of this.springs.values()) {
      if (spring.a === cid && spring.strength > bestStr && this.bodies.has(spring.b)) { best = spring.b; bestStr = spring.strength }
      if (spring.b === cid && spring.strength > bestStr && this.bodies.has(spring.a)) { best = spring.a; bestStr = spring.strength }
    }
    return best
  }
}

const _sim = new ForceSimulation()

// ═══════════════════════════════════════════════════════════════════════
//  LIVE mode — force-directed, capped edges
// ═══════════════════════════════════════════════════════════════════════

function LiveRenderer({ onHover }: { onHover: (info: HoverInfo | null) => void }) {
  const clusters = useGraphStore(s => s.clusters)
  const nodes = useGraphStore(s => s.nodes)
  const edges = useGraphStore(s => s.edges)
  const activations = useGraphStore(s => s.activations)

  const activeIds = useMemo(() => clusters.filter(c => !c.dormant).map(c => c.id), [clusters])
  useEffect(() => { _sim.syncClusters(activeIds) }, [activeIds])
  useEffect(() => {
    _sim.fetchCofiring()
    const timer = setInterval(() => _sim.fetchCofiring(), 60_000)
    return () => clearInterval(timer)
  }, [])

  const positionsRef = useRef<Record<string, [number, number, number]>>({})

  // Top 300 strongest edges only
  const visibleEdges = useMemo(() => topEdges(edges, MAX_LIVE_EDGES), [edges])

  useFrame((_, delta) => {
    _sim.step(Math.min(delta, 1 / 30))
    for (const cid of activeIds) positionsRef.current[cid] = _sim.getPosition(cid)
  })

  return (
    <group>
      {visibleEdges.map(edge => (
        <LiveEdge key={`${edge.from}-${edge.to}`} edge={edge} positionsRef={positionsRef} />
      ))}
      {clusters.filter(c => !c.dormant).map(cluster => (
        <LiveSphere key={cluster.id} cluster={cluster} activation={activations[cluster.id] ?? 0} positionsRef={positionsRef} onHover={onHover} />
      ))}
    </group>
  )
}

function LiveSphere({ cluster, activation, positionsRef, onHover }: {
  cluster: GraphCluster; activation: number
  positionsRef: React.MutableRefObject<Record<string, [number, number, number]>>
  onHover: (info: HoverInfo | null) => void
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const matRef = useRef<THREE.MeshStandardMaterial>(null)
  const color = clusterColor(cluster.cluster_type)
  const baseScale = 0.12 + activation * 0.08
  const pulsePhase = useRef(Math.random() * Math.PI * 2)

  useFrame((_, delta) => {
    if (!meshRef.current) return
    const pos = positionsRef.current[cluster.id]
    if (pos) meshRef.current.position.set(pos[0], pos[1], pos[2])
    const pulseSpeed = activation > 0.1 ? 6 : 1.5
    pulsePhase.current += delta * pulseSpeed
    const pulse = 1 + Math.sin(pulsePhase.current) * 0.15 * (activation > 0.1 ? 1 : 0.3)
    const s = meshRef.current.scale.x
    meshRef.current.scale.setScalar(s + (baseScale * pulse - s) * Math.min(delta * ANIMATION.PULSE_LERP_SPEED, 1))
    if (matRef.current) {
      const op = Math.max(0.25, 0.3 + activation * 0.7)
      matRef.current.opacity += (op - matRef.current.opacity) * Math.min(delta * 4, 1)
    }
  })

  const onEnter = useCallback((e: any) => {
    e.stopPropagation()
    const pos = positionsRef.current[cluster.id] ?? [0, 0, 0]
    const vec = new THREE.Vector3(...pos)
    const canvas = e.target?.ownerDocument?.querySelector('canvas') as HTMLCanvasElement | null
    if (canvas) { vec.project(e.camera); const r = canvas.getBoundingClientRect(); onHover({ clusterId: cluster.id, clusterType: cluster.cluster_type, screenX: ((vec.x + 1) / 2) * r.width + r.left, screenY: ((-vec.y + 1) / 2) * r.height + r.top }) }
  }, [cluster.id, cluster.cluster_type, positionsRef, onHover])
  const onLeave = useCallback((e: any) => { e.stopPropagation(); onHover(null) }, [onHover])

  return (
    <mesh ref={meshRef} geometry={SHARED_SPHERE} position={[0, 0, 0]} scale={baseScale} onPointerEnter={onEnter} onPointerLeave={onLeave}>
      <meshStandardMaterial ref={matRef} color={color} emissive={color} emissiveIntensity={0.2 + activation * 1.2} roughness={0.6} metalness={0.2} transparent opacity={0.5} />
    </mesh>
  )
}

function LiveEdge({ edge, positionsRef }: { edge: GraphEdge; positionsRef: React.MutableRefObject<Record<string, [number, number, number]>> }) {
  const opacityRef = useRef(0)
  const targetOpacity = edgeOpacity(edge.strength)
  const geometry = useMemo(() => { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3)); return g }, [])
  const material = useMemo(() => new THREE.LineBasicMaterial({ color: '#4effa0', transparent: true, opacity: 0 }), [])
  useFrame((_, delta) => {
    const a = positionsRef.current[edge.from] ?? [0, 0, 0], b = positionsRef.current[edge.to] ?? [0, 0, 0]
    const p = geometry.getAttribute('position') as THREE.BufferAttribute
    p.setXYZ(0, a[0], a[1], a[2]); p.setXYZ(1, b[0], b[1], b[2]); p.needsUpdate = true
    opacityRef.current += (Math.min(targetOpacity * 1.5, 1) - opacityRef.current) * Math.min(delta * 3, 1)
    material.opacity = opacityRef.current
  })
  return <primitive object={new THREE.Line(geometry, material)} />
}

// ═══════════════════════════════════════════════════════════════════════
//  UMAP mode — top 500 edges, shared geometry
// ═══════════════════════════════════════════════════════════════════════

function UmapRenderer({ onHover }: { onHover: (info: HoverInfo | null) => void }) {
  const clusters = useGraphStore(s => s.clusters)
  const nodes = useGraphStore(s => s.nodes)
  const edges = useGraphStore(s => s.edges)
  const activations = useGraphStore(s => s.activations)

  // Top 500 strongest edges only
  const visibleEdges = useMemo(() => topEdges(edges, MAX_UMAP_EDGES), [edges])

  return (
    <group>
      {visibleEdges.map(edge => (
        <UmapEdge key={`${edge.from}-${edge.to}`} edge={edge} fromPos={clusterCentroid(nodes, edge.from)} toPos={clusterCentroid(nodes, edge.to)} />
      ))}
      {clusters.map(cluster => (
        <UmapClusterGroup key={cluster.id} cluster={cluster} nodes={nodes.filter(n => n.cluster === cluster.id)} activation={activations[cluster.id] ?? 0} onHover={onHover} />
      ))}
    </group>
  )
}

function UmapClusterGroup({ cluster, nodes, activation, onHover }: {
  cluster: GraphCluster; nodes: GraphNode[]; activation: number; onHover: (info: HoverInfo | null) => void
}) {
  const color = clusterColor(cluster.cluster_type)
  return (
    <group>
      {nodes.map(node => (
        <UmapNodeSphere key={node.id} node={node} clusterId={cluster.id} clusterType={cluster.cluster_type} color={color} activation={activation} onHover={onHover} />
      ))}
    </group>
  )
}

function UmapNodeSphere({ node, clusterId, clusterType, color, activation, onHover }: {
  node: GraphNode; clusterId: string; clusterType: string; color: string; activation: number; onHover: (info: HoverInfo | null) => void
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const matRef = useRef<THREE.MeshStandardMaterial>(null)
  const baseScale = 0.12 + Math.abs(node.activation_mean) * 0.08
  const dormantTarget = node.alive ? 0.9 : 0.15

  useFrame((_, delta) => {
    if (!meshRef.current) return
    const target = activation > 0.1 ? baseScale * 1.5 : baseScale
    const s = meshRef.current.scale.x
    meshRef.current.scale.setScalar(s + (target - s) * Math.min(delta * ANIMATION.PULSE_LERP_SPEED, 1))
    if (matRef.current) matRef.current.opacity += (dormantTarget - matRef.current.opacity) * Math.min(delta * 4, 1)
  })

  const onEnter = useCallback((e: any) => {
    e.stopPropagation()
    const pos = node.pos ?? [0, 0, 0]
    const vec = new THREE.Vector3(...pos)
    const canvas = e.target?.ownerDocument?.querySelector('canvas') as HTMLCanvasElement | null
    if (canvas) { vec.project(e.camera); const r = canvas.getBoundingClientRect(); onHover({ clusterId, clusterType, screenX: ((vec.x + 1) / 2) * r.width + r.left, screenY: ((-vec.y + 1) / 2) * r.height + r.top }) }
  }, [node.pos, clusterId, clusterType, onHover])
  const onLeave = useCallback((e: any) => { e.stopPropagation(); onHover(null) }, [onHover])

  return (
    <mesh ref={meshRef} geometry={SHARED_SPHERE} position={node.pos ?? [0, 0, 0]} scale={baseScale} onPointerEnter={onEnter} onPointerLeave={onLeave}>
      <meshStandardMaterial ref={matRef} color={node.alive ? color : '#2a2a2a'} emissive={node.alive ? color : '#2a2a2a'} emissiveIntensity={0.4 + activation * 0.6} roughness={0.6} metalness={0.2} transparent opacity={dormantTarget} />
    </mesh>
  )
}

function UmapEdge({ edge, fromPos, toPos }: { edge: GraphEdge; fromPos: [number, number, number]; toPos: [number, number, number] }) {
  const opacityRef = useRef(0)
  const targetOpacity = edgeOpacity(edge.strength)
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([fromPos[0], fromPos[1], fromPos[2], toPos[0], toPos[1], toPos[2]]), 3))
    return g
  }, [fromPos[0], fromPos[1], fromPos[2], toPos[0], toPos[1], toPos[2]])
  const material = useMemo(() => new THREE.LineBasicMaterial({ color: '#4effa0', transparent: true, opacity: 0 }), [])
  useFrame((_, delta) => { opacityRef.current += (targetOpacity - opacityRef.current) * Math.min(delta * 3, 1); material.opacity = opacityRef.current })
  return <primitive object={new THREE.Line(geometry, material)} />
}

function edgeOpacity(strength: number): number {
  if (strength < 0.2) return 0.1
  if (strength < 0.5) return 0.3
  if (strength < 0.8) return 0.6
  return 1.0
}

// ═══════════════════════════════════════════════════════════════════════
//  TREE mode — stacked horizontal planes, layout computed once
// ═══════════════════════════════════════════════════════════════════════

const PLANE_SPACING = 2.0
const TREE_REPULSION_ITERS = 60
const TREE_REPULSION_STRENGTH = 0.08

function TreeRenderer({ treeData, onHover }: {
  treeData: { nodes: TreeNode[]; edges: { source: string; target: string }[]; max_depth: number }
  onHover: (info: HoverInfo | null) => void
}) {
  const activations = useGraphStore(s => s.activations)

  const childCount = useMemo(() => {
    const counts = new Map<string, number>()
    for (const n of treeData.nodes) counts.set(n.id, 0)
    for (const n of treeData.nodes) {
      if (n.parent && counts.has(n.parent)) counts.set(n.parent, (counts.get(n.parent) ?? 0) + 1)
    }
    return counts
  }, [treeData.nodes])

  // Layout computed ONCE when treeData changes — not every frame
  const layout = useMemo(() => {
    if (treeData.nodes.length === 0) return new Map<string, [number, number, number]>()
    const positions = new Map<string, [number, number, number]>()
    const posById = new Map<string, { x: number; y: number }>()
    const byDepth = new Map<number, TreeNode[]>()
    for (const n of treeData.nodes) {
      const list = byDepth.get(n.depth) ?? []; list.push(n); byDepth.set(n.depth, list)
    }
    const depths = [...byDepth.keys()].sort((a, b) => a - b)
    for (const depth of depths) {
      const nodesAtDepth = byDepth.get(depth)!
      const count = nodesAtDepth.length
      const xy: { x: number; y: number }[] = nodesAtDepth.map((node, i) => {
        if (node.parent && posById.has(node.parent)) {
          const pp = posById.get(node.parent)!
          return { x: pp.x + (Math.random() - 0.5) * 0.5, y: pp.y + (Math.random() - 0.5) * 0.5 }
        }
        const spread = Math.max(count - 1, 1)
        return { x: (i / spread - 0.5) * Math.min(count * 0.4, 3), y: (Math.random() - 0.5) * 0.3 }
      })
      for (let iter = 0; iter < TREE_REPULSION_ITERS; iter++) {
        const temp = 1 - iter / TREE_REPULSION_ITERS
        for (let i = 0; i < count; i++) {
          let ffx = 0, ffy = 0
          for (let j = 0; j < count; j++) {
            if (i === j) continue
            const dx = xy[i].x - xy[j].x, dy = xy[i].y - xy[j].y
            const distSq = dx * dx + dy * dy + 0.001
            if (distSq < 2.25) { // 1.5² cutoff
              const dist = Math.sqrt(distSq)
              ffx += (dx / dist) * TREE_REPULSION_STRENGTH / distSq
              ffy += (dy / dist) * TREE_REPULSION_STRENGTH / distSq
            }
          }
          xy[i].x += ffx * temp; xy[i].y += ffy * temp
        }
      }
      if (count > 0) {
        const cx = xy.reduce((s, p) => s + p.x, 0) / count
        const cy = xy.reduce((s, p) => s + p.y, 0) / count
        for (const p of xy) { p.x -= cx; p.y -= cy }
      }
      const z = depth * PLANE_SPACING
      for (let i = 0; i < count; i++) {
        posById.set(nodesAtDepth[i].id, xy[i])
        positions.set(nodesAtDepth[i].id, [xy[i].x, xy[i].y, z])
      }
    }
    return positions
  }, [treeData])

  const maxZ = treeData.max_depth * PLANE_SPACING
  const camTarget = useMemo<[number, number, number]>(() => [0, 0, maxZ / 2], [maxZ])

  return (
    <group>
      <TreeCamera target={camTarget} distance={Math.max(4, maxZ * 0.8 + 2)} />
      {Array.from({ length: treeData.max_depth + 1 }, (_, d) => (
        <gridHelper key={`plane-${d}`} args={[6, 8, '#111111', '#0c0c0c']} position={[0, 0, d * PLANE_SPACING]} rotation={[Math.PI / 2, 0, 0]} />
      ))}
      {treeData.edges.map(edge => {
        const from = layout.get(edge.source) ?? [0, 0, 0], to = layout.get(edge.target) ?? [0, 0, 0]
        return <TreeEdgeLine key={`${edge.source}-${edge.target}`} from={from} to={to} />
      })}
      {treeData.nodes.map(node => {
        const pos = layout.get(node.id) ?? [0, 0, 0]
        return <TreeSphere key={node.id} node={node} position={pos} activation={activations[node.id] ?? 0} childCount={childCount.get(node.id) ?? 0} onHover={onHover} />
      })}
    </group>
  )
}

function TreeCamera({ target, distance }: { target: [number, number, number]; distance: number }) {
  const { camera } = useThree()
  useFrame((_, delta) => {
    const tx = 3, ty = distance * 0.5, tz = target[2] - distance * 0.3
    const speed = Math.min(delta * 2, 1)
    camera.position.x += (tx - camera.position.x) * speed
    camera.position.y += (ty - camera.position.y) * speed
    camera.position.z += (tz - camera.position.z) * speed
    camera.lookAt(new THREE.Vector3(target[0], target[1], target[2]))
  })
  return null
}

function TreeSphere({ node, position, activation, childCount, onHover }: {
  node: TreeNode; position: [number, number, number]; activation: number; childCount: number; onHover: (info: HoverInfo | null) => void
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const matRef = useRef<THREE.MeshStandardMaterial>(null)
  const isPhantom = node.phantom
  const isActive = !node.dormant && !isPhantom
  const baseScale = isPhantom ? 0.05 : 0.08 + Math.min(childCount, 8) * 0.03 + activation * 0.04
  const color = node.cluster_type ? clusterColor(node.cluster_type) : (isPhantom ? '#333333' : '#888888')
  const geo = isPhantom ? SHARED_SPHERE_LOWPOLY : SHARED_SPHERE

  useFrame((_, delta) => {
    if (!meshRef.current) return
    const cur = meshRef.current.position, speed = Math.min(delta * 3, 1)
    cur.x += (position[0] - cur.x) * speed; cur.y += (position[1] - cur.y) * speed; cur.z += (position[2] - cur.z) * speed
    const pt = isActive ? baseScale * (1 + activation * 0.3) : baseScale * 0.6
    const s = meshRef.current.scale.x
    meshRef.current.scale.setScalar(s + (pt - s) * Math.min(delta * 6, 1))
    if (matRef.current) {
      const op = isPhantom ? 0.12 : (isActive ? 0.5 + activation * 0.5 : 0.2)
      matRef.current.opacity += (op - matRef.current.opacity) * Math.min(delta * 4, 1)
    }
  })

  const onEnter = useCallback((e: any) => {
    e.stopPropagation()
    const vec = new THREE.Vector3(...position)
    const canvas = e.target?.ownerDocument?.querySelector('canvas') as HTMLCanvasElement | null
    if (canvas) { vec.project(e.camera); const r = canvas.getBoundingClientRect(); onHover({ clusterId: node.id, clusterType: node.cluster_type ?? 'unknown', screenX: ((vec.x + 1) / 2) * r.width + r.left, screenY: ((-vec.y + 1) / 2) * r.height + r.top }) }
  }, [position, node, onHover])
  const onLeave = useCallback((e: any) => { e.stopPropagation(); onHover(null) }, [onHover])

  return (
    <mesh ref={meshRef} geometry={geo} position={position} scale={baseScale} onPointerEnter={onEnter} onPointerLeave={onLeave}>
      <meshStandardMaterial ref={matRef} color={color} emissive={isActive ? color : '#111111'} emissiveIntensity={isActive ? 0.3 + activation * 0.9 : 0.05} roughness={0.6} metalness={0.2} transparent opacity={isPhantom ? 0.12 : (isActive ? 0.8 : 0.2)} />
    </mesh>
  )
}

function TreeEdgeLine({ from, to }: { from: [number, number, number]; to: [number, number, number] }) {
  const opacityRef = useRef(0)
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([from[0], from[1], from[2], to[0], to[1], to[2]]), 3))
    return g
  }, [from[0], from[1], from[2], to[0], to[1], to[2]])
  const material = useMemo(() => new THREE.LineBasicMaterial({ color: '#2a4a3a', transparent: true, opacity: 0 }), [])
  useFrame((_, delta) => { opacityRef.current += (0.35 - opacityRef.current) * Math.min(delta * 3, 1); material.opacity = opacityRef.current })
  return <primitive object={new THREE.Line(geometry, material)} />
}
