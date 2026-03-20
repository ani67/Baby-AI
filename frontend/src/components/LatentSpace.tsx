import { useRef, useMemo, useState, useCallback, useEffect } from 'react'
import { Canvas, useFrame, extend } from '@react-three/fiber'
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
      style={{
        left: hover.screenX + 14,
        top: hover.screenY - 10,
      }}
    >
      <div className="cluster-tooltip-id">
        {hover.clusterId}
        <span className="cluster-tooltip-type">{hover.clusterType}</span>
      </div>
      {labels.length > 0 ? (
        <div className="cluster-tooltip-labels">
          {labels.map(w => (
            <span key={w} className="cluster-tooltip-tag">{w}</span>
          ))}
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
        <div className="legend-title">Layout</div>
        <div className="legend-desc">Y axis = layer depth (bottom → top)</div>
        <div className="legend-desc">X/Z = clusters spread per layer</div>
      </div>
      <div className="legend-section">
        <div className="legend-title">Nodes</div>
        <div className="legend-desc">Size = mean activation strength</div>
        <div className="legend-desc">Pulse = currently active</div>
        <div className="legend-desc">Dim = dead / pruned node</div>
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
  return (
    <gridHelper
      args={[10, 20, '#1a1a1a', '#111111']}
      position={[0, -1.5, 0]}
    />
  )
}

// ═══════════════════════════════════════════════════════════════════════
//  Force-directed simulation state (shared mutable, lives outside React)
// ═══════════════════════════════════════════════════════════════════════

interface SimBody {
  x: number; y: number; z: number
  vx: number; vy: number; vz: number
}

interface CofiringSpring {
  a: string
  b: string
  strength: number        // current (lerped)
  targetStrength: number  // from latest fetch
}

class ForceSimulation {
  bodies: Map<string, SimBody> = new Map()
  springs: Map<string, CofiringSpring> = new Map()  // key = "a|b" canonical
  damping = 0.85
  maxVelocity = 2
  repulsionStrength = 0.15
  attractionStrength = 0.8
  springLerpSpeed = 0.0033  // ~5 seconds to converge at 60fps (1 - 0.0033^300)

  private _fetchTimer = 0
  private _hasFetched = false

  /** Ensure a body exists for each cluster, remove stale ones. */
  syncClusters(clusterIds: string[]) {
    const idSet = new Set(clusterIds)
    // Remove bodies for clusters no longer present
    for (const cid of this.bodies.keys()) {
      if (!idSet.has(cid)) this.bodies.delete(cid)
    }
    // Add new clusters
    for (const cid of clusterIds) {
      if (!this.bodies.has(cid)) {
        // Place near best co-firing partner, or random
        const partner = this._bestPartner(cid)
        if (partner) {
          const p = this.bodies.get(partner)!
          this.bodies.set(cid, {
            x: p.x + (Math.random() - 0.5) * 0.3,
            y: p.y + (Math.random() - 0.5) * 0.3,
            z: p.z + (Math.random() - 0.5) * 0.3,
            vx: 0, vy: 0, vz: 0,
          })
        } else {
          const angle = Math.random() * Math.PI * 2
          const r = 0.5 + Math.random() * 0.5
          this.bodies.set(cid, {
            x: Math.cos(angle) * r,
            y: (Math.random() - 0.5) * 0.4,
            z: Math.sin(angle) * r,
            vx: 0, vy: 0, vz: 0,
          })
        }
      }
    }
  }

  /** Fetch co-firing data from backend. */
  async fetchCofiring() {
    try {
      const data = await api.clusterCofiring()
      const pairs = data.pairs ?? []
      const newKeys = new Set<string>()
      for (const p of pairs) {
        const key = p.a < p.b ? `${p.a}|${p.b}` : `${p.b}|${p.a}`
        newKeys.add(key)
        const existing = this.springs.get(key)
        if (existing) {
          existing.targetStrength = p.strength
        } else {
          this.springs.set(key, {
            a: p.a < p.b ? p.a : p.b,
            b: p.a < p.b ? p.b : p.a,
            strength: this._hasFetched ? 0 : p.strength,  // snap on first load
            targetStrength: p.strength,
          })
        }
      }
      // Remove springs for pairs no longer in data
      for (const key of this.springs.keys()) {
        if (!newKeys.has(key)) this.springs.delete(key)
      }
      this._hasFetched = true
    } catch {}
  }

  /** Run one simulation step. */
  step(dt: number) {
    const ids = [...this.bodies.keys()]
    const n = ids.length
    if (n === 0) return

    // Boundary radius scales with sqrt(cluster count)
    const boundaryR = Math.max(1.5, Math.sqrt(n) * 0.5)

    // Lerp spring strengths toward targets
    for (const spring of this.springs.values()) {
      spring.strength += (spring.targetStrength - spring.strength) * this.springLerpSpeed
    }

    // Accumulate forces
    const forces = new Map<string, [number, number, number]>()
    for (const id of ids) forces.set(id, [0, 0, 0])

    // Repulsion: all pairs, inverse square
    for (let i = 0; i < n; i++) {
      const ai = ids[i]
      const a = this.bodies.get(ai)!
      for (let j = i + 1; j < n; j++) {
        const bi = ids[j]
        const b = this.bodies.get(bi)!
        let dx = a.x - b.x
        let dy = a.y - b.y
        let dz = a.z - b.z
        const distSq = dx * dx + dy * dy + dz * dz + 0.001
        const dist = Math.sqrt(distSq)
        const force = this.repulsionStrength / distSq
        const fx = (dx / dist) * force
        const fy = (dy / dist) * force
        const fz = (dz / dist) * force
        const fa = forces.get(ai)!
        const fb = forces.get(bi)!
        fa[0] += fx; fa[1] += fy; fa[2] += fz
        fb[0] -= fx; fb[1] -= fy; fb[2] -= fz
      }
    }

    // Attraction: co-firing springs
    for (const spring of this.springs.values()) {
      const a = this.bodies.get(spring.a)
      const b = this.bodies.get(spring.b)
      if (!a || !b) continue
      const dx = b.x - a.x
      const dy = b.y - a.y
      const dz = b.z - a.z
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz + 0.001)
      const force = this.attractionStrength * spring.strength * dist * 0.1
      const fx = (dx / dist) * force
      const fy = (dy / dist) * force
      const fz = (dz / dist) * force
      const fa = forces.get(spring.a)!
      const fb = forces.get(spring.b)!
      fa[0] += fx; fa[1] += fy; fa[2] += fz
      fb[0] -= fx; fb[1] -= fy; fb[2] -= fz
    }

    // Boundary: soft push toward center
    for (const id of ids) {
      const body = this.bodies.get(id)!
      const dist = Math.sqrt(body.x * body.x + body.y * body.y + body.z * body.z)
      if (dist > boundaryR) {
        const overshoot = dist - boundaryR
        const pushStrength = overshoot * 0.3
        const f = forces.get(id)!
        f[0] -= (body.x / dist) * pushStrength
        f[1] -= (body.y / dist) * pushStrength
        f[2] -= (body.z / dist) * pushStrength
      }
    }

    // Integrate: apply forces, damp, clamp velocity
    for (const id of ids) {
      const body = this.bodies.get(id)!
      const f = forces.get(id)!
      body.vx = (body.vx + f[0] * dt) * this.damping
      body.vy = (body.vy + f[1] * dt) * this.damping
      body.vz = (body.vz + f[2] * dt) * this.damping
      // Clamp velocity
      const speed = Math.sqrt(body.vx * body.vx + body.vy * body.vy + body.vz * body.vz)
      if (speed > this.maxVelocity) {
        const scale = this.maxVelocity / speed
        body.vx *= scale; body.vy *= scale; body.vz *= scale
      }
      body.x += body.vx * dt
      body.y += body.vy * dt
      body.z += body.vz * dt
    }
  }

  getPosition(id: string): [number, number, number] {
    const b = this.bodies.get(id)
    return b ? [b.x, b.y, b.z] : [0, 0, 0]
  }

  private _bestPartner(cid: string): string | null {
    let best: string | null = null
    let bestStr = 0
    for (const spring of this.springs.values()) {
      if (spring.a === cid && spring.strength > bestStr && this.bodies.has(spring.b)) {
        best = spring.b; bestStr = spring.strength
      }
      if (spring.b === cid && spring.strength > bestStr && this.bodies.has(spring.a)) {
        best = spring.a; bestStr = spring.strength
      }
    }
    return best
  }
}

// Singleton simulation — lives across React re-renders
const _sim = new ForceSimulation()

// ═══════════════════════════════════════════════════════════════════════
//  LIVE mode renderer — force-directed simulation
// ═══════════════════════════════════════════════════════════════════════

function LiveRenderer({ onHover }: { onHover: (info: HoverInfo | null) => void }) {
  const clusters = useGraphStore(s => s.clusters)
  const nodes = useGraphStore(s => s.nodes)
  const edges = useGraphStore(s => s.edges)
  const activations = useGraphStore(s => s.activations)
  const lastFetch = useRef(0)

  // Sync cluster set into simulation
  const activeIds = useMemo(() => clusters.filter(c => !c.dormant).map(c => c.id), [clusters])
  useEffect(() => { _sim.syncClusters(activeIds) }, [activeIds])

  // Fetch co-firing on mount + every 60s
  useEffect(() => {
    _sim.fetchCofiring()
    const timer = setInterval(() => _sim.fetchCofiring(), 60_000)
    return () => clearInterval(timer)
  }, [])

  // Mutable positions that update every frame — avoids React re-renders
  const positionsRef = useRef<Record<string, [number, number, number]>>({})

  useFrame((_, delta) => {
    // Run physics at ~60fps step rate, clamped
    const dt = Math.min(delta, 1 / 30)
    _sim.step(dt)
    // Copy positions out for rendering
    for (const cid of activeIds) {
      positionsRef.current[cid] = _sim.getPosition(cid)
    }
  })

  return (
    <group>
      {/* Co-firing edges */}
      {edges.map(edge => (
        <LiveEdge
          key={`${edge.from}-${edge.to}`}
          edge={edge}
          positionsRef={positionsRef}
        />
      ))}

      {/* Cluster spheres at simulation positions */}
      {clusters.filter(c => !c.dormant).map(cluster => (
        <LiveClusterSphere
          key={cluster.id}
          cluster={cluster}
          clusterNodes={nodes.filter(n => n.cluster === cluster.id)}
          activation={activations[cluster.id] ?? 0}
          positionsRef={positionsRef}
          onHover={onHover}
        />
      ))}
    </group>
  )
}

// ── Live cluster sphere — reads position from sim each frame ──

interface LiveClusterSphereProps {
  cluster: GraphCluster
  clusterNodes: GraphNode[]
  activation: number
  positionsRef: React.MutableRefObject<Record<string, [number, number, number]>>
  onHover: (info: HoverInfo | null) => void
}

function LiveClusterSphere({ cluster, clusterNodes, activation, positionsRef, onHover }: LiveClusterSphereProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const matRef = useRef<THREE.MeshStandardMaterial>(null)
  const color = clusterColor(cluster.cluster_type)
  const baseScale = 0.12 + activation * 0.08
  const pulsePhase = useRef(Math.random() * Math.PI * 2)

  useFrame((_, delta) => {
    if (!meshRef.current) return
    // Move mesh to sim position
    const pos = positionsRef.current[cluster.id]
    if (pos) {
      meshRef.current.position.set(pos[0], pos[1], pos[2])
    }

    // Pulse: faster for recently active
    const pulseSpeed = activation > 0.1 ? 6 : 1.5
    pulsePhase.current += delta * pulseSpeed
    const pulse = 1 + Math.sin(pulsePhase.current) * 0.15 * (activation > 0.1 ? 1 : 0.3)
    const targetScale = baseScale * pulse
    const s = meshRef.current.scale.x
    meshRef.current.scale.setScalar(s + (targetScale - s) * Math.min(delta * ANIMATION.PULSE_LERP_SPEED, 1))

    // Opacity: brighter for more active
    if (matRef.current) {
      const opTarget = Math.max(0.25, 0.3 + activation * 0.7)
      matRef.current.opacity += (opTarget - matRef.current.opacity) * Math.min(delta * 4, 1)
    }
  })

  const handlePointerEnter = useCallback((e: any) => {
    e.stopPropagation()
    const pos = positionsRef.current[cluster.id] ?? [0, 0, 0]
    const vec = new THREE.Vector3(pos[0], pos[1], pos[2])
    const canvas = e.target?.ownerDocument?.querySelector('canvas') as HTMLCanvasElement | null
    if (canvas) {
      vec.project(e.camera)
      const rect = canvas.getBoundingClientRect()
      onHover({
        clusterId: cluster.id,
        clusterType: cluster.cluster_type,
        screenX: ((vec.x + 1) / 2) * rect.width + rect.left,
        screenY: ((-vec.y + 1) / 2) * rect.height + rect.top,
      })
    }
  }, [cluster.id, cluster.cluster_type, positionsRef, onHover])

  const handlePointerLeave = useCallback((e: any) => {
    e.stopPropagation()
    onHover(null)
  }, [onHover])

  return (
    <mesh
      ref={meshRef}
      position={[0, 0, 0]}
      scale={[baseScale, baseScale, baseScale]}
      onPointerEnter={handlePointerEnter}
      onPointerLeave={handlePointerLeave}
    >
      <sphereGeometry args={[1, 8, 8]} />
      <meshStandardMaterial
        ref={matRef}
        color={color}
        emissive={color}
        emissiveIntensity={0.2 + activation * 1.2}
        roughness={0.6}
        metalness={0.2}
        transparent
        opacity={0.5}
      />
    </mesh>
  )
}

// ── Live edge — position updates each frame from sim ──

interface LiveEdgeProps {
  edge: GraphEdge
  positionsRef: React.MutableRefObject<Record<string, [number, number, number]>>
}

function LiveEdge({ edge, positionsRef }: LiveEdgeProps) {
  const lineRef = useRef<THREE.Line>(null)
  const opacityRef = useRef(0)
  const targetOpacity = edgeOpacity(edge.strength)

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3))
    return geo
  }, [])

  const material = useMemo(() => {
    return new THREE.LineBasicMaterial({
      color: '#4effa0',
      transparent: true,
      opacity: 0,
    })
  }, [])

  useFrame((_, delta) => {
    // Update line endpoints from simulation positions
    const a = positionsRef.current[edge.from] ?? [0, 0, 0]
    const b = positionsRef.current[edge.to] ?? [0, 0, 0]
    const posAttr = geometry.getAttribute('position') as THREE.BufferAttribute
    posAttr.setXYZ(0, a[0], a[1], a[2])
    posAttr.setXYZ(1, b[0], b[1], b[2])
    posAttr.needsUpdate = true

    const tgt = Math.min(targetOpacity * 1.5, 1)
    opacityRef.current += (tgt - opacityRef.current) * Math.min(delta * 3, 1)
    material.opacity = opacityRef.current
  })

  return <primitive ref={lineRef} object={new THREE.Line(geometry, material)} />
}

// ═══════════════════════════════════════════════════════════════════════
//  UMAP mode renderer (original, unchanged)
// ═══════════════════════════════════════════════════════════════════════

function UmapRenderer({ onHover }: { onHover: (info: HoverInfo | null) => void }) {
  const clusters = useGraphStore(s => s.clusters)
  const nodes = useGraphStore(s => s.nodes)
  const edges = useGraphStore(s => s.edges)
  const activations = useGraphStore(s => s.activations)

  return (
    <group>
      {edges.map(edge => (
        <UmapEdge
          key={`${edge.from}-${edge.to}`}
          edge={edge}
          fromPos={clusterCentroid(nodes, edge.from)}
          toPos={clusterCentroid(nodes, edge.to)}
        />
      ))}

      {clusters.map(cluster => (
        <UmapClusterGroup
          key={cluster.id}
          cluster={cluster}
          nodes={nodes.filter(n => n.cluster === cluster.id)}
          activation={activations[cluster.id] ?? 0}
          onHover={onHover}
        />
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
        <UmapNodeSphere
          key={node.id}
          node={node}
          clusterId={cluster.id}
          clusterType={cluster.cluster_type}
          color={color}
          activation={activation}
          onHover={onHover}
        />
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
    const current = meshRef.current.scale.x
    meshRef.current.scale.setScalar(current + (target - current) * Math.min(delta * ANIMATION.PULSE_LERP_SPEED, 1))
    if (matRef.current) {
      matRef.current.opacity += (dormantTarget - matRef.current.opacity) * Math.min(delta * 4, 1)
    }
  })

  const handlePointerEnter = useCallback((e: any) => {
    e.stopPropagation()
    const pos = node.pos ?? [0, 0, 0]
    const vec = new THREE.Vector3(pos[0], pos[1], pos[2])
    const canvas = e.target?.ownerDocument?.querySelector('canvas') as HTMLCanvasElement | null
    if (canvas) {
      vec.project(e.camera)
      const rect = canvas.getBoundingClientRect()
      onHover({ clusterId, clusterType, screenX: ((vec.x + 1) / 2) * rect.width + rect.left, screenY: ((-vec.y + 1) / 2) * rect.height + rect.top })
    }
  }, [node.pos, clusterId, clusterType, onHover])

  const handlePointerLeave = useCallback((e: any) => { e.stopPropagation(); onHover(null) }, [onHover])

  return (
    <mesh ref={meshRef} position={node.pos ?? [0, 0, 0]} scale={[baseScale, baseScale, baseScale]}
      onPointerEnter={handlePointerEnter} onPointerLeave={handlePointerLeave}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshStandardMaterial ref={matRef} color={node.alive ? color : '#2a2a2a'}
        emissive={node.alive ? color : '#2a2a2a'} emissiveIntensity={0.4 + activation * 0.6}
        roughness={0.6} metalness={0.2} transparent opacity={dormantTarget} />
    </mesh>
  )
}

function UmapEdge({ edge, fromPos, toPos }: { edge: GraphEdge; fromPos: [number, number, number]; toPos: [number, number, number] }) {
  const opacityRef = useRef(0)
  const targetOpacity = edgeOpacity(edge.strength)
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      fromPos[0], fromPos[1], fromPos[2], toPos[0], toPos[1], toPos[2],
    ]), 3))
    return geo
  }, [fromPos[0], fromPos[1], fromPos[2], toPos[0], toPos[1], toPos[2]])
  const material = useMemo(() => new THREE.LineBasicMaterial({ color: '#4effa0', transparent: true, opacity: 0 }), [])
  useFrame((_, delta) => {
    opacityRef.current += (targetOpacity - opacityRef.current) * Math.min(delta * 3, 1)
    material.opacity = opacityRef.current
  })
  return <primitive object={new THREE.Line(geometry, material)} />
}

function edgeOpacity(strength: number): number {
  if (strength < 0.2) return 0.1
  if (strength < 0.5) return 0.3
  if (strength < 0.8) return 0.6
  return 1.0
}

// ═══════════════════════════════════════════════════════════════════════
//  TREE mode renderer — radial BUD-tree layout
// ═══════════════════════════════════════════════════════════════════════

interface TreeRendererProps {
  treeData: { nodes: TreeNode[]; edges: { source: string; target: string }[]; max_depth: number }
  onHover: (info: HoverInfo | null) => void
}

function TreeRenderer({ treeData, onHover }: TreeRendererProps) {
  const activations = useGraphStore(s => s.activations)

  const layout = useMemo(() => {
    const empty = { positions: new Map<string, [number, number, number]>() }
    if (treeData.nodes.length === 0) return empty

    const childrenOf = new Map<string, TreeNode[]>()
    for (const n of treeData.nodes) {
      if (n.parent) {
        const list = childrenOf.get(n.parent) ?? []
        list.push(n)
        childrenOf.set(n.parent, list)
      }
    }

    const roots = treeData.nodes.filter(n => n.parent === null)
    const positions = new Map<string, [number, number, number]>()
    const maxDepth = Math.max(treeData.max_depth, 1)
    const radiusScale = 3 / maxDepth

    if (roots.length === 1) {
      positions.set(roots[0].id, [0, 0, 0])
    } else {
      for (let i = 0; i < roots.length; i++) {
        const angle = (i / roots.length) * Math.PI * 2
        const r = Math.min(0.5, radiusScale * 0.3)
        positions.set(roots[i].id, [Math.cos(angle) * r, 0, Math.sin(angle) * r])
      }
    }

    interface QueueItem { node: TreeNode; angleStart: number; angleEnd: number }
    const queue: QueueItem[] = roots.map((root, i) => ({
      node: root,
      angleStart: (i / Math.max(roots.length, 1)) * Math.PI * 2,
      angleEnd: ((i + 1) / Math.max(roots.length, 1)) * Math.PI * 2,
    }))

    while (queue.length > 0) {
      const { node, angleStart, angleEnd } = queue.shift()!
      const children = childrenOf.get(node.id) ?? []
      if (children.length === 0) continue
      const angleStep = (angleEnd - angleStart) / children.length
      for (let i = 0; i < children.length; i++) {
        const child = children[i]
        const angle = angleStart + angleStep * (i + 0.5)
        const radius = child.depth * radiusScale
        positions.set(child.id, [Math.cos(angle) * radius, child.depth * 0.1, Math.sin(angle) * radius])
        queue.push({ node: child, angleStart: angleStart + angleStep * i, angleEnd: angleStart + angleStep * (i + 1) })
      }
    }

    return { positions }
  }, [treeData])

  return (
    <group>
      {treeData.edges.map(edge => {
        const from = layout.positions.get(edge.source) ?? [0, 0, 0]
        const to = layout.positions.get(edge.target) ?? [0, 0, 0]
        return <TreeEdgeLine key={`${edge.source}-${edge.target}`} from={from as [number, number, number]} to={to as [number, number, number]} />
      })}
      {treeData.nodes.map(node => {
        const pos = layout.positions.get(node.id) ?? [0, 0, 0]
        return <TreeClusterSphere key={node.id} node={node} position={pos as [number, number, number]} activation={activations[node.id] ?? 0} onHover={onHover} />
      })}
    </group>
  )
}

function TreeClusterSphere({ node, position, activation, onHover }: {
  node: TreeNode; position: [number, number, number]; activation: number; onHover: (info: HoverInfo | null) => void
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const matRef = useRef<THREE.MeshStandardMaterial>(null)
  const isPhantom = node.phantom
  const isActive = !node.dormant && !isPhantom
  const baseScale = isPhantom ? 0.06 : 0.12 + activation * 0.06
  const color = node.cluster_type ? clusterColor(node.cluster_type) : (isPhantom ? '#333333' : '#888888')

  useFrame((_, delta) => {
    if (!meshRef.current) return
    const cur = meshRef.current.position
    const speed = Math.min(delta * 3, 1)
    cur.x += (position[0] - cur.x) * speed
    cur.y += (position[1] - cur.y) * speed
    cur.z += (position[2] - cur.z) * speed
    const pulseTarget = isActive ? baseScale * (1 + activation * 0.4) : baseScale * 0.6
    const s = meshRef.current.scale.x
    meshRef.current.scale.setScalar(s + (pulseTarget - s) * Math.min(delta * 6, 1))
    if (matRef.current) {
      const opTarget = isActive ? 0.5 + activation * 0.5 : 0.15
      matRef.current.opacity += (opTarget - matRef.current.opacity) * Math.min(delta * 4, 1)
    }
  })

  const handlePointerEnter = useCallback((e: any) => {
    e.stopPropagation()
    const vec = new THREE.Vector3(position[0], position[1], position[2])
    const canvas = e.target?.ownerDocument?.querySelector('canvas') as HTMLCanvasElement | null
    if (canvas) {
      vec.project(e.camera)
      const rect = canvas.getBoundingClientRect()
      onHover({ clusterId: node.id, clusterType: node.cluster_type ?? 'unknown', screenX: ((vec.x + 1) / 2) * rect.width + rect.left, screenY: ((-vec.y + 1) / 2) * rect.height + rect.top })
    }
  }, [position, node, onHover])
  const handlePointerLeave = useCallback((e: any) => { e.stopPropagation(); onHover(null) }, [onHover])

  return (
    <mesh ref={meshRef} position={position} scale={[baseScale, baseScale, baseScale]}
      onPointerEnter={handlePointerEnter} onPointerLeave={handlePointerLeave}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshStandardMaterial ref={matRef} color={color} emissive={isActive ? color : '#111111'}
        emissiveIntensity={isActive ? 0.3 + activation * 0.9 : 0.05}
        roughness={0.6} metalness={0.2} transparent opacity={isActive ? 0.8 : 0.15} />
    </mesh>
  )
}

function TreeEdgeLine({ from, to }: { from: [number, number, number]; to: [number, number, number] }) {
  const opacityRef = useRef(0)
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([from[0], from[1], from[2], to[0], to[1], to[2]]), 3))
    return geo
  }, [from[0], from[1], from[2], to[0], to[1], to[2]])
  const material = useMemo(() => new THREE.LineBasicMaterial({ color: '#2a4a3a', transparent: true, opacity: 0 }), [])
  useFrame((_, delta) => {
    opacityRef.current += (0.3 - opacityRef.current) * Math.min(delta * 3, 1)
    material.opacity = opacityRef.current
  })
  return <primitive object={new THREE.Line(geometry, material)} />
}
