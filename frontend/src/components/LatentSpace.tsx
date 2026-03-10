import { useRef, useMemo, useState, useCallback, useEffect } from 'react'
import { Canvas, useFrame, extend } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { useGraphStore, GraphNode, GraphEdge, GraphCluster } from '../store/graphStore'
import { clusterColor } from '../lib/colors'
import { ANIMATION } from '../lib/constants'
import { clusterCentroid } from '../hooks/useGraphState'

// Extend r3f catalog so <line_> maps to THREE.Line
extend({ Line_: THREE.Line })

export function LatentSpace() {
  const [autoRotate, setAutoRotate] = useState(true)
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>()

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
      <Canvas
        camera={{ position: [0, 0, 4], fov: 50 }}
        gl={{ antialias: true, alpha: false }}
        style={{ background: '#080808' }}
      >
        <ambientLight intensity={0.1} />
        <pointLight position={[2, 2, 2]} intensity={0.3} color="#4effa0" />

        <GridPlane />
        <GraphRenderer />

        <OrbitControls
          enablePan={false}
          enableZoom={true}
          autoRotate={autoRotate}
          autoRotateSpeed={0.3}
          makeDefault
        />
      </Canvas>
      {showLegend && <Legend onClose={() => setShowLegend(false)} />}
      {!showLegend && (
        <button className="legend-toggle" onClick={() => setShowLegend(true)}>?</button>
      )}
    </div>
  )
}

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

function GraphRenderer() {
  const clusters = useGraphStore(s => s.clusters)
  const nodes = useGraphStore(s => s.nodes)
  const edges = useGraphStore(s => s.edges)
  const activations = useGraphStore(s => s.activations)

  return (
    <group>
      {edges.map(edge => (
        <ClusterEdge
          key={`${edge.from}-${edge.to}`}
          edge={edge}
          fromPos={clusterCentroid(nodes, edge.from)}
          toPos={clusterCentroid(nodes, edge.to)}
        />
      ))}

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

interface ClusterGroupProps {
  cluster: GraphCluster
  nodes: GraphNode[]
  activation: number
}

function ClusterGroup({ cluster, nodes, activation }: ClusterGroupProps) {
  const color = clusterColor(cluster.cluster_type)

  return (
    <group>
      {nodes.map(node => (
        <NodeSphere
          key={node.id}
          node={node}
          color={color}
          activation={activation}
        />
      ))}
    </group>
  )
}

interface NodeSphereProps {
  node: GraphNode
  color: string
  activation: number
}

function NodeSphere({ node, color, activation }: NodeSphereProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const baseScale = 0.12 + Math.abs(node.activation_mean) * 0.08

  useFrame((_, delta) => {
    if (!meshRef.current) return
    const target = activation > 0.1
      ? baseScale * 1.5
      : baseScale
    const current = meshRef.current.scale.x
    const next = current + (target - current) * Math.min(delta * ANIMATION.PULSE_LERP_SPEED, 1)
    meshRef.current.scale.setScalar(next)
  })

  return (
    <mesh
      ref={meshRef}
      position={node.pos ?? [0, 0, 0]}
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

function edgeOpacity(strength: number): number {
  if (strength < 0.2) return 0.1
  if (strength < 0.5) return 0.3
  if (strength < 0.8) return 0.6
  return 1.0
}

interface ClusterEdgeProps {
  edge: GraphEdge
  fromPos: [number, number, number]
  toPos: [number, number, number]
}

function ClusterEdge({ edge, fromPos, toPos }: ClusterEdgeProps) {
  const ref = useRef<THREE.Line>(null)
  const opacityRef = useRef(0)
  const targetOpacity = edgeOpacity(edge.strength)

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    const positions = new Float32Array([
      fromPos[0], fromPos[1], fromPos[2],
      toPos[0], toPos[1], toPos[2],
    ])
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    return geo
  }, [fromPos[0], fromPos[1], fromPos[2], toPos[0], toPos[1], toPos[2]])

  const material = useMemo(() => {
    return new THREE.LineBasicMaterial({
      color: '#4effa0',
      transparent: true,
      opacity: 0,
    })
  }, [])

  useFrame((_, delta) => {
    opacityRef.current += (targetOpacity - opacityRef.current) * Math.min(delta * 3, 1)
    material.opacity = opacityRef.current
  })

  return <primitive object={new THREE.Line(geometry, material)} />
}
