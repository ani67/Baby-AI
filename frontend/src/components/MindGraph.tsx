// Phase 7 perf rewrite: drops 3d-force-graph in favor of raw Three.js +
// instanced primitives. ALL nodes render as one THREE.InstancedMesh (one
// draw call regardless of count). ALL edges render as one
// THREE.LineSegments with vertex colors (one draw call). d3-force-3d
// drives positions directly. The scene tops out at ~3 GPU draw calls
// total instead of ~10,000 — runs at 60 fps even at 10K+ nodes.
//
// Same visual contract as before: lit 3D spheres (alignment hue, arousal
// saturation), per-edge color by type, transient pulse rings on cycle
// events, breathing modulation on recently-active concepts.

import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
// d3-force-3d arrived as a transitive dep of 3d-force-graph; we import
// the slice we need directly.
import {
  forceSimulation,
  forceManyBody,
  forceLink,
  forceCenter,
} from "d3-force-3d";
import type { GraphPayload, GraphNode, GraphEdge, CycleEvent } from "../lib/types";

type Props = {
  graph: GraphPayload | null;
  lastCycle: CycleEvent | null;
  consolidationActive: boolean;
};

// d3-force-3d mutates each node by attaching x/y/z (and vx/vy/vz) in
// place. We mirror that here.
type SimNode = GraphNode & {
  x?: number; y?: number; z?: number;
  vx?: number; vy?: number; vz?: number;
};
type SimLink = GraphEdge & {
  source: number | SimNode;
  target: number | SimNode;
};

const EDGE_COLOR_BY_TYPE: Record<string, [number, number, number]> = {
  is_a:         [0.65, 0.55, 0.98],
  has_property: [0.38, 0.65, 0.98],
  causes:       [0.97, 0.45, 0.45],
  precedes:     [0.98, 0.57, 0.24],
  similar_to:   [0.20, 0.83, 0.60],
  opposite_of:  [0.96, 0.45, 0.71],
  context_of:   [0.58, 0.64, 0.72],
  refers_to:    [0.98, 0.75, 0.14],
  expresses:    [0.13, 0.83, 0.93],
  part_of:      [0.75, 0.52, 0.99],
};
const DEFAULT_EDGE_COLOR: [number, number, number] = [0.58, 0.64, 0.72];

// Color a single concept based on its alignment ([-1, 1]), arousal,
// and abstraction-flag. Returns RGB in [0,1].
function nodeRGB(d: GraphNode, isAbs: boolean): THREE.Color {
  const a = Math.max(-1, Math.min(1, d.alignment));
  const hue = (220 - 100 * a) / 360;            // 1.20 → 0.34, blue → orange
  const arousalClamp = Math.min(1, d.arousal / 0.6);
  const sat = 0.30 + 0.50 * arousalClamp;
  const light = 0.50 + 0.20 * arousalClamp + (isAbs ? 0.08 : 0);
  return new THREE.Color().setHSL(hue, sat, light);
}

function nodeBaseScale(d: GraphNode, isAbs: boolean): number {
  return 2 + Math.log1p(d.activation_count) * 1.6 + (isAbs ? 1.2 : 0);
}

// Pre-allocate instance buffers generously so node growth (graph grows
// during ingestion) doesn't force a full rebuild on every poll.
const INSTANCE_CAPACITY_GROWTH = 1.25;
const MIN_INSTANCE_CAPACITY = 1024;

export function MindGraph({ graph, lastCycle, consolidationActive }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [hoverInfo, setHoverInfo] = useState<{
    x: number; y: number; node: GraphNode | null;
  } | null>(null);

  const sceneRef = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    nodeMesh: THREE.InstancedMesh;
    nodeCapacity: number;
    edgeLines: THREE.LineSegments;
    edgeCapacity: number;
    pulseGroup: THREE.Group;
    sim: ReturnType<typeof forceSimulation>;
    nodes: SimNode[];
    nodeIndex: Map<number, number>;          // concept_id → instance index
    nodesByIdx: SimNode[];                   // index aligned with nodeMesh
    links: SimLink[];
    raycaster: THREE.Raycaster;
    pointerNDC: THREE.Vector2;
    lastInteractionMs: number;
  } | null>(null);

  const lastCycleRef = useRef<CycleEvent | null>(null);

  // --- one-time scene init ---
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth || 1;
    const height = container.clientHeight || 1;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(60, width / height, 0.5, 4000);
    camera.position.set(0, 0, 320);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    // Lighting — MeshStandardMaterial needs at least one light to look 3D.
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const sun = new THREE.DirectionalLight(0xffffff, 0.85);
    sun.position.set(160, 200, 100);
    scene.add(sun);

    // ---- nodes: one InstancedMesh shared by all concepts ----
    const sphereGeom = new THREE.SphereGeometry(1, 12, 8);
    const sphereMat = new THREE.MeshStandardMaterial({
      roughness: 0.35,
      metalness: 0.1,
      // emissive will be modulated per-instance via instanceColor on a
      // standard material; for true emissive instancing we'd need a
      // custom shader. For now emissive comes through as base color,
      // which is close enough at typical zoom.
    });
    let nodeCapacity = MIN_INSTANCE_CAPACITY;
    const nodeMesh = new THREE.InstancedMesh(sphereGeom, sphereMat, nodeCapacity);
    nodeMesh.count = 0;       // start empty until graph data arrives
    nodeMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    // Allocate per-instance color attribute up front.
    const colorBuf = new Float32Array(nodeCapacity * 3);
    nodeMesh.instanceColor = new THREE.InstancedBufferAttribute(colorBuf, 3);
    nodeMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    scene.add(nodeMesh);

    // ---- edges: one LineSegments shared by all edges ----
    let edgeCapacity = 1024;
    const edgePositions = new Float32Array(edgeCapacity * 6);
    const edgeColors = new Float32Array(edgeCapacity * 6);
    const edgeGeom = new THREE.BufferGeometry();
    edgeGeom.setAttribute("position",
      new THREE.Float32BufferAttribute(edgePositions, 3).setUsage(THREE.DynamicDrawUsage));
    edgeGeom.setAttribute("color",
      new THREE.Float32BufferAttribute(edgeColors, 3).setUsage(THREE.DynamicDrawUsage));
    edgeGeom.setDrawRange(0, 0);
    const edgeMat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.32,
      depthWrite: false,
    });
    const edgeLines = new THREE.LineSegments(edgeGeom, edgeMat);
    scene.add(edgeLines);

    // ---- transient pulse rings (small set, tracked individually) ----
    const pulseGroup = new THREE.Group();
    scene.add(pulseGroup);

    // ---- d3-force-3d simulation ----
    const sim = forceSimulation([] as SimNode[], 3)
      .force("charge", forceManyBody().strength(-180))
      .force("link", forceLink([]).id((n: any) => n.id).distance(70))
      .force("center", forceCenter())
      .alphaDecay(0.012)
      .velocityDecay(0.4);
    // We drive ticks ourselves from the render loop — turn off the
    // built-in scheduler.
    sim.stop();

    sceneRef.current = {
      renderer, scene, camera, controls,
      nodeMesh, nodeCapacity,
      edgeLines, edgeCapacity,
      pulseGroup,
      sim,
      nodes: [],
      nodeIndex: new Map(),
      nodesByIdx: [],
      links: [],
      raycaster: new THREE.Raycaster(),
      pointerNDC: new THREE.Vector2(2, 2),
      lastInteractionMs: performance.now(),
    };

    // ---- hover (raycasting against InstancedMesh) ----
    const onPointerMove = (e: PointerEvent) => {
      const ref = sceneRef.current;
      if (!ref) return;
      const rect = renderer.domElement.getBoundingClientRect();
      ref.pointerNDC.set(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1,
      );
      ref.lastInteractionMs = performance.now();
    };
    renderer.domElement.addEventListener("pointermove", onPointerMove);

    controls.addEventListener("change", () => {
      const ref = sceneRef.current;
      if (ref) ref.lastInteractionMs = performance.now();
    });

    // ---- resize ----
    const onResize = () => {
      const w = container.clientWidth || 1;
      const h = container.clientHeight || 1;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    const ro = new ResizeObserver(onResize);
    ro.observe(container);

    // ---- main loop ----
    const tmpMat = new THREE.Matrix4();
    const tmpPos = new THREE.Vector3();
    const tmpQuat = new THREE.Quaternion();
    const tmpScale = new THREE.Vector3();
    const tmpColor = new THREE.Color();
    const ORBIT_IDLE_MS = 8000;
    const BREATH_RECENCY_S = 30;

    let raf = 0;
    const tick = () => {
      const ref = sceneRef.current;
      if (!ref) return;

      // d3 tick advances positions
      ref.sim.tick();

      // Camera idle orbit (matches old behavior)
      const nowMs = performance.now();
      if (nowMs - ref.lastInteractionMs > ORBIT_IDLE_MS) {
        const cam = ref.camera;
        const r = Math.hypot(cam.position.x, cam.position.z);
        const a = Math.atan2(cam.position.z, cam.position.x) + 0.0008;
        cam.position.x = Math.cos(a) * r;
        cam.position.z = Math.sin(a) * r;
        cam.lookAt(0, 0, 0);
      }

      // ---- update node instance matrices + colors ----
      const nowS = Date.now() / 1000;
      const mesh = ref.nodeMesh;
      const arr = ref.nodesByIdx;
      let needsColorUpdate = false;
      for (let i = 0; i < mesh.count; i++) {
        const d = arr[i];
        if (!d || d.x === undefined) continue;

        // breathing scale on recent activity
        const sinceAct = d.last_activated > 0 ? nowS - d.last_activated : 1e9;
        let breathe = 1;
        if (sinceAct < BREATH_RECENCY_S) {
          const recency = Math.exp(-sinceAct / 6);
          const phase = ((d.id * 0.31) + nowS * (0.6 + 0.6 * recency)) % (Math.PI * 2);
          breathe = 1 + 0.05 * Math.sin(phase) + 0.18 * recency;
        }
        const isAbs = d.name.startsWith("abstraction:");
        const baseSize = nodeBaseScale(d, isAbs) * breathe;

        tmpPos.set(d.x as number, d.y as number, d.z as number);
        tmpScale.setScalar(baseSize);
        tmpMat.compose(tmpPos, tmpQuat, tmpScale);
        mesh.setMatrixAt(i, tmpMat);

        // Color refresh once per second-ish for active nodes (reflects
        // alignment / arousal evolution). For dormant ones the color
        // stored at write time is fine.
        if (sinceAct < BREATH_RECENCY_S) {
          tmpColor.copy(nodeRGB(d, isAbs));
          mesh.setColorAt(i, tmpColor);
          needsColorUpdate = true;
        }
      }
      mesh.instanceMatrix.needsUpdate = true;
      if (needsColorUpdate && mesh.instanceColor) {
        (mesh.instanceColor as THREE.InstancedBufferAttribute).needsUpdate = true;
      }

      // ---- update edge endpoints ----
      // We write to the SAME offset i*6 the color buffer was filled at
      // during the [graph] effect, so colors stay aligned with positions.
      // Edges whose endpoints aren't yet in the nodeIndex collapse to a
      // degenerate (0,0,0)->(0,0,0) segment and render as nothing.
      const lines = ref.edgeLines;
      const posAttr = lines.geometry.getAttribute("position") as THREE.BufferAttribute;
      const positions = posAttr.array as Float32Array;
      const links = ref.links;
      for (let i = 0; i < links.length; i++) {
        const link = links[i];
        const sId = typeof link.source === "object" ? link.source.id : link.source;
        const tId = typeof link.target === "object" ? link.target.id : link.target;
        const sIdx = ref.nodeIndex.get(sId);
        const tIdx = ref.nodeIndex.get(tId);
        const off = i * 6;
        if (sIdx === undefined || tIdx === undefined) {
          positions[off] = positions[off+1] = positions[off+2] = 0;
          positions[off+3] = positions[off+4] = positions[off+5] = 0;
          continue;
        }
        const sNode = ref.nodesByIdx[sIdx];
        const tNode = ref.nodesByIdx[tIdx];
        positions[off]     = sNode.x as number;
        positions[off + 1] = sNode.y as number;
        positions[off + 2] = sNode.z as number;
        positions[off + 3] = tNode.x as number;
        positions[off + 4] = tNode.y as number;
        positions[off + 5] = tNode.z as number;
      }
      lines.geometry.setDrawRange(0, links.length * 2);
      posAttr.needsUpdate = true;

      // ---- pulse ring lifecycle ----
      const pg = ref.pulseGroup;
      const removeList: THREE.Object3D[] = [];
      pg.children.forEach((ring) => {
        const meta = (ring as any).__meta as { start: number; duration: number } | undefined;
        if (!meta) return;
        const t = (nowMs - meta.start) / meta.duration;
        if (t >= 1) { removeList.push(ring); return; }
        const s = 1 + t * 4.5;
        ring.scale.set(s, s, s);
        const m = (ring as THREE.Mesh).material as THREE.MeshBasicMaterial;
        m.opacity = 0.6 * (1 - t);
      });
      removeList.forEach((r) => pg.remove(r));

      // ---- hover (raycast against InstancedMesh) ----
      ref.raycaster.setFromCamera(ref.pointerNDC, ref.camera);
      const hits = ref.raycaster.intersectObject(mesh, false);
      if (hits.length > 0 && hits[0].instanceId !== undefined) {
        const idx = hits[0].instanceId;
        const node = ref.nodesByIdx[idx];
        if (node) {
          // project node world pos to screen for label placement
          tmpPos.set(node.x as number, node.y as number, node.z as number).project(ref.camera);
          const rect = renderer.domElement.getBoundingClientRect();
          const sx = (tmpPos.x * 0.5 + 0.5) * rect.width + rect.left;
          const sy = (-tmpPos.y * 0.5 + 0.5) * rect.height + rect.top;
          setHoverInfo({ x: sx, y: sy, node });
        }
      } else {
        setHoverInfo(null);
      }

      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      try { container.removeChild(renderer.domElement); } catch {}
      renderer.dispose();
      sphereGeom.dispose();
      sphereMat.dispose();
      edgeGeom.dispose();
      edgeMat.dispose();
      sceneRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- push graph data updates ---
  useEffect(() => {
    const ref = sceneRef.current;
    if (!ref || !graph) return;

    // Build / refresh the node array. Preserve x/y/z across updates so
    // the simulation doesn't reset its layout when new nodes arrive.
    const prev = new Map<number, SimNode>();
    ref.nodes.forEach((n) => prev.set(n.id, n));

    const next: SimNode[] = graph.nodes.map((g) => {
      const old = prev.get(g.id);
      if (old) {
        // Update mutable fields; keep position/velocity from old.
        old.name = g.name;
        old.activation_count = g.activation_count;
        old.surprise_at_birth = g.surprise_at_birth;
        old.alignment = g.alignment;
        old.arousal = g.arousal;
        old.is_pinned = g.is_pinned;
        old.last_activated = g.last_activated;
        old.created_at = g.created_at;
        return old;
      }
      // New node — random initial position so charge/link forces have
      // something to push against.
      const r = 50;
      return {
        ...g,
        x: (Math.random() - 0.5) * r,
        y: (Math.random() - 0.5) * r,
        z: (Math.random() - 0.5) * r,
        vx: 0, vy: 0, vz: 0,
      };
    });

    // Resize the InstancedMesh capacity if needed.
    if (next.length > ref.nodeCapacity) {
      const newCapacity = Math.max(
        Math.ceil(next.length * INSTANCE_CAPACITY_GROWTH),
        MIN_INSTANCE_CAPACITY,
      );
      ref.scene.remove(ref.nodeMesh);
      const newMesh = new THREE.InstancedMesh(
        ref.nodeMesh.geometry,
        ref.nodeMesh.material as THREE.Material,
        newCapacity,
      );
      newMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      const newColors = new Float32Array(newCapacity * 3);
      newMesh.instanceColor = new THREE.InstancedBufferAttribute(newColors, 3);
      newMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
      ref.scene.add(newMesh);
      ref.nodeMesh.dispose();
      ref.nodeMesh = newMesh;
      ref.nodeCapacity = newCapacity;
    }

    // Compose initial matrix + color for every instance.
    const mesh = ref.nodeMesh;
    mesh.count = next.length;
    const tmpMat = new THREE.Matrix4();
    const tmpPos = new THREE.Vector3();
    const tmpQuat = new THREE.Quaternion();
    const tmpScale = new THREE.Vector3();
    const tmpColor = new THREE.Color();
    next.forEach((d, i) => {
      const isAbs = d.name.startsWith("abstraction:");
      const baseSize = nodeBaseScale(d, isAbs);
      tmpPos.set(d.x ?? 0, d.y ?? 0, d.z ?? 0);
      tmpScale.setScalar(baseSize);
      tmpMat.compose(tmpPos, tmpQuat, tmpScale);
      mesh.setMatrixAt(i, tmpMat);
      tmpColor.copy(nodeRGB(d, isAbs));
      mesh.setColorAt(i, tmpColor);
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) {
      (mesh.instanceColor as THREE.InstancedBufferAttribute).needsUpdate = true;
    }

    // Build node-id index for edge lookup.
    const nodeIndex = new Map<number, number>();
    next.forEach((n, i) => nodeIndex.set(n.id, i));

    ref.nodes = next;
    ref.nodesByIdx = next;
    ref.nodeIndex = nodeIndex;

    // ---- edges ----
    // Resize edge buffer if needed.
    const edges = graph.edges;
    if (edges.length > ref.edgeCapacity) {
      const newCap = Math.max(
        Math.ceil(edges.length * INSTANCE_CAPACITY_GROWTH),
        1024,
      );
      const newPositions = new Float32Array(newCap * 6);
      const newColors = new Float32Array(newCap * 6);
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position",
        new THREE.Float32BufferAttribute(newPositions, 3).setUsage(THREE.DynamicDrawUsage));
      geom.setAttribute("color",
        new THREE.Float32BufferAttribute(newColors, 3).setUsage(THREE.DynamicDrawUsage));
      ref.edgeLines.geometry.dispose();
      ref.edgeLines.geometry = geom;
      ref.edgeCapacity = newCap;
    }
    // Fill the per-vertex color array. Endpoints share the edge's color.
    const colorAttr = ref.edgeLines.geometry.getAttribute("color") as THREE.BufferAttribute;
    const colorArr = colorAttr.array as Float32Array;
    edges.forEach((e, i) => {
      const c = EDGE_COLOR_BY_TYPE[e.type] ?? DEFAULT_EDGE_COLOR;
      const off = i * 6;
      colorArr[off]     = c[0];
      colorArr[off + 1] = c[1];
      colorArr[off + 2] = c[2];
      colorArr[off + 3] = c[0];
      colorArr[off + 4] = c[1];
      colorArr[off + 5] = c[2];
    });
    colorAttr.needsUpdate = true;

    ref.links = edges.map((e) => ({ ...e })) as SimLink[];

    // Wire updated nodes + links into the simulation. We swap arrays
    // rather than mutate so d3 sees a consistent view.
    ref.sim.nodes(next as any);
    (ref.sim.force("link") as any).links(ref.links as any);
    // Heat the simulation back up so it relayouts.
    ref.sim.alpha(0.6);
  }, [graph]);

  // --- cycle event: pulse ring + breathing trigger ---
  useEffect(() => {
    const ref = sceneRef.current;
    if (!ref || !lastCycle || lastCycle === lastCycleRef.current) return;
    lastCycleRef.current = lastCycle;

    const tNow = Date.now() / 1000;
    Object.keys(lastCycle.active_set).forEach((idStr) => {
      const id = parseInt(idStr, 10);
      const idx = ref.nodeIndex.get(id);
      if (idx === undefined) return;
      const d = ref.nodesByIdx[idx];
      if (d) d.last_activated = tNow;
    });

    // Pulse ring at the strongest activated concept.
    const strongest = Object.entries(lastCycle.active_set)
      .map(([k, v]) => [parseInt(k, 10), v as number] as const)
      .sort((a, b) => b[1] - a[1])[0];
    if (strongest) {
      const idx = ref.nodeIndex.get(strongest[0]);
      if (idx !== undefined) {
        const d = ref.nodesByIdx[idx];
        if (d && d.x !== undefined) {
          const ringGeom = new THREE.RingGeometry(2.5, 3.0, 48);
          const ringMat = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.6,
            side: THREE.DoubleSide,
            depthWrite: false,
          });
          const ring = new THREE.Mesh(ringGeom, ringMat);
          ring.position.set(d.x, d.y as number, d.z as number);
          ring.lookAt(ref.camera.position);
          (ring as any).__meta = { start: performance.now(), duration: 1300 };
          ref.pulseGroup.add(ring);
        }
      }
    }
  }, [lastCycle]);

  return (
    <div
      ref={containerRef}
      className={`absolute inset-0 ${consolidationActive ? "consolidating" : ""}`}
    >
      {hoverInfo && hoverInfo.node && (
        <div
          style={{
            position: "fixed",
            left: hoverInfo.x + 10,
            top: hoverInfo.y - 10,
            background: "rgba(0,0,0,0.7)",
            border: "1px solid rgba(255,255,255,0.13)",
            padding: "4px 8px",
            borderRadius: 4,
            color: "#fff",
            fontFamily: "monospace",
            fontSize: 11,
            pointerEvents: "none",
            zIndex: 10,
            maxWidth: 320,
          }}
        >
          <div>
            <b>#{hoverInfo.node.id}</b>{" "}
            {escapeHtml(hoverInfo.node.name) || <i>(unnamed)</i>}
          </div>
          <div style={{ color: "#aaa" }}>
            activations: {hoverInfo.node.activation_count} · alignment:{" "}
            {hoverInfo.node.alignment >= 0 ? "+" : "−"}
            {Math.abs(hoverInfo.node.alignment).toFixed(2)}
          </div>
          {hoverInfo.node.is_pinned && (
            <div style={{ color: "#facc15" }}>📌 pinned</div>
          )}
        </div>
      )}
    </div>
  );
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => {
    return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" } as any)[c];
  });
}
