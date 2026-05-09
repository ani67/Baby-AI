// GPU-instanced graph render with state textures + Web Worker force sim.
//
// Per-frame work on the main thread is reduced to: nothing (the worker
// drives layout, the GPU resolves positions in the vertex shader). The
// only main-thread JS in the render loop is OrbitControls + a single
// `posTex.needsUpdate = true` whenever the worker delivers a TICK.
//
// Architecture (one draw call for nodes, one for edges):
//
//   posTex     RGBA32F, one texel/node:    pos.xyz, _
//   stateTex   RGBA32F, one texel/node:    activation, arousal, act_count, pinned
//   edgeTex    RGBA32F, one texel/edge:    src_idx, tgt_idx, weight, type_norm
//
//   node draw  InstancedMesh(sphere, count=N), shader:
//                idx = gl_InstanceID
//                pos = texelFetch(posTex, idx2(idx))
//                state = texelFetch(stateTex, idx2(idx))
//                color/scale derived in shader from state + time
//
//   edge draw  LineSegments(count=2*E), shader:
//                edgeIdx  = gl_VertexID >> 1
//                endpoint = gl_VertexID & 1
//                edge = texelFetch(edgeTex, idx2(edgeIdx))
//                nodeIdx = endpoint == 0 ? edge.r : edge.g
//                pos = texelFetch(posTex, idx2(nodeIdx))
//
// The CPU loops over 73K nodes and 416K edges that used to run every
// frame are gone. Position updates cost one Float32Array copy + one
// texture upload (~1 MB) per worker TICK, not per frame.

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type { BinaryGraph, BinaryNode } from "../lib/api";
import type { CycleEvent } from "../lib/types";

// Vite ?worker import — bundles + spawns the file as a Worker.
import ForceWorker from "../workers/forceWorker.ts?worker";

type Props = {
  graph: BinaryGraph | null;
  lastCycle: CycleEvent | null;
  consolidationActive: boolean;
};

// ---------------------------------------------------------------------
// Shaders (GLSL ES 3.00 — uses gl_InstanceID, gl_VertexID, texelFetch).
// ---------------------------------------------------------------------

const NODE_VERT = /* glsl */ `
precision highp float;
precision highp sampler2D;

uniform sampler2D posTex;
uniform sampler2D stateTex;
uniform int       texW;          // shared width (posTex == stateTex grid)
uniform float     time;

out vec3  vColor;
out float vActivation;
out vec3  vWorldPos;

vec3 hsl2rgb(vec3 hsl) {
  float h = hsl.x; float s = hsl.y; float l = hsl.z;
  vec3 rgb = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
  return l + s * (rgb - 0.5) * (1.0 - abs(2.0 * l - 1.0));
}

ivec2 idx2(int i) {
  return ivec2(i % texW, i / texW);
}

void main() {
  int  iid     = gl_InstanceID;
  vec4 posTexel  = texelFetch(posTex,   idx2(iid), 0);
  vec4 stateT    = texelFetch(stateTex, idx2(iid), 0);

  float activation = stateT.r;     // [0,1], from active_set or breathing
  float arousal    = stateT.g;     // affect proxy
  float actCount   = stateT.b;     // total activations, log-normalized
  float isPinned   = stateT.a;

  // breathing: only "warm" recently-active nodes pulse fast
  float breathFreq = 0.6 + activation * 2.0;
  float breathAmp  = 0.04 + activation * 0.16;
  float scale      = 1.0 + breathAmp * sin(time * breathFreq + float(iid) * 0.31);

  // base radius — pinned dots are bigger; abstractions read activation
  float baseR = (isPinned > 0.5)
    ? 1.8
    : (1.0 + actCount * 1.5 + activation * 0.8);
  scale *= baseR;

  // color: blue (cool) → orange (active)
  float hue = (220.0 - 140.0 * activation) / 360.0;
  float sat = 0.30 + 0.55 * arousal;
  float lit = 0.45 + 0.25 * actCount + 0.10 * activation;
  vColor       = hsl2rgb(vec3(hue, sat, lit));
  vActivation  = activation;

  vec3 worldPos = posTexel.xyz + position * scale;
  vWorldPos = worldPos;
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(worldPos, 1.0);
}
`;

const NODE_FRAG = /* glsl */ `
precision highp float;
in vec3  vColor;
in float vActivation;
in vec3  vWorldPos;
out vec4 fragColor;
void main() {
  // simple cheap shading: head-light from camera
  float ndotL = clamp(0.55 + 0.55 * normalize(vWorldPos).z, 0.4, 1.0);
  vec3 c = vColor * ndotL;
  // a small emissive bump for active nodes
  c += vColor * vActivation * 0.35;
  fragColor = vec4(c, 1.0);
}
`;

const EDGE_VERT = /* glsl */ `
precision highp float;
precision highp sampler2D;

uniform sampler2D posTex;
uniform sampler2D edgeTex;
uniform sampler2D stateTex;
uniform int posTexW;
uniform int edgeTexW;

flat out vec3  vColor;
flat out float vAlpha;

ivec2 idx2(int i, int w) { return ivec2(i % w, i / w); }

// 10 edge types, 0..9 stored as type_norm = idx / 10
const vec3 EDGE_PALETTE[10] = vec3[10](
  vec3(0.65, 0.55, 0.98),  // 0 is_a
  vec3(0.38, 0.65, 0.98),  // 1 has_property
  vec3(0.97, 0.45, 0.45),  // 2 causes
  vec3(0.98, 0.57, 0.24),  // 3 precedes
  vec3(0.20, 0.83, 0.60),  // 4 similar_to
  vec3(0.96, 0.45, 0.71),  // 5 opposite_of
  vec3(0.58, 0.64, 0.72),  // 6 context_of
  vec3(0.98, 0.75, 0.14),  // 7 refers_to
  vec3(0.13, 0.83, 0.93),  // 8 expresses
  vec3(0.75, 0.52, 0.99)   // 9 part_of
);

void main() {
  int edgeIdx  = gl_VertexID / 2;
  int endpoint = gl_VertexID & 1;

  vec4 e = texelFetch(edgeTex, idx2(edgeIdx, edgeTexW), 0);
  int nodeIdx = (endpoint == 0) ? int(e.r + 0.5) : int(e.g + 0.5);
  vec4 nodePos = texelFetch(posTex, idx2(nodeIdx, posTexW), 0);

  // edge alpha rises with edge weight; recently-active endpoints pop
  float w = clamp(e.b, 0.0, 1.0);
  vec4 ns = texelFetch(stateTex, idx2(nodeIdx, posTexW), 0);
  float activation = ns.r;
  vAlpha = clamp(0.10 + 0.55 * w + 0.40 * activation, 0.06, 0.95);

  int t = clamp(int(e.a * 10.0 + 0.5), 0, 9);
  vColor = EDGE_PALETTE[t];

  gl_Position = projectionMatrix * viewMatrix * modelMatrix
                * vec4(nodePos.xyz, 1.0);
}
`;

const EDGE_FRAG = /* glsl */ `
precision highp float;
flat in vec3  vColor;
flat in float vAlpha;
out vec4 fragColor;
void main() {
  fragColor = vec4(vColor, vAlpha);
}
`;

// ---------------------------------------------------------------------

function sqW(n: number): number {
  return Math.max(1, Math.ceil(Math.sqrt(n)));
}

export function MindGraph({ graph, lastCycle, consolidationActive }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [hoverInfo, setHoverInfo] = useState<{
    x: number; y: number; node: BinaryNode | null;
  } | null>(null);

  const sceneRef = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;

    // node state
    posData:  Float32Array;
    posTex:   THREE.DataTexture;
    stateData: Float32Array;
    stateTex:  THREE.DataTexture;
    nodeTexW: number;
    nodeMesh: THREE.InstancedMesh | null;
    nodeMat:  THREE.ShaderMaterial | null;

    // edge state
    edgeData: Float32Array;
    edgeTex:  THREE.DataTexture;
    edgeTexW: number;
    edgeMesh: THREE.LineSegments | null;
    edgeMat:  THREE.ShaderMaterial | null;

    // graph contents (kept on main thread for hover only)
    nodes:  BinaryNode[];
    nodeIdById: Map<number, number>;

    // worker
    worker: Worker | null;

    raycaster: THREE.Raycaster;
    pointerNDC: THREE.Vector2;
    lastInteractionMs: number;

    timeUniform: { value: number };
  } | null>(null);

  // -----------------------------------------------------------------
  // 1. one-time scene init
  // -----------------------------------------------------------------
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth || 1;
    const height = container.clientHeight || 1;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(60, width / height, 0.5, 8000);
    camera.position.set(0, 0, 600);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const raycaster = new THREE.Raycaster();
    const pointerNDC = new THREE.Vector2(2, 2);

    sceneRef.current = {
      renderer, scene, camera, controls,
      posData:  new Float32Array(0),
      posTex:   new THREE.DataTexture(new Float32Array(4), 1, 1, THREE.RGBAFormat, THREE.FloatType),
      stateData: new Float32Array(0),
      stateTex: new THREE.DataTexture(new Float32Array(4), 1, 1, THREE.RGBAFormat, THREE.FloatType),
      nodeTexW: 1,
      nodeMesh: null,
      nodeMat:  null,
      edgeData: new Float32Array(0),
      edgeTex:  new THREE.DataTexture(new Float32Array(4), 1, 1, THREE.RGBAFormat, THREE.FloatType),
      edgeTexW: 1,
      edgeMesh: null,
      edgeMat:  null,
      nodes: [],
      nodeIdById: new Map(),
      worker: null,
      raycaster,
      pointerNDC,
      lastInteractionMs: performance.now(),
      timeUniform: { value: 0 },
    };

    // ---- pointer hover ----
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

    // ---- main render loop (pure GPU) ----
    let raf = 0;
    const ORBIT_IDLE_MS = 8000;
    const tick = () => {
      const ref = sceneRef.current;
      if (!ref) return;

      const nowMs = performance.now();
      ref.timeUniform.value = nowMs / 1000;

      // idle camera orbit (visual signature kept from old build)
      if (nowMs - ref.lastInteractionMs > ORBIT_IDLE_MS) {
        const cam = ref.camera;
        const r = Math.hypot(cam.position.x, cam.position.z);
        const a = Math.atan2(cam.position.z, cam.position.x) + 0.0008;
        cam.position.x = Math.cos(a) * r;
        cam.position.z = Math.sin(a) * r;
        cam.lookAt(0, 0, 0);
      }

      controls.update();

      // Cheap hover. Only run a raycast every ~80 ms when the user
      // is actively moving — the raycast against a 73K-instance mesh
      // is the one main-thread cost we still pay.
      if (
        nowMs - ref.lastInteractionMs < 1500 &&
        ref.nodeMesh && (nowMs % 80) < 16
      ) {
        ref.raycaster.setFromCamera(ref.pointerNDC, ref.camera);
        const hits = ref.raycaster.intersectObject(ref.nodeMesh, false);
        if (hits.length > 0 && hits[0].instanceId !== undefined) {
          const idx = hits[0].instanceId;
          const node = ref.nodes[idx];
          if (node) {
            const tmp = new THREE.Vector3(
              ref.posData[idx * 4],
              ref.posData[idx * 4 + 1],
              ref.posData[idx * 4 + 2],
            ).project(ref.camera);
            const rect = renderer.domElement.getBoundingClientRect();
            const sx = (tmp.x * 0.5 + 0.5) * rect.width + rect.left;
            const sy = (-tmp.y * 0.5 + 0.5) * rect.height + rect.top;
            setHoverInfo({ x: sx, y: sy, node });
          }
        } else {
          setHoverInfo(null);
        }
      }

      renderer.render(ref.scene, ref.camera);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      const ref = sceneRef.current;
      if (ref?.worker) {
        ref.worker.postMessage({ type: "STOP" });
        ref.worker.terminate();
      }
      try { container.removeChild(renderer.domElement); } catch {}
      renderer.dispose();
      ref?.posTex.dispose();
      ref?.stateTex.dispose();
      ref?.edgeTex.dispose();
      ref?.nodeMat?.dispose();
      ref?.edgeMat?.dispose();
      sceneRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // -----------------------------------------------------------------
  // 2. graph load — build textures + mesh + spawn worker
  // -----------------------------------------------------------------
  useEffect(() => {
    const ref = sceneRef.current;
    if (!ref || !graph) return;

    const N = graph.nodes.length;
    const E = graph.edges.length;
    const nodeTexW = sqW(N);
    const nodeTexH = Math.max(1, Math.ceil(N / nodeTexW));
    const edgeTexW = sqW(E);
    const edgeTexH = Math.max(1, Math.ceil(E / edgeTexW));

    // -- POSITION texture (filled by worker TICKs) -----------------
    const posData = new Float32Array(nodeTexW * nodeTexH * 4);
    // seed from embedding[0..2] so the first frame isn't a black screen
    for (let i = 0; i < N; i++) {
      const n = graph.nodes[i];
      posData[i * 4]     = n.ex * 200;
      posData[i * 4 + 1] = n.ey * 200;
      posData[i * 4 + 2] = n.ez * 200;
      posData[i * 4 + 3] = 0;
    }
    const posTex = new THREE.DataTexture(
      posData, nodeTexW, nodeTexH, THREE.RGBAFormat, THREE.FloatType,
    );
    posTex.minFilter = THREE.NearestFilter;
    posTex.magFilter = THREE.NearestFilter;
    posTex.wrapS = THREE.ClampToEdgeWrapping;
    posTex.wrapT = THREE.ClampToEdgeWrapping;
    posTex.internalFormat = "RGBA32F";
    posTex.type = THREE.FloatType;
    posTex.needsUpdate = true;

    // -- STATE texture (activation, arousal, act_count, pinned) ----
    const stateData = new Float32Array(nodeTexW * nodeTexH * 4);
    for (let i = 0; i < N; i++) {
      const n = graph.nodes[i];
      stateData[i * 4]     = 0;             // current activation (filled on cycle event)
      stateData[i * 4 + 1] = n.arousal;     // affect proxy
      stateData[i * 4 + 2] = n.activation;  // historical activation_count, normalized
      stateData[i * 4 + 3] = 0;             // pinned (no signal yet from binary)
    }
    const stateTex = new THREE.DataTexture(
      stateData, nodeTexW, nodeTexH, THREE.RGBAFormat, THREE.FloatType,
    );
    stateTex.minFilter = THREE.NearestFilter;
    stateTex.magFilter = THREE.NearestFilter;
    stateTex.internalFormat = "RGBA32F";
    stateTex.needsUpdate = true;

    // -- EDGE texture (src_idx, tgt_idx, weight, type_norm) --------
    const edgeData = new Float32Array(edgeTexW * edgeTexH * 4);
    for (let i = 0; i < E; i++) {
      const e = graph.edges[i];
      edgeData[i * 4]     = e.sourceIdx;
      edgeData[i * 4 + 1] = e.targetIdx;
      edgeData[i * 4 + 2] = e.weight;
      edgeData[i * 4 + 3] = e.typeNorm;
    }
    const edgeTex = new THREE.DataTexture(
      edgeData, edgeTexW, edgeTexH, THREE.RGBAFormat, THREE.FloatType,
    );
    edgeTex.minFilter = THREE.NearestFilter;
    edgeTex.magFilter = THREE.NearestFilter;
    edgeTex.internalFormat = "RGBA32F";
    edgeTex.needsUpdate = true;

    // dispose any previous textures + meshes
    ref.posTex.dispose();
    ref.stateTex.dispose();
    ref.edgeTex.dispose();
    if (ref.nodeMesh) ref.scene.remove(ref.nodeMesh);
    if (ref.edgeMesh) ref.scene.remove(ref.edgeMesh);
    ref.nodeMat?.dispose();
    ref.edgeMat?.dispose();

    ref.posData   = posData;
    ref.posTex    = posTex;
    ref.stateData = stateData;
    ref.stateTex  = stateTex;
    ref.edgeData  = edgeData;
    ref.edgeTex   = edgeTex;
    ref.nodeTexW  = nodeTexW;
    ref.edgeTexW  = edgeTexW;
    ref.nodes     = graph.nodes;
    ref.nodeIdById = new Map(graph.nodes.map((n, i) => [n.id, i] as const));

    // -- node mesh: instanced low-poly sphere with custom shader ---
    const sphereGeom = new THREE.SphereGeometry(1, 8, 6);
    const nodeMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: NODE_VERT,
      fragmentShader: NODE_FRAG,
      uniforms: {
        posTex:   { value: posTex },
        stateTex: { value: stateTex },
        texW:     { value: nodeTexW },
        time:     ref.timeUniform,
      },
    });
    const nodeMesh = new THREE.InstancedMesh(sphereGeom, nodeMat, N);
    // we don't use instanceMatrix at all — shader reads pos from texture.
    // But Three still wants a non-zero count to draw.
    nodeMesh.frustumCulled = false;
    ref.scene.add(nodeMesh);
    ref.nodeMesh = nodeMesh;
    ref.nodeMat  = nodeMat;

    // -- edge mesh: LineSegments with dummy positions, shader-resolved
    const edgeGeom = new THREE.BufferGeometry();
    // 2 vertices per edge; gl_VertexID drives everything.
    const dummyPos = new Float32Array(E * 2 * 3);
    edgeGeom.setAttribute(
      "position",
      new THREE.BufferAttribute(dummyPos, 3),
    );
    edgeGeom.setDrawRange(0, E * 2);
    const edgeMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: EDGE_VERT,
      fragmentShader: EDGE_FRAG,
      uniforms: {
        posTex:    { value: posTex },
        edgeTex:   { value: edgeTex },
        stateTex:  { value: stateTex },
        posTexW:   { value: nodeTexW },
        edgeTexW:  { value: edgeTexW },
      },
      transparent: true,
      depthWrite: false,
    });
    const edgeMesh = new THREE.LineSegments(edgeGeom, edgeMat);
    edgeMesh.frustumCulled = false;
    ref.scene.add(edgeMesh);
    ref.edgeMesh = edgeMesh;
    ref.edgeMat  = edgeMat;

    // -- spawn worker ----------------------------------------------
    if (ref.worker) ref.worker.terminate();
    const worker = new ForceWorker();
    ref.worker = worker;
    worker.onmessage = (ev: MessageEvent) => {
      const m = ev.data;
      const r = sceneRef.current;
      if (!r) return;
      if (m.type === "TICK") {
        const positions: Float32Array = m.positions;
        // Copy N×3 → posData N×4 (RGBA pad).
        const pd = r.posData;
        for (let i = 0; i < N; i++) {
          pd[i * 4]     = positions[i * 3];
          pd[i * 4 + 1] = positions[i * 3 + 1];
          pd[i * 4 + 2] = positions[i * 3 + 2];
        }
        r.posTex.needsUpdate = true;
      }
    };
    worker.postMessage({
      type: "INIT",
      data: {
        nodes: graph.nodes.map((n) => ({
          id: n.id, ex: n.ex, ey: n.ey, ez: n.ez,
        })),
        // d3-force-link expects {source, target} with the .id() accessor.
        // We pass concept_ids; the worker resolves them through .id().
        links: graph.edges.map((e) => ({
          source: graph.nodes[e.sourceIdx].id,
          target: graph.nodes[e.targetIdx].id,
        })),
      },
    });
  }, [graph]);

  // -----------------------------------------------------------------
  // 3. cycle event → bump active concepts in the state texture
  // -----------------------------------------------------------------
  useEffect(() => {
    const ref = sceneRef.current;
    if (!ref || !lastCycle) return;
    const sd = ref.stateData;
    // decay all current activations slightly (so older highlights fade)
    for (let i = 0; i < ref.nodes.length; i++) {
      sd[i * 4] *= 0.85;
    }
    Object.entries(lastCycle.active_set).forEach(([k, v]) => {
      const id = parseInt(k, 10);
      const idx = ref.nodeIdById.get(id);
      if (idx === undefined) return;
      sd[idx * 4] = Math.min(1, Math.max(sd[idx * 4], v as number));
    });
    ref.stateTex.needsUpdate = true;
    if (ref.worker) ref.worker.postMessage({ type: "REHEAT" });
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
            <b>#{hoverInfo.node.id}</b>
          </div>
          <div style={{ color: "#aaa" }}>
            act:{hoverInfo.node.activation.toFixed(2)} ·
            surp:{hoverInfo.node.surprise.toFixed(2)} ·
            arousal:{hoverInfo.node.arousal.toFixed(2)}
          </div>
        </div>
      )}
    </div>
  );
}
