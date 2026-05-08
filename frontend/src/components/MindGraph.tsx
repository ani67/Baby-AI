import { useEffect, useRef } from "react";
import ForceGraph3D from "3d-force-graph";
import * as THREE from "three";
import type { GraphPayload, GraphNode, GraphEdge, CycleEvent } from "../lib/types";

type Props = {
  graph: GraphPayload | null;
  /** A monotonically-increasing object that fires on each cycle event so
   *  the renderer can pulse / flash the right nodes and edges. */
  lastCycle: CycleEvent | null;
  consolidationActive: boolean;
};

type Datum = GraphNode & {
  __obj?: THREE.Object3D;          // attached so we can poke it in update loop
  __activatedAt?: number;          // wall-clock seconds; used for breathing
  __pulseUntil?: number;           // ring pulse expiry
};

type LinkDatum = GraphEdge & { __obj?: THREE.Object3D };

const EDGE_COLOR_BY_TYPE: Record<string, string> = {
  is_a:         "#a78bfa",   // violet
  has_property: "#60a5fa",   // blue
  causes:       "#f87171",   // red
  precedes:     "#fb923c",   // orange
  similar_to:   "#34d399",   // green
  opposite_of:  "#f472b6",   // pink
  context_of:   "#94a3b8",   // slate
  refers_to:    "#fbbf24",   // amber
  expresses:    "#22d3ee",   // cyan
  part_of:      "#c084fc",   // purple
};

/** Map alignment ∈ [-1,1] and arousal ∈ [0,2+] to an HSL color.
 *  - alignment=+1 (current felt) → warm hues (red/orange)
 *  - alignment=-1 (counter-felt) → cool hues (blue/cyan)
 *  - arousal scales saturation + lightness (high arousal pops; calm fades)
 */
function nodeColor(d: Datum, age: number, isAbstraction: boolean) {
  // Base hue: alignment=+1 → 20° (warm), alignment=-1 → 220° (cool)
  const a = Math.max(-1, Math.min(1, d.alignment));
  const hue = 220 - 100 * a; // 120° (yellow-green) at zero
  const arousalClamp = Math.min(1, d.arousal / 0.6);
  const sat = 30 + 50 * arousalClamp;
  const light = 50 + 20 * arousalClamp + (isAbstraction ? 8 : 0);
  return `hsl(${hue}, ${sat}%, ${light}%)`;
}

export function MindGraph({ graph, lastCycle, consolidationActive }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const fgRef = useRef<ReturnType<typeof ForceGraph3D> | null>(null);
  const dataRef = useRef<{ nodes: Datum[]; links: LinkDatum[] }>({ nodes: [], links: [] });
  const indexRef = useRef<Map<number, Datum>>(new Map());
  const pulseGroupRef = useRef<THREE.Group | null>(null);
  const lastCycleRef = useRef<CycleEvent | null>(null);

  // Init 3D graph once.
  useEffect(() => {
    if (!containerRef.current) return;
    const fg = ForceGraph3D()(containerRef.current)
      .backgroundColor("#000000")
      .showNavInfo(false)
      .nodeRelSize(4)
      .nodeOpacity(0.92)
      .nodeResolution(16)
      .linkOpacity(0.32)
      .linkWidth((l: LinkDatum) => Math.max(0.25, l.weight * 1.4))
      .linkDirectionalParticles((l: LinkDatum) => Math.min(4, l.activation_count))
      .linkDirectionalParticleSpeed(0.006)
      .linkDirectionalParticleWidth(1.2)
      .linkColor((l: LinkDatum) => EDGE_COLOR_BY_TYPE[l.type] ?? "#94a3b8")
      .nodeLabel((d: Datum) => {
        const valence = d.alignment >= 0 ? "+" : "−";
        return `<div style="background:#000a;border:1px solid #fff2;padding:4px 8px;border-radius:4px;color:#fff;font-family:monospace;font-size:11px;">
          <div><b>#${d.id}</b> ${escapeHtml(d.name) || "<i>(unnamed)</i>"}</div>
          <div style="color:#aaa">activations: ${d.activation_count} · alignment: ${valence}${Math.abs(d.alignment).toFixed(2)}</div>
          ${d.is_pinned ? '<div style="color:#facc15">📌 pinned</div>' : ""}
        </div>`;
      })
      .nodeThreeObject((d: Datum) => {
        const isAbs = d.name.startsWith("abstraction:");
        const baseSize = 2 + Math.log1p(d.activation_count) * 1.6 + (isAbs ? 1.2 : 0);
        const group = new THREE.Group();

        // Inner core (slightly emissive sphere for crispness)
        const coreMat = new THREE.MeshStandardMaterial({
          color: nodeColor(d, 0, isAbs),
          emissive: new THREE.Color(nodeColor(d, 0, isAbs)).multiplyScalar(0.4),
          roughness: 0.3,
          metalness: 0.1,
        });
        const core = new THREE.Mesh(new THREE.SphereGeometry(baseSize, 18, 18), coreMat);
        group.add(core);

        // Outer translucent halo for "alive" glow
        const haloMat = new THREE.MeshBasicMaterial({
          color: nodeColor(d, 0, isAbs),
          transparent: true,
          opacity: 0.18,
          side: THREE.BackSide,
          depthWrite: false,
        });
        const halo = new THREE.Mesh(new THREE.SphereGeometry(baseSize * 1.55, 16, 16), haloMat);
        group.add(halo);

        if (d.is_pinned) {
          // Subtle gold ring around pinned concepts
          const ringGeom = new THREE.RingGeometry(baseSize * 1.7, baseSize * 1.85, 32);
          const ringMat = new THREE.MeshBasicMaterial({
            color: 0xfacc15,
            transparent: true,
            opacity: 0.55,
            side: THREE.DoubleSide,
            depthWrite: false,
          });
          const ring = new THREE.Mesh(ringGeom, ringMat);
          ring.rotation.x = Math.PI / 2;
          group.add(ring);
        }

        d.__obj = group;
        return group;
      });

    // d3 force tweaks: slightly less stiff, room to breathe
    fg.d3Force("charge")?.strength(-180);
    fg.d3Force("link")?.distance(70);

    // Add a group for transient pulse rings
    const pulseGroup = new THREE.Group();
    fg.scene().add(pulseGroup);
    pulseGroupRef.current = pulseGroup;

    fgRef.current = fg;

    // Continuous render-time animations: breathing on active nodes,
    // pulse ring lifecycle, halo opacity tied to recent activity.
    let raf = 0;
    const animate = () => {
      const tNow = performance.now() / 1000;
      // Breathing on every node, scaled by recent activity
      indexRef.current.forEach((d) => {
        const obj = d.__obj;
        if (!obj) return;
        const sinceAct = d.last_activated > 0 ? Math.max(0, (Date.now() / 1000) - d.last_activated) : 999;
        const recencyBoost = Math.exp(-sinceAct / 6);   // ~6s decay
        const phase = ((d.id * 0.31) + tNow * (0.6 + 0.6 * recencyBoost)) % (Math.PI * 2);
        const breathe = 1 + 0.05 * Math.sin(phase) + 0.18 * recencyBoost;
        obj.scale.set(breathe, breathe, breathe);
      });

      // Pulse rings: fade out and remove finished
      const pg = pulseGroupRef.current;
      if (pg) {
        const remove: THREE.Object3D[] = [];
        pg.children.forEach((ring) => {
          const meta = (ring as any).__meta as { start: number; duration: number; baseSize: number } | undefined;
          if (!meta) return;
          const t = (performance.now() - meta.start) / meta.duration;
          if (t >= 1) { remove.push(ring); return; }
          const s = 1 + t * 4.5;
          ring.scale.set(s, s, s);
          const mat = (ring as THREE.Mesh).material as THREE.MeshBasicMaterial;
          mat.opacity = 0.6 * (1 - t);
        });
        remove.forEach((r) => pg.remove(r));
      }

      raf = requestAnimationFrame(animate);
    };
    raf = requestAnimationFrame(animate);

    // Slow camera orbit while idle gives the scene a sense of life even
    // when no events are firing. We disable it on user interaction by
    // tracking the last interaction timestamp.
    let lastInteraction = performance.now();
    const controls = (fg.controls() as any);
    if (controls?.addEventListener) {
      controls.addEventListener("change", () => { lastInteraction = performance.now(); });
    }
    const orbitInterval = window.setInterval(() => {
      if (performance.now() - lastInteraction < 8000) return;
      const cam = fg.camera();
      const r = Math.hypot(cam.position.x, cam.position.z);
      const a = Math.atan2(cam.position.z, cam.position.x) + 0.0008;
      cam.position.x = Math.cos(a) * r;
      cam.position.z = Math.sin(a) * r;
      cam.lookAt(0, 0, 0);
    }, 16);

    return () => {
      cancelAnimationFrame(raf);
      window.clearInterval(orbitInterval);
      try { fg._destructor(); } catch {}
      fgRef.current = null;
    };
  }, []);

  // Push graph data updates.
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg || !graph) return;

    const incomingNodes = graph.nodes.map((n) => ({ ...n })) as Datum[];
    const incomingLinks = graph.edges.map((e) => ({ ...e })) as LinkDatum[];

    // Preserve x/y/z positions across updates so the graph doesn't jump.
    const prevById = indexRef.current;
    incomingNodes.forEach((d) => {
      const old = prevById.get(d.id) as any;
      if (old && old.x !== undefined) {
        (d as any).x = old.x; (d as any).y = old.y; (d as any).z = old.z;
        d.__obj = old.__obj; d.__activatedAt = old.__activatedAt;
      }
    });

    const newIndex = new Map<number, Datum>();
    incomingNodes.forEach((d) => newIndex.set(d.id, d));
    indexRef.current = newIndex;

    dataRef.current = { nodes: incomingNodes, links: incomingLinks };
    fg.graphData(dataRef.current);
  }, [graph]);

  // React to cycle events: pulse ring at active concepts, brief edge flashes.
  useEffect(() => {
    if (!lastCycle || lastCycle === lastCycleRef.current) return;
    lastCycleRef.current = lastCycle;
    const fg = fgRef.current;
    const pg = pulseGroupRef.current;
    if (!fg || !pg) return;

    // Mark activated nodes so the breathing animation amplifies.
    const tNow = Date.now() / 1000;
    Object.keys(lastCycle.active_set).forEach((idStr) => {
      const id = parseInt(idStr, 10);
      const d = indexRef.current.get(id);
      if (d) d.last_activated = tNow;
    });

    // Pulse ring at the strongest seed concept.
    const strongest = Object.entries(lastCycle.active_set)
      .map(([k, v]) => [parseInt(k, 10), v as number] as const)
      .sort((a, b) => b[1] - a[1])[0];
    if (strongest) {
      const d = indexRef.current.get(strongest[0]) as any;
      if (d && d.x !== undefined && d.__obj) {
        const ringGeom = new THREE.RingGeometry(2.5, 3.0, 48);
        const ringMat = new THREE.MeshBasicMaterial({
          color: 0xffffff, transparent: true, opacity: 0.6, side: THREE.DoubleSide, depthWrite: false,
        });
        const ring = new THREE.Mesh(ringGeom, ringMat);
        ring.position.set(d.x, d.y, d.z);
        ring.lookAt(fg.camera().position);
        (ring as any).__meta = { start: performance.now(), duration: 1300, baseSize: 1 };
        pg.add(ring);
      }
    }
  }, [lastCycle]);

  return (
    <div
      ref={containerRef}
      className={`absolute inset-0 ${consolidationActive ? "consolidating" : ""}`}
    />
  );
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => {
    return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" } as any)[c];
  });
}
