/// <reference lib="webworker" />
/**
 * d3-force-3d in a Web Worker.
 *
 * The main thread no longer runs the simulation. It posts an INIT
 * message with nodes + links, the worker runs ticks, and after each
 * tick it posts a Float32Array of positions back as a transferable —
 * zero-copy. The render loop on the main thread reads those positions
 * straight into a DataTexture and the GPU does the rest.
 *
 * Messages
 *   in : { type: "INIT", data: { nodes: {id}[], links: {source,target}[] } }
 *   in : { type: "REHEAT" }            — restart with alpha=0.1
 *   in : { type: "PIN", id }           — fix this node at its current pos
 *   in : { type: "UNPIN", id }
 *   in : { type: "STOP" }
 *   out: { type: "TICK", positions: Float32Array(N*3), alpha: number }
 *   out: { type: "READY", count: N }
 *   out: { type: "END" }
 */
import {
  forceCenter,
  forceLink,
  forceManyBody,
  forceSimulation,
  // @ts-ignore — d3-force-3d ships .js with a barebones .d.ts
} from "d3-force-3d";

type SimNode = {
  id: number;
  x?: number; y?: number; z?: number;
  vx?: number; vy?: number; vz?: number;
  fx?: number | null; fy?: number | null; fz?: number | null;
};
type SimLink = { source: number; target: number };

let simulation: any = null;
let nodes: SimNode[] = [];
let links: SimLink[] = [];

function postPositions() {
  const positions = new Float32Array(nodes.length * 3);
  for (let i = 0; i < nodes.length; i++) {
    positions[i * 3]     = nodes[i].x ?? 0;
    positions[i * 3 + 1] = nodes[i].y ?? 0;
    positions[i * 3 + 2] = nodes[i].z ?? 0;
  }
  // Transfer the buffer — main thread receives it without a copy.
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(
    {
      type: "TICK",
      positions,
      alpha: simulation ? simulation.alpha() : 0,
    },
    [positions.buffer],
  );
}

self.onmessage = (e: MessageEvent) => {
  const msg = e.data;
  if (!msg || typeof msg !== "object") return;

  if (msg.type === "INIT") {
    // Seed with deterministic positions on a sphere so the layout
    // settles fast even at N=73K. Using `embedding[0..2]` from the
    // binary endpoint as initial position gives semantically-clustered
    // starting points (similar concepts begin near each other).
    const incoming: { id: number; ex: number; ey: number; ez: number }[] =
      msg.data.nodes;
    nodes = incoming.map((n) => ({
      id: n.id,
      // scale up a bit so the charge force has room to push.
      x: (n.ex || 0) * 200,
      y: (n.ey || 0) * 200,
      z: (n.ez || 0) * 200,
    }));
    links = msg.data.links.map((l: any) => ({ source: l.source, target: l.target }));

    simulation = forceSimulation(nodes, 3)
      // weaker charge at scale — N² behavior would dominate otherwise.
      .force("charge", forceManyBody().strength(-30).distanceMax(120))
      .force(
        "link",
        forceLink(links)
          .id((d: any) => d.id)
          .distance(40)
          .strength(0.3),
      )
      .force("center", forceCenter())
      .alphaDecay(0.02)
      .velocityDecay(0.5)
      .on("tick", postPositions)
      .on("end", () => {
        (self as unknown as DedicatedWorkerGlobalScope).postMessage({ type: "END" });
      });
    // Don't call .stop() — let it run.
    (self as unknown as DedicatedWorkerGlobalScope).postMessage({
      type: "READY",
      count: nodes.length,
    });
    return;
  }

  if (msg.type === "REHEAT") {
    if (simulation) simulation.alpha(0.1).restart();
    return;
  }

  if (msg.type === "PIN") {
    const node = nodes.find((n) => n.id === msg.id);
    if (node) {
      node.fx = node.x ?? 0;
      node.fy = node.y ?? 0;
      node.fz = node.z ?? 0;
    }
    return;
  }

  if (msg.type === "UNPIN") {
    const node = nodes.find((n) => n.id === msg.id);
    if (node) {
      node.fx = null;
      node.fy = null;
      node.fz = null;
    }
    return;
  }

  if (msg.type === "STOP") {
    if (simulation) simulation.stop();
    return;
  }
};

export {};
