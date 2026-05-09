/// <reference types="vite/client" />

// `?worker` Vite imports — bundles the file as a Web Worker.
declare module "*?worker" {
  const WorkerCtor: {
    new (options?: { name?: string }): Worker;
  };
  export default WorkerCtor;
}

// d3-force-3d ships JS with no types — declare as any so the worker
// can import what it needs without TS bailing.
declare module "d3-force-3d" {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const d3: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const forceSimulation: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const forceManyBody: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const forceLink: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const forceCenter: any;
  export default d3;
}
