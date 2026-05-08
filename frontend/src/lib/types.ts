// Mirror of the backend payload shapes. Keep in sync with backend/api.py.

export type Stats = {
  cycle_count: number;
  node_count: number;
  edge_count: number;
  pin_count: number;
  replay_buffer_size: number;
  consolidation_active: boolean;
  composite: number[];
  arousal: number;
};

export type StateSnapshot = Stats & {
  now: number;
  last_observation_t: number;
  agent_count: number;
  observation_count: number;
  surprise_count: number;
  character: number[];
  self_concept_id: number;
  mind_uuid: string;
  top_active_concepts: TopActive[];
  recent_emissions: [number, string, string][];   // (t, modality, surface)
};

export type TopActive = {
  id: number;
  name: string;
  activation_count: number;
  alignment: number;
  arousal: number;
};

export type GraphNode = {
  id: number;
  name: string;
  activation_count: number;
  surprise_at_birth: number;
  alignment: number;     // [-1, 1]
  arousal: number;       // ≥ 0
  is_pinned: boolean;
  last_activated: number;
  created_at: number;
};

export type GraphEdge = {
  source: number;
  target: number;
  type: string;
  weight: number;
  activation_count: number;
};

export type GraphPayload = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  self_concept_id: number;
  now: number;
};

export type ExpressionDecision =
  | { type: "chosen";      surface: string; expression_gap: number; score: number }
  | { type: "revision";    reason: string;  best_gap: number }
  | { type: "suppression"; reason: string;  best_gap: number };

export type CycleEvent = {
  type: "cycle";
  stimulus_id: number;
  now: number;
  active_set: Record<string, number>;
  traversed_edges: { source: number; target: number; type: string; activation: number }[];
  phase: string;
  mode: string;
  arousal: number;
  action: { kind: string; target_concept_id: number | null; score: number };
  expression: ExpressionDecision | null;
  emitted_surface: string | null;
  stats: Stats;
};

export type IdleEvent = {
  type: "idle";
  now: number;
  replayed_count: number;
  stats: Stats;
};

export type SleepEvent = {
  type: "sleep";
  now: number;
  replays_fired: number;
  abstractions_formed: number;
  duration_actual: number;
  stats: Stats;
};

export type StateEvent = { type: "state" } & StateSnapshot;

export type AnyEvent = CycleEvent | IdleEvent | SleepEvent | StateEvent;
