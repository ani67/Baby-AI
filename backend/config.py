import os
from dataclasses import dataclass


@dataclass
class Config:
    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Ollama
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    teacher_model: str = os.environ.get("TEACHER_MODEL", "llava")

    # Database
    db_path: str = os.environ.get("DB_PATH", "data/dev.db")

    # Data
    data_dir: str = os.environ.get("DATA_DIR", "data")

    # Model
    initial_clusters: int = 4
    nodes_per_cluster: int = 8

    # Viz
    snapshot_interval: int = 50
    projection_interval: int = 10

    # Inhibition
    inhibition_radius: float = 0.92       # cosine similarity threshold (very similar only)
    suppression_factor: float = 0.5       # halve suppressed activations (not obliterate)

    # Resonance
    resonance_threshold: float = 0.02     # min cosine sim to input for cluster to participate

    # Memory buffer — decaying echo of recent activations
    buffer_decay: float = 0.9             # per-step decay (0.9 = ~10 step half-life)
    buffer_weight: float = 0.15           # how much buffer biases current input
    buffer_top_k: int = 5                 # clusters contributing to echo each step

    # C.1: Per-cluster staged learning signal
    per_cluster_signal: bool = True       # Enable staged per-cluster signal
    per_cluster_global_steps: int = 5000  # Steps 0–5K: 100% global signal (training wheels)
    per_cluster_blend_steps: int = 10000  # Steps 5K–10K: linear blend global→per-cluster
    # After 10K: 100% per-cluster signal

    # C.2: Content-aware routing
    gate_activation_step: int = 2000      # Gates inactive before this (sigmoid≈0.5 is noise)

    # FF Signal Enrichment Experiments (all default OFF for baseline)
    exp_per_cluster_sign: bool = False    # Exp 1 (superseded by C.1 above)
    exp_error_direction: bool = False     # Exp 2: push toward teacher answer, not just input
    exp_contrastive_pairs: bool = False   # Exp 3: rank pairs within batch instead of threshold
    exp_multi_target: bool = False        # Exp 4: additive bonus update toward teacher direction
    exp_structure_reuse: bool = False     # Exp 5: save/load graph topology for warm starts

    @property
    def device(self) -> str:
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
