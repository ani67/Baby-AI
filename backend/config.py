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

    @property
    def device(self) -> str:
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
