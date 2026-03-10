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

    @property
    def device(self) -> str:
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
