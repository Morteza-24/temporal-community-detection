import torch
from torch_geometric.data import TemporalData
from pathlib import Path


def load_enron() -> TemporalData:
    path = Path("data/processed/enron_temporal.pt")
    if not path.exists():
        raise FileNotFoundError(
            "Processed Enron data not found. "
            "Run: python scripts/preprocess_enron.py"
        )
    return torch.load(path, weights_only=False)
