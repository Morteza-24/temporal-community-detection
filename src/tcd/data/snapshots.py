import numpy as np
import pandas as pd
from torch_geometric.data import TemporalData
from typing import List


def temporal_to_snapshots(
    data: TemporalData,
    freq: str = "M",          # pandas freq: "D", "W", "M", "Q", "Y" ...
    undirected: bool = True,
    min_edges_per_snapshot: int = 0,
) -> List[np.ndarray]:
    """
    Convert your continuous TemporalData into a list of adjacency matrices (one per time bin).
    """
    if len(data.t) == 0:
        raise ValueError("Empty TemporalData")

    N = data.num_nodes

    df = pd.DataFrame({
        "src": data.src.numpy(),
        "dst": data.dst.numpy(),
        "ts": pd.to_datetime(data.t.numpy(), unit="s")
    })

    df["bin"] = df["ts"].dt.to_period(freq)

    snapshots: List[np.ndarray] = []
    for _, group in df.groupby("bin", sort=True):
        if len(group) < min_edges_per_snapshot:
            continue

        adj = np.zeros((N, N), dtype=np.float32)

        for s, d in zip(group["src"].astype(int), group["dst"].astype(int)):
            adj[s, d] = 1.0
            if undirected:
                adj[d, s] = 1.0

        np.fill_diagonal(adj, 0)
        snapshots.append(adj)

    return snapshots
