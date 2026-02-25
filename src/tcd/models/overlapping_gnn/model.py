from . import nocd
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


class OverlappingGNN:
    """
    Overlapping Community Detection with Graph Neural Networks
    - Stores soft memberships Z per snapshot
    """

    def __init__(
        self,
        num_communities: int = 20,      # K
        threshold: float = 0.5,         # membership threshold for hard clusters
        hidden_sizes: list = None,
        weight_decay: float = 1e-2,
        dropout: float = 0.5,
        batch_norm: bool = True,
        lr: float = 1e-3,
        max_epochs: int = 500,
        balance_loss: bool = True,
        stochastic_loss: bool = True,
        batch_size: int = 20000,
        random_state: int = 42,
    ):
        if hidden_sizes is None:
            hidden_sizes = [128]

        self.K = num_communities
        self.threshold = threshold
        self.hidden_sizes = hidden_sizes
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.lr = lr
        self.max_epochs = max_epochs
        self.balance_loss = balance_loss
        self.stochastic_loss = stochastic_loss
        self.batch_size = batch_size

        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        np.random.seed(random_state)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)

        self.memberships_list: list[np.ndarray] = []   # soft Z per snapshot

    def fit(self, snapshots: list[np.ndarray]) -> "OverlappingGNN":
        """Run NOCD on each snapshot independently (temporal by snapshot)."""
        print(f"[OverlappingGNN] Training on {len(snapshots)} snapshots | device={self.device}")

        for t, adj_dense in enumerate(snapshots):
            print(f"  → Snapshot {t+1}/{len(snapshots)} ({adj_dense.shape[0]} nodes)")

            A = sp.csr_matrix(adj_dense)
            N = A.shape[0]

            # Node features: identity (standard when no attributes)
            X = sp.eye(N, format="csr")
            x_norm = sp.hstack([X, A])
            x_norm = nocd.utils.to_sparse_tensor(x_norm).to(self.device)

            # GNN + Decoder
            gnn = nocd.nn.GCN(
                x_norm.shape[1],
                self.hidden_sizes,
                self.K,
                dropout=self.dropout,
                batch_norm=self.batch_norm,
            ).to(self.device)

            adj_norm = gnn.normalize_adj(A)
            decoder = nocd.nn.BerpoDecoder(N, A.nnz, balance_loss=self.balance_loss)
            opt = torch.optim.Adam(gnn.parameters(), lr=self.lr)

            sampler = nocd.sampler.get_edge_sampler(A, self.batch_size, self.batch_size, num_workers=0, device=self.device)

            val_loss = np.inf
            early_stopping = nocd.train.NoImprovementStopping(lambda: val_loss, patience=10)
            model_saver = nocd.train.ModelSaver(gnn)

            for epoch, batch in enumerate(sampler):
                if epoch > self.max_epochs:
                    break

                if epoch % 25 == 0:
                    with torch.no_grad():
                        gnn.eval()
                        Z = F.relu(gnn(x_norm, adj_norm))
                        val_loss = decoder.loss_full(Z, A)
                        print(f"    Epoch {epoch:4d} | loss.full = {val_loss:.4f}")
                        early_stopping.next_step()
                        if early_stopping.should_save():
                            model_saver.save()
                        if early_stopping.should_stop():
                            break

                # Training step
                gnn.train()
                opt.zero_grad()
                Z = F.relu(gnn(x_norm, adj_norm))
                ones_idx, zeros_idx = batch
                loss = decoder.loss_batch(Z, ones_idx, zeros_idx) if self.stochastic_loss else decoder.loss_full(Z, A)
                loss += nocd.utils.l2_reg_loss(gnn, scale=self.weight_decay)
                loss.backward()
                opt.step()

            # Final inference + normalization
            with torch.no_grad():
                gnn.eval()
                Z = F.relu(gnn(x_norm, adj_norm))
                Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-8)
                self.memberships_list.append(Z.cpu().detach().numpy())

        print(f"[OverlappingGNN] Finished — soft memberships ready for {len(snapshots)} snapshots")
        return self

    def get_soft_memberships(self, t: int) -> np.ndarray:
        """Soft membership matrix Z (N x K) for snapshot t"""
        if t < 0:
            t = len(self.memberships_list) + t
        return self.memberships_list[t].copy()

    def get_communities(self, t: int, threshold: float | None = None) -> np.ndarray:
        """Hard disjoint communities via argmax (for compatibility with consecutive_nmi)"""
        if threshold is None:
            threshold = self.threshold
        Z = self.get_soft_memberships(t)
        return np.argmax(Z, axis=1)

    def get_num_snapshots(self) -> int:
        """Number of snapshots the model was trained on"""
        return len(self.memberships_list)

    def get_overlapping_clusters(self, t: int, threshold: float | None = None) -> list[set]:
        """List of sets — each node belongs to zero or more communities"""
        if threshold is None:
            threshold = self.threshold
        Z = self.get_soft_memberships(t)
        memberships = Z.copy()

        for col in range(memberships.shape[1]):
            memberships[:, col][memberships[:, col] < threshold] = -1
            memberships[:, col][memberships[:, col] >= threshold] = col

        clusters = []
        for row in memberships.astype(int):
            s = set(row)
            if len(s) > 1:
                s.discard(-1)
            clusters.append(s)
        return clusters
