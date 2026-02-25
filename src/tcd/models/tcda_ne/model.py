import numpy as np
from typing import List, Optional


class TCDA_NE:
    """
    Exact implementation of TCDA-NE (Yuan et al., Mathematics 2025)
    - Uses first-order + second-order proximity (common-neighbor cosine)
    - Convex-style NMF with multiplicative updates
    - Evolutionary clustering via temporal smoothness
    """

    def __init__(
        self,
        num_communities: int = 20,      # K in the paper
        alpha: float = 0.8,             # temporal smoothness weight (paper best ≈ 0.8)
        eta: float = 5.0,               # first vs second-order weight (paper best = 5)
        max_iter: int = 200,            # inner iterations per snapshot
        tol: float = 1e-50,              # convergence tolerance
        random_state: Optional[int] = 42,
    ):
        self.K = num_communities
        self.alpha = alpha
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)

        # Stored after fit()
        self.S_list: List[np.ndarray] = []   # proximity matrices
        self.W_list: List[np.ndarray] = []   # weight matrices (centroids)
        self.G_list: List[np.ndarray] = []   # community membership matrices

    def _compute_S(self, X: np.ndarray) -> np.ndarray:
        """S = S¹ + η·S²"""
        S1 = X.astype(np.float64).copy()  # first-order (adjacency)

        # Second-order: cosine similarity of neighbor vectors
        norms = np.linalg.norm(S1, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        S2 = (S1 @ S1.T) / (norms @ norms.T)

        return S1 + self.eta * S2

    def _update_w(self, S: np.ndarray, G: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Multiplicative update for W"""
        # num = (Sᵀ S G)
        num = S.T @ (S @ G)
        # denom = (Sᵀ S W Gᵀ G)
        denom = S.T @ (S @ W @ (G.T @ G))

        W = W * (num / (denom + 1e-12))

        # Row normalization: ∑ⱼ W_{ij} = 1 ∀i (Algorithm 1, line 5)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-12] = 1.0
        W /= row_sums
        return W

    def _update_g(
        self,
        S: np.ndarray,
        W: np.ndarray,
        G: np.ndarray,
        S_prev: Optional[np.ndarray] = None,
        W_prev: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Multiplicative update for G (exact Equation (14) in the paper)"""
        # Base term: Sᵀ S W
        num = S.T @ (S @ W)
        denom = G @ (W.T @ S.T @ (S @ W))

        # Temporal smoothness term (α)
        if S_prev is not None and W_prev is not None and self.alpha > 0:
            num += self.alpha * (S_prev.T @ (S_prev @ W_prev))
            denom += self.alpha * (G @ (W_prev.T @ S_prev.T @ (S_prev @ W_prev)))

        G = G * (num / (denom + 1e-12))
        return G

    def fit(self, snapshots: List[np.ndarray]) -> "TCDA_NE":
        """
        Fit TCDA-NE on a list of adjacency matrices (one per time snapshot).
        """
        if len(snapshots) == 0:
            raise ValueError("snapshots list cannot be empty")

        print(f"TCDA-NE: Computing proximity matrices for {len(snapshots)} snapshots...")
        self.S_list = [self._compute_S(X) for X in snapshots]

        for t, S in enumerate(self.S_list):
            print(f"Fitting snapshot {t+1}/{len(self.S_list)}...")
            N = S.shape[0]

            # Random initialization (paper does this independently for every snapshot)
            W = self.rng.random((N, self.K))
            W /= (W.sum(axis=1, keepdims=True) + 1e-12)

            G = self.rng.random((N, self.K))

            historical_err = 0.0
            if t > 0:
                S_prev = self.S_list[t - 1]
                W_prev = self.W_list[t - 1]
                historical_err = np.linalg.norm(S_prev - S_prev @ W_prev @ G.T, ord="fro") ** 2

            prev_obj = np.inf
            for it in range(self.max_iter):
                # Update W
                W = self._update_w(S, G, W)

                # Update G (with temporal term only from t ≥ 1)
                if t == 0 or self.alpha <= 0:
                    G = self._update_g(S, W, G)
                else:
                    G = self._update_g(
                        S, W, G,
                        self.S_list[t - 1],
                        self.W_list[t - 1],
                    )

                # Convergence check on current snapshot reconstruction
                recon = S @ W @ G.T
                obj = np.linalg.norm(S - recon, ord="fro") ** 2
                obj += self.alpha * historical_err
                if it > 0 and abs(prev_obj - obj) / (prev_obj + 1e-12) < self.tol:
                    print(f"Snapshot {t} converged after {it} iterations (obj={obj:.2f})")
                    break
                prev_obj = obj

            self.W_list.append(W)
            self.G_list.append(G)

        print(f"TCDA-NE fitted on {len(snapshots)} snapshots (K={self.K}, α={self.alpha}, η={self.eta})")
        return self

    def get_communities(self, t: int) -> np.ndarray:
        """Hard community assignment for snapshot t (0-based, supports negative indexing)"""
        if t < 0:
            t = len(self.G_list) + t
        return np.argmax(self.G_list[t], axis=1)

    def get_membership_matrix(self, t: int) -> np.ndarray:
        """Soft membership matrix G_t (N × K)"""
        if t < 0:
            t = len(self.G_list) + t
        return self.G_list[t].copy()

    def get_num_snapshots(self) -> int:
        return len(self.G_list)
