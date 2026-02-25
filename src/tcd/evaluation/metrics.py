import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import List, Tuple


def nmi_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Normalized Mutual Information"""
    if len(labels_true) != len(labels_pred):
        raise ValueError("labels_true and labels_pred must have the same length")

    labels_true = np.asarray(labels_true, dtype=int)
    labels_pred = np.asarray(labels_pred, dtype=int)

    max_true = labels_true.max() + 1
    max_pred = labels_pred.max() + 1
    contingency = np.bincount(
        labels_true * max_pred + labels_pred,
        minlength=max_true * max_pred
    ).reshape(max_true, max_pred)

    p_ij = contingency / contingency.sum()
    p_i = p_ij.sum(axis=1)
    p_j = p_ij.sum(axis=0)

    mi = 0.0
    for i in range(max_true):
        for j in range(max_pred):
            if p_ij[i, j] > 0:
                mi += p_ij[i, j] * np.log(p_ij[i, j] / (p_i[i] * p_j[j] + 1e-12))

    h_true = -np.sum(p_i * np.log(p_i + 1e-12))
    h_pred = -np.sum(p_j * np.log(p_j + 1e-12))

    if h_true == 0 or h_pred == 0:
        return 0.0

    return mi / max(h_true, h_pred)


def f1_score_communities(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    F-score for community detection with optimal matching.
    
    For every pair of communities (one from ground truth, one from prediction):
        F1 = 2 * |intersection| / (|gt| + |pred|)
    
    Then we solve the assignment problem (Hungarian algorithm) to get the best one-to-one
    matching and average the F1 scores.
    
    Returns value in [0, 1]. 1.0 = perfect match.
    """
    if len(labels_true) != len(labels_pred):
        raise ValueError("labels_true and labels_pred must have the same length")

    labels_true = np.asarray(labels_true, dtype=int)
    labels_pred = np.asarray(labels_pred, dtype=int)

    # Convert to sets of node indices per community
    comms_true = [set(np.where(labels_true == c)[0]) for c in np.unique(labels_true)]
    comms_pred = [set(np.where(labels_pred == c)[0]) for c in np.unique(labels_pred)]

    if not comms_true or not comms_pred:
        return 0.0

    # Build F1 matrix
    n_true = len(comms_true)
    n_pred = len(comms_pred)
    f1_matrix = np.zeros((n_true, n_pred))

    for i, gt in enumerate(comms_true):
        for j, pr in enumerate(comms_pred):
            inter = len(gt & pr)
            if inter == 0:
                f1_matrix[i, j] = 0.0
            else:
                prec = inter / len(pr)
                rec = inter / len(gt)
                f1_matrix[i, j] = 2 * prec * rec / (prec + rec + 1e-12)

    # Optimal assignment (maximize sum of F1)
    row_ind, col_ind = linear_sum_assignment(-f1_matrix)   # negative because we maximize
    best_f1 = f1_matrix[row_ind, col_ind].mean()

    return float(best_f1)


def consecutive_nmi(model) -> np.ndarray:
    """NMI between every pair of consecutive snapshots."""
    n = model.get_num_snapshots()
    if n < 2:
        return np.array([], dtype=float)

    scores = []
    for t in range(1, n):
        comm_prev = model.get_communities(t - 1)
        comm_curr = model.get_communities(t)
        scores.append(nmi_score(comm_prev, comm_curr))
    return np.array(scores)


def consecutive_f1(model) -> np.ndarray:
    """F1-score between every pair of consecutive snapshots (temporal smoothness)."""
    n = model.get_num_snapshots()
    if n < 2:
        return np.array([], dtype=float)

    scores = []
    for t in range(1, n):
        comm_prev = model.get_communities(t - 1)
        comm_curr = model.get_communities(t)
        scores.append(f1_score_communities(comm_prev, comm_curr))
    return np.array(scores)


def average_nmi_across_snapshots(ground_truth_list: List[np.ndarray],
                                 pred_list: List[np.ndarray]) -> float:
    """Average NMI when you have ground-truth per snapshot."""
    scores = [nmi_score(gt, pred) for gt, pred in zip(ground_truth_list, pred_list)]
    return float(np.mean(scores))


def average_f1_across_snapshots(ground_truth_list: List[np.ndarray],
                                pred_list: List[np.ndarray]) -> float:
    """Average F1 when you have ground-truth per snapshot."""
    scores = [f1_score_communities(gt, pred) for gt, pred in zip(ground_truth_list, pred_list)]
    return float(np.mean(scores))


def snapshot_modularity(adj: np.ndarray, communities: np.ndarray) -> float:
    """Modularity Q for a single snapshot (higher = better communities)."""
    G = nx.from_numpy_array(adj)
    # Convert to dict of lists (networkx format)
    partition = {}
    for node, comm in enumerate(communities):
        if comm not in partition:
            partition[comm] = []
        partition[comm].append(node)
    
    return nx.community.modularity(G, partition.values())


def average_modularity(model, snapshots: list[np.ndarray]) -> float:
    """Average modularity across all snapshots."""
    scores = []
    for t in range(model.get_num_snapshots()):
        adj = snapshots[t]
        comm = model.get_communities(t)
        scores.append(snapshot_modularity(adj, comm))
    return float(np.mean(scores))


def temporal_link_prediction_auc(
    model, snapshots: list[np.ndarray], k_future: int = 1
) -> Tuple[float, float]:
    """
    Temporal link prediction: Use communities at time t to predict edges at t+1.
    Returns (AUC, Average Precision)
    """
    aucs, aps = [], []
    
    for t in range(len(snapshots) - k_future):
        comm_t = model.get_communities(t)                    # current communities
        A_future = snapshots[t + k_future]                   # future adjacency
        
        # Simple but strong baseline: predict edge if nodes in same community
        pred_scores = []
        true_labels = []
        
        # Sample some existing and non-existing edges in future snapshot
        edges = np.argwhere(A_future > 0)
        non_edges = np.argwhere((A_future == 0) & (np.eye(A_future.shape[0]) == 0))
        
        # Balance sampling
        n_samples = min(len(edges), len(non_edges), 5000)
        np.random.seed(42)
        edges_idx = np.random.choice(len(edges), n_samples, replace=False)
        non_edges_idx = np.random.choice(len(non_edges), n_samples, replace=False)
        
        for i in edges_idx:
            u, v = edges[i]
            score = 1.0 if comm_t[u] == comm_t[v] else 0.0
            pred_scores.append(score)
            true_labels.append(1)
        
        for i in non_edges_idx:
            u, v = non_edges[i]
            score = 1.0 if comm_t[u] == comm_t[v] else 0.0
            pred_scores.append(score)
            true_labels.append(0)
        
        if len(set(true_labels)) < 2:
            continue
            
        auc = roc_auc_score(true_labels, pred_scores)
        ap = average_precision_score(true_labels, pred_scores)
        aucs.append(auc)
        aps.append(ap)
    
    return float(np.mean(aucs)), float(np.mean(aps))
