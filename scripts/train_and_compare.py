import time
import pandas as pd

from tcd.data.loader import get_dataset
from tcd.data.snapshots import temporal_to_snapshots
from tcd.models.tcda_ne import TCDA_NE
from tcd.models.overlapping_gnn import OverlappingGNN
from tcd.evaluation.metrics import (
    consecutive_nmi, consecutive_f1,
    average_modularity, temporal_link_prediction_auc
)

# ====================== CONFIG ======================
SNAPSHOT_FREQ = "M"
K = 15

TCDA_PARAMS = {"num_communities": K, "alpha": 0.8, "eta": 5.0, "max_iter": 200, "random_state": 42}
NOCD_PARAMS = {"num_communities": K, "threshold": 0.5, "max_epochs": 300, "random_state": 42}

def main():
    print("="*90)
    print("TEMPORAL COMMUNITY DETECTION — FAIR COMPARISON ON ENRON")
    print("="*90)

    data = get_dataset("enron_mail")
    snapshots = temporal_to_snapshots(data, freq=SNAPSHOT_FREQ)
    print(f"Loaded {len(snapshots)} monthly snapshots\n")

    results = {}

    # === TCDA-NE ===
    print("Training TCDA-NE...")
    start = time.time()
    tcda = TCDA_NE(**TCDA_PARAMS)
    tcda.fit(snapshots)
    tcda_time = time.time() - start

    results["TCDA-NE"] = {
        "Time (s)": round(tcda_time, 1),
        "Avg Modularity": f"{average_modularity(tcda, snapshots):.4f}",
        "Link Pred AUC": f"{temporal_link_prediction_auc(tcda, snapshots)[0]:.4f}",
        "Link Pred AP":  f"{temporal_link_prediction_auc(tcda, snapshots)[1]:.4f}",
        "Consec. NMI": f"{consecutive_nmi(tcda).mean():.4f}",
        "Consec. F1":  f"{consecutive_f1(tcda).mean():.4f} ± {consecutive_f1(tcda).std():.4f}",
    }

    # === OverlappingGNN ===
    print("\nTraining OverlappingGNN...")
    start = time.time()
    nocd = OverlappingGNN(**NOCD_PARAMS)
    nocd.fit(snapshots)
    nocd_time = time.time() - start

    results["OverlappingGNN"] = {
        "Time (s)": round(nocd_time, 1),
        "Avg Modularity": f"{average_modularity(nocd, snapshots):.4f}",
        "Link Pred AUC": f"{temporal_link_prediction_auc(nocd, snapshots)[0]:.4f}",
        "Link Pred AP":  f"{temporal_link_prediction_auc(nocd, snapshots)[1]:.4f}",
        "Consec. NMI": f"{consecutive_nmi(nocd).mean():.4f}",
        "Consec. F1":  f"{consecutive_f1(nocd).mean():.4f} ± {consecutive_f1(nocd).std():.4f}",
    }

    # ====================== FINAL TABLE ======================
    print("\n" + "="*90)
    print("FINAL FAIR COMPARISON (Higher = Better)")
    print("="*90)
    df = pd.DataFrame(results).T
    print(df.to_string())

if __name__ == "__main__":
    main()
