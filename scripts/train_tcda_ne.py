import argparse
import numpy as np
from tcd.data.loader import get_dataset
from tcd.data.snapshots import temporal_to_snapshots
from tcd.models import TCDA_NE
from tcd.evaluation.metrics import (
    nmi_score, consecutive_nmi, consecutive_f1,
    average_modularity, temporal_link_prediction_auc
)


parser = argparse.ArgumentParser(description="Train TCDA_NE model")
parser.add_argument("--dataset", type=str, default="enron_mail", help="Dataset name")
args = parser.parse_args()

data = get_dataset(args.dataset)
snapshots = temporal_to_snapshots(data, freq="M")

model = TCDA_NE(num_communities=5, alpha=0.8, eta=5.0, random_state=42)
model.fit(snapshots)

print("Communities at month 0:", np.unique(model.get_communities(0)))
print("Communities at last month:", np.unique(model.get_communities(-1)))
print("Community assignments at last month:\n", model.get_communities(-1))

# === Evaluation ===
print("\n=== TCDA-NE Evaluation ===")
print(f"Number of snapshots: {model.get_num_snapshots()}")

# Temporal smoothness (most important for this model)
temporal_scores = consecutive_nmi(model)
print(f"Consecutive NMI (mean ± std): {temporal_scores.mean():.4f} ± {temporal_scores.std():.4f}")
print("All consecutive NMIs:", np.round(temporal_scores, 4))

# Example: compare two specific snapshots (e.g. first vs last)
nmi_first_last = nmi_score(model.get_communities(0), model.get_communities(-1))
print(f"NMI between first and last snapshot: {nmi_first_last:.4f}")

print(f"Consecutive F1  : {consecutive_f1(model).mean():.4f} ± {consecutive_f1(model).std():.4f}")
print(f"Avg Modularity: {average_modularity(model, snapshots):.4f}")
print(f"Link Pred AUC: {temporal_link_prediction_auc(model, snapshots)[0]:.4f}")
print(f"Link Pred AP: {temporal_link_prediction_auc(model, snapshots)[1]:.4f}")
