import argparse
from tcd.data.loader import get_dataset
from tcd.data.snapshots import temporal_to_snapshots
from tcd.models import OverlappingGNN
from tcd.evaluation.metrics import (
    consecutive_nmi, consecutive_f1,
    average_modularity, temporal_link_prediction_auc
)


parser = argparse.ArgumentParser(description="Train TCDA_NE model")
parser.add_argument("--dataset", type=str, default="enron_mail", help="Dataset name")
args = parser.parse_args()

data = get_dataset(args.dataset)
snapshots = temporal_to_snapshots(data, freq="M")

model = OverlappingGNN(
    num_communities=5,
    threshold=0.5,
    max_epochs=500,
    random_state=42
)
model.fit(snapshots)

# Evaluation
temporal_scores = consecutive_nmi(model)
print(f"\nConsecutive NMI (hard): {temporal_scores.mean():.4f} ± {temporal_scores.std():.4f}")
print(f"Consecutive F1  : {consecutive_f1(model).mean():.4f} ± {consecutive_f1(model).std():.4f}")
print(f"Avg Modularity: {average_modularity(model, snapshots):.4f}")
print(f"Link Pred AUC: {temporal_link_prediction_auc(model, snapshots)[0]:.4f}")
print(f"Link Pred AP: {temporal_link_prediction_auc(model, snapshots)[1]:.4f}")
