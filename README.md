# Temporal Community Detection

**Detecting evolving communities in continuous-time dynamic graphs**  

---

## Overview

This repository implements **temporal community detection** methods for dynamic graphs that evolve over continuous time (event streams).

The core idea is to learn rich temporal node embeddings that capture how communities form, split, merge, and dissolve, then cluster those embeddings to recover communities at any point in time.

**Key design principles**:
- **Dataset-agnostic**: One unified `TemporalData` format (PyG standard) works for Enron, Reddit threads, Wikipedia edits, contact traces, etc.
- **Modular**: Easy to plug in new models, datasets, and evaluation metrics.
- **Reproducible**: Everything is scriptable.

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/morteza-24/temporal-community-detection.git
cd temporal-community-detection

# 2. Create environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -e .          # editable install (recommended)
# or
pip install -r requirements.txt
```

## Data Preparation

### Prepare Enron

> **Note:** The processed Enron dataset is already included in the `data/processed/` directory, so you can skip this section and just use the uploaded files.

First, download the raw dataset from the following url and put it in `./data/raw/`:

http://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz

Then:

```bash
# Extract files
tar -xf data/raw/enron_mail_20150507.tar.gz -C data/raw/
# Rename Directory
mv data/raw/maildir data/raw/enron_mail_corpus
# One-time preprocessing
python scripts/preprocess_enron.py --force
```

This script:

- Parses all emails (sender → recipients, timestamps)
- Builds contiguous node IDs (0 ... N-1)
- Saves a clean TemporalData object

You can now load it anywhere with:

```python
from tcd.data.loader import get_dataset

data = get_dataset("enron_mail")
print(data)          # TemporalData with src, dst, t, ...
```

## Quick Start

### Compare Models

Run the full comparison script:

```bash
python scripts/train_and_compare.py
```

This trains both models and reports:
- **Average Modularity** — quality of community structure
- **Link Prediction AUC/AP** — predictive power of communities
- **Consecutive NMI/F1** — temporal smoothness

### Command-Line Training

Train individual models and see their evaluations:

```bash
# TCDA-NE (fast, NMF-based)
python scripts/train_tcda_ne.py --dataset enron_mail

# Overlapping GNN (slower, neural)
python scripts/train_overlapping_gnn.py --dataset enron_mail
```

### Basic Scripting Usage

```python
from tcd.data.loader import get_dataset
from tcd.data.snapshots import temporal_to_snapshots
from tcd.models import TCDA_NE, OverlappingGNN
from tcd.evaluation.metrics import consecutive_nmi, consecutive_f1, average_modularity, temporal_link_prediction_auc

# 1. Load data (Enron dataset included)
data = get_dataset("enron_mail")
print(f"Loaded {data.num_nodes} nodes, {len(data.t)} temporal edges")

# 2. Convert to monthly snapshots
snapshots = temporal_to_snapshots(data, freq="M")
print(f"Created {len(snapshots)} monthly snapshots")

# 3. Train TCDA-NE (fast, interpretable)
model = TCDA_NE(num_communities=15, alpha=0.8, eta=5.0, random_state=42)
model.fit(snapshots)

# 4. Get communities at any time
communities_t0 = model.get_communities(0)      # first month
communities_now = model.get_communities(-1)    # last month
print(f"Communities at t=0: {len(set(communities_t0))} unique groups")

# 5. Evaluate temporal smoothness
nmi_scores = consecutive_nmi(model)
f1_scores = consecutive_f1(model)
print(f"Consecutive NMI: {nmi_scores.mean():.4f} ± {nmi_scores.std():.4f}")
print(f"Consecutive F1:  {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
print(f"Avg Modularity: {average_modularity(model, snapshots):.4f}")
print(f"Link Pred AUC: {temporal_link_prediction_auc(model, snapshots)[0]:.4f}")
print(f"Link Pred AP: {temporal_link_prediction_auc(model, snapshots)[1]:.4f}")


# Train with Overlapping GNN
# GNN-based overlapping community detection
gnn_model = OverlappingGNN(
    num_communities=15,
    threshold=0.5,
    max_epochs=300,
    random_state=42
)
gnn_model.fit(snapshots)

# Get overlapping memberships (soft clusters)
soft_memberships = gnn_model.get_soft_memberships(0)  # N x K matrix
overlapping = gnn_model.get_overlapping_clusters(0, threshold=0.5)

# Evaluate
print(f"Consecutive NMI: {consecutive_nmi(gnn_model).mean():.4f}")
print(f"Avg Modularity: {average_modularity(gnn_model, snapshots):.4f}")
print(f"Link Pred AUC: {temporal_link_prediction_auc(gnn_model, snapshots)[0]:.4f}")
print(f"Link Pred AP: {temporal_link_prediction_auc(gnn_model, snapshots)[1]:.4f}")
```

## Evaluation

### Available Metrics

The [`metrics.py`](src/tcd/evaluation/metrics.py) module provides comprehensive evaluation:

| Metric | Description | Use Case |
|--------|-------------|----------|
| `nmi_score` | Normalized Mutual Information | Compare two clusterings |
| `f1_score_communities` | F1 with optimal matching | Community similarity |
| `consecutive_nmi` | NMI between adjacent snapshots | Temporal smoothness |
| `consecutive_f1` | F1 between adjacent snapshots | Community persistence |
| `snapshot_modularity` | Modularity Q for one snapshot | Community quality |
| `average_modularity` | Mean modularity across time | Overall quality |
| `temporal_link_prediction_auc` | Predict future edges from communities | Predictive power |

### Interpreting Results

- **High Consecutive NMI**: Communities evolve smoothly — ideal for tracking
- **High Modularity**: Well-separated community structure
- **High Link Prediction AUC**: Communities capture real network structure
- **Low Consecutive NMI**: Communities change rapidly — may need different `alpha` or `K`
