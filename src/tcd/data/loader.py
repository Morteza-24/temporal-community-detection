from .enron_mail import load_enron
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
import torch_geometric

def get_dataset(name: str) -> torch_geometric.data.TemporalData:
    name = name.lower()
    if name == "enron_mail":
        return load_enron()
    # official TGB loader
    dataset = PyGLinkPropPredDataset(name=name, root="datasets")
    return dataset.get_TemporalData()
