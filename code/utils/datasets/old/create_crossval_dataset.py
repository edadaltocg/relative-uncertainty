import os
import torch
import random
import numpy as np
import joblib
import pickle
import json
from torch.utils.data import Subset
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from . import get_dataset
from ..models import get_model


DATA_DIR = os.environ.get("DATA_DIR", "./data")
CHECKPOINTS_DIR_BASE = os.environ.get("CHECKPOINTS_DIR", "checkpoints/")


if __name__ == "__main__":

    # ── Top‐of‐script reproducibility setup ────────────────────────────────────────
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # Master generator (if an explicit control is needed elsewhere)
    master_gen = torch.Generator().manual_seed(seed)

    # ── Hyperparameters ───────────────────────────────────────────────────────────

    model_name = "resnet34"
    model_seed = 1
    dataset_name = "cifar10"
    batch_size = 256
    n_splits = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(model_name=model_name, 
                      dataset_name=dataset_name,
                      model_seed=model_seed,
                      checkpoint_dir = os.path.join(CHECKPOINTS_DIR_BASE, "ce")
                      )
    model = model.to(device)

    dataset = get_dataset(
            dataset_name=dataset_name, model_name=model_name, root=DATA_DIR,
            shuffle=True, random_state=seed)
    print("Dataset size", len(dataset))

    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=6, prefetch_factor=2
        )
    
    all_detector_labels = []
    for inputs, labels in tqdm(dataloader, desc=f"Building Stratify"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():      
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            detector_labels = preds != labels

        all_detector_labels.append(detector_labels.cpu().numpy())

    all_detector_labels = np.concatenate(all_detector_labels, axis=0)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # 3. generate and save all the (train_idx, val_idx) pairs
    splits = list(skf.split(np.arange(len(dataset)), all_detector_labels))
    
    folder_path = os.path.join(DATA_DIR, "cifar10-crossval", "crossval-1")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(os.path.join(folder_path, "crossval-1.pkl"), "wb") as f:
        pickle.dump(splits, f)
    # joblib.dump(splits, os.path.join(DATA_DIR, "crossval-1.pkl"))

    config = {
        "model_name": model_name,
        "model_seed": model_seed,
        "dataset_name": dataset_name,
        "seed": seed,
        "n_folds": n_splits,
        "batch_size": batch_size,
    }
    with open(os.path.join(folder_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


    

