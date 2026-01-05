import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from code.utils.metrics import compute_all_metrics
from code.utils.clustering.soft_kmeans import SoftKMeans
from code.utils.clustering.utils import get_clusters_info


# ============================================================
#  Reproductibilité
# ============================================================

def set_seed(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
#  Lecture des logits / probits
# ============================================================

def read_probits(
    latent_path: str,
    n_samples: int = None,
    order: bool = True,
    subclasses=None,
    temperature: float = 2.0,
    space: str = "probits",
    top_k: int = 10,
):
    """
    Charge un fichier latent .pt contenant:
        - logits
        - labels
        - model_preds
    et renvoie (features, detector_labels),
    où detector_labels = 1 si erreur du modèle.

    On ne garde que les top_k plus grands logits/probas après tri décroissant.
    """
    pkg = torch.load(latent_path, map_location="cpu")
    all_logits = pkg["logits"].to(torch.float32)        # (N, C)
    all_labels = pkg["labels"]                          # (N,)
    all_model_preds = pkg["model_preds"]                # (N,)

    all_detector_labels = (all_model_preds != all_labels).int()

    if subclasses is not None:
        sub_idx = [i for i, label in enumerate(all_labels)
                   if label in subclasses]
        sub_idx = torch.tensor(sub_idx, dtype=torch.long)
        all_logits = all_logits.index_select(0, sub_idx)
        all_logits = all_logits[:, subclasses]
        all_detector_labels = all_detector_labels.index_select(0, sub_idx)

    if space == "logits":
        # normalisation L2
        all_logits = all_logits / (all_logits ** 2).sum(dim=1, keepdim=True).sqrt()
    elif space == "probits":
        all_logits = torch.softmax(all_logits / temperature, dim=1)
    else:
        raise ValueError(f"Unknown space: {space}")

    x = all_logits
    if order:
        x = x.sort(dim=1, descending=True)[0]

    # On garde seulement les top_k dimensions
    if top_k is not None:
        top_k_eff = min(top_k, x.size(1))
        x = x[:, :top_k_eff]

    if n_samples is None:
        return x, all_detector_labels

    return x[:n_samples], all_detector_labels[:n_samples]


DIC_DATASETS_SPLITS = {
    "cifar10":  (2000, 3000, 5000),
    "cifar100": (2000, 3000, 5000),
    "imagenet": (10000, 15000, 25000),
}


def load_raw_splits(
    data_name: str,
    model_name: str,
    seed_split: int = 9,
    order: bool = True,
    temperature: float = 2.0,
    space: str = "probits",
    top_k: int = 10,
):
    """
    Charge les trois splits bruts: res, cal, test.
    Retourne:
        (res_x, res_err, cal_x, cal_err, test_x, test_err)
    """
    n_res, n_cal, n_test = DIC_DATASETS_SPLITS[data_name]

    latent_res = (
        f"./latent/{data_name}_{model_name}/seed-split-{seed_split}/"
        f"res_n-samples-{n_res}_transform-test_n-epochs-1.pt"
    )
    res_x, res_err = read_probits(
        latent_res,
        order=order,
        temperature=temperature,
        space=space,
        top_k=top_k,
    )

    latent_cal = (
        f"./latent/{data_name}_{model_name}/seed-split-{seed_split}/"
        f"cal_n-samples-{n_cal}_transform-test_n-epochs-1.pt"
    )
    cal_x, cal_err = read_probits(
        latent_cal,
        order=order,
        temperature=temperature,
        space=space,
        top_k=top_k,
    )

    latent_test = (
        f"./latent/{data_name}_{model_name}/seed-split-{seed_split}/"
        f"test_n-samples-{n_test}.pt"
    )
    test_x, test_err = read_probits(
        latent_test,
        order=order,
        temperature=temperature,
        space=space,
        top_k=top_k,
    )

    return res_x, res_err, cal_x, cal_err, test_x, test_err


def build_mlp_dataloaders(
    res_x: torch.Tensor,
    res_err: torch.Tensor,
    cal_x: torch.Tensor,
    cal_err: torch.Tensor,
    test_x: torch.Tensor,
    test_err: torch.Tensor,
    n_val: int,
    batch_size: int,
    generator=None,
):
    """
    Pour l'entraînement du MLP, on utilise:
        - train = res ∪ cal
        - val   = première partie de test
        - test  = reste de test
    """
    data_train = torch.cat([res_x, cal_x], dim=0)
    err_train = torch.cat([res_err, cal_err], dim=0)

    data_val = test_x[:n_val]
    err_val = test_err[:n_val]

    data_test = test_x[n_val:]
    err_test = test_err[n_val:]

    train_ds = TensorDataset(data_train, err_train)
    val_ds = TensorDataset(data_val, err_val)
    test_ds = TensorDataset(data_test, err_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        generator=generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        generator=generator,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        generator=generator,
        worker_init_fn=seed_worker,
    )

    return train_loader, val_loader, test_loader, err_train, err_val, err_test


# ============================================================
#  Petit MLP (2 couches cachées) pour prédire l'erreur
# ============================================================

class SmallMLP(nn.Module):
    """
    MLP compact:
        in_dim -> 32 -> 32 -> 1
    """

    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout_p: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Représentation latente h(x) ∈ R^{hidden_dim}.
        """
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor, return_features: bool = False):
        h = self.encode(x)
        logits = self.classifier(h).squeeze(-1)
        if return_features:
            return logits, h
        return logits


# ============================================================
#  Evaluation ROC / FPR du MLP
# ============================================================

@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    all_scores = []
    all_labels = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        probs = torch.sigmoid(logits)

        all_scores.append(probs.detach().cpu())
        all_labels.append(y_batch.detach().cpu())

    all_scores = torch.cat(all_scores, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    results = compute_all_metrics(
        conf=all_scores,
        detector_labels=all_labels,
    )

    return pd.DataFrame([results])


# ============================================================
#  Encodage latent de tous les points
# ============================================================

@torch.no_grad()
def encode_all(model: SmallMLP, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    x = x.to(device)
    h = model.encode(x)
    return h.detach().cpu()  # (N, D_latent)


# ============================================================
#  Script principal
# ============================================================

if __name__ == "__main__":
    # ----------------- Config -----------------
    seed = 41
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_name = "imagenet"          # "cifar10", "cifar100", "imagenet"
    model_name = "timm-vit-tiny16"  # ou "resnet34", etc.
    space = "logits"                # "logits" ou "probits"
    seed_split = 9
    order = True
    top_k = 10                      # on garde les 10 plus gros logits
    batch_size = 512
    lr = 1e-3
    weight_decay = 0.0
    n_epochs = 100
    n_val = 12500                   # nombre de points test pour validation MLP
    dropout_p = 0.0
    n_clusters = 40                 # nombre de clusters dans l’espace latent
    bound_type = "bernstein"        # pour get_clusters_info

    save_root = "./code/utils/postprocessors/mlp_latent_cluster_upper_results"
    file_id = (
        f"{data_name}_{model_name}_seed-split-{seed_split}_order-{order}_"
        f"topk-{top_k}_hidden-32x2_lr-{lr}_epochs-{n_epochs}_wd-{weight_decay}_"
        f"space-{space}_K-{n_clusters}_bound-{bound_type}"
    )
    save_folder = os.path.join(save_root, file_id, f"seed-{seed}")
    os.makedirs(save_folder, exist_ok=True)

    # ----------------- Chargement des splits bruts -----------------
    res_x, res_err, cal_x, cal_err, test_x, test_err = load_raw_splits(
        data_name=data_name,
        model_name=model_name,
        seed_split=seed_split,
        order=order,
        temperature=2.0,
        space=space,
        top_k=top_k,
    )

    # ----------------- Dataloaders pour le MLP -----------------
    train_loader, val_loader, test_loader, err_train_all, err_val_all, err_test_all = build_mlp_dataloaders(
        res_x=res_x,
        res_err=res_err,
        cal_x=cal_x,
        cal_err=cal_err,
        test_x=test_x,
        test_err=test_err,
        n_val=n_val,
        batch_size=batch_size,
        generator=g,
    )

    in_dim = res_x.shape[1]  # = top_k

    # ----------------- Petit MLP 2 couches -----------------
    model = SmallMLP(in_dim=in_dim, hidden_dim=32, dropout_p=dropout_p).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = -float("inf")
    best_state_dict = None

    print("=== Entraînement du petit MLP (2 couches) sur les 10 plus gros logits ===")
    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).float()

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            total += y_batch.size(0)

        train_loss = running_loss / total

        val_df = evaluate(model, val_loader, device=device)
        val_auc = val_df["roc_auc"].iloc[0]
        print(val_df)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state_dict = model.state_dict()

        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"val_roc_auc={val_auc:.4f} (best={best_val_auc:.4f})"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    torch.save(model.state_dict(), os.path.join(save_folder, "best_mlp.pt"))

    # ----------------- Evaluation MLP direct -----------------
    print("\n=== Evaluation MLP direct (sigmoïde) sur test ===")
    test_df_mlp = evaluate(model, test_loader, device=device)
    print(test_df_mlp)
    test_df_mlp.to_csv(os.path.join(save_folder, "mlp_direct_test_results.csv"), index=False)

    # ============================================================
    #  Clustering dans l’espace latent du MLP
    #  + get_clusters_info pour obtenir les upper bounds par cluster
    # ============================================================

    print("\n=== Clustering dans l’espace latent du MLP et upper bound par cluster ===")

    # Encodage latent pour chaque split brut (res, cal, test)
    feats_res = encode_all(model, res_x, device=device)    # (n_res, D_latent=32)
    feats_cal = encode_all(model, cal_x, device=device)    # (n_cal, D_latent)
    feats_test = encode_all(model, test_x, device=device)  # (n_test, D_latent)

    # ----------------- SoftKMeans sur les features de res -----------------
    soft_kmeans = SoftKMeans(
        temp=5.0,
        reg_covar=1e-6,
        init_scheme_covs="statistics",
        init_kmeans_method="k-means++",
        cov_momentum=0.0,
        seed=seed,
        init_method="k-means++",
        max_iter=100,
        num_init=1,
        verbose=0,
        collect_info=False,
    )

    # Clustering de res (train "résiduel") dans l’espace latent
    # feats_res : (n_res, D_latent)
    clusters_res = soft_kmeans.fit_predict(feats_res.to(device), k=n_clusters)
    # clusters_res : (1, n_res) car num_init=1, bs=1
    clusters_res = clusters_res.squeeze(0).cpu()  # (n_res,)

    # Assignation des autres splits avec le modèle appris
    clusters_cal = soft_kmeans.predict(feats_cal.to(device)).squeeze(0).cpu()   # (n_cal,)
    clusters_test = soft_kmeans.predict(feats_test.to(device)).squeeze(0).cpu() # (n_test,)

    # get_clusters_info attend k de forme (bs,) -> ici bs=1
    k_tensor = torch.tensor([n_clusters], device=device, dtype=torch.long)

    # On utilise le split cal pour estimer les upper bounds (style calibration)
    means_cal, upper_cal = get_clusters_info(
        cal_err.to(device),
        clusters_cal.to(device),
        k_tensor,
        bound=bound_type,
    )
    # shapes: (1, K) -> on enlève la dimension batch
    means_cal = means_cal.squeeze(0)   # (K,)
    upper_cal = upper_cal.squeeze(0)   # (K,)

    print("Taux d'erreur moyens par cluster (cal):", means_cal.cpu().tolist())
    print("Upper bounds par cluster (cal):", upper_cal.cpu().tolist())

    # Score(x) = upper_bound(cluster(x))
    scores_res = upper_cal[clusters_res.to(device)].cpu().numpy()
    scores_cal = upper_cal[clusters_cal.to(device)].cpu().numpy()
    scores_test = upper_cal[clusters_test.to(device)].cpu().numpy()

    # Calcul des métriques de détection (erreur vs non-erreur)
    res_metrics_res = compute_all_metrics(
        conf=scores_res,
        detector_labels=res_err.numpy(),
    )
    res_metrics_cal = compute_all_metrics(
        conf=scores_cal,
        detector_labels=cal_err.numpy(),
    )
    res_metrics_test = compute_all_metrics(
        conf=scores_test,
        detector_labels=test_err.numpy(),
    )

    df_res = pd.DataFrame([res_metrics_res])
    df_cal = pd.DataFrame([res_metrics_cal])
    df_test = pd.DataFrame([res_metrics_test])

    print("\n=== Résultats (score = upper bound de l'erreur par cluster) ===")
    print("RES  :", df_res[["fpr", "roc_auc"]])
    print("CAL  :", df_cal[["fpr", "roc_auc"]])
    print("TEST :", df_test[["fpr", "roc_auc"]])

    df_res.to_csv(os.path.join(save_folder, "cluster_upper_res_results.csv"), index=False)
    df_cal.to_csv(os.path.join(save_folder, "cluster_upper_cal_results.csv"), index=False)
    df_test.to_csv(os.path.join(save_folder, "cluster_upper_test_results.csv"), index=False)
