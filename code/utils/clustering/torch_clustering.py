# Minimal parity check between your Torch KMeans and scikit-learn's KMeans.
# - Samples n points in R^d with torch
# - Compares only inertia and the first few cluster centers (no alignment, no labels)
#
# Usage:
# 1) Put your class in ./kmeans.py (same folder as the notebook)
# 2) Ensure its imports resolve (utils/distances, etc.), or adjust the import below.
# 3) Make sure scikit-learn is installed: pip install scikit-learn
#
# Feel free to edit n, d, k, seed, and init_method.

import os, sys, importlib
import numpy as np
import torch
from sklearn.cluster import KMeans as SKKMeans
from .kmeans import KMeans  as TorchKMeans  # Adjust if your class is named differently
import time
from .gaussian_mixture import GaussianMixture as GMTorch
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_info
print(threadpool_info())
# -------- params you may tweak --------
# n = 25000
# d = 1000
k = 200
seed = 0
init_method = "random"   # or "rnd" to match your class
num_init = 10000
max_iter = 10
# tol = 1
first_centers_to_show = 3
# -------------------------------------
device = torch.device("cuda")
torch.manual_seed(seed)


temperature = 1
data_name = "imagenet"
model_name = "timm_vit_base16"
r=2
n_epochs = 1
seed_split = 9
transform = "test"
root = f"storage_latent/{data_name}_{model_name}_r-{r}_seed-split-{seed_split}/"
latent_path = root + f"logits_train_n-epochs{n_epochs}_transform-{transform}.pt"

def storage_bytes(t): 
    GIB = 1024 ** 3
    return t.untyped_storage().nbytes() / GIB

pkg = torch.load(latent_path, map_location="cpu")
X = pkg["logits"].to(torch.float32)[:300]
X = torch.softmax(X / temperature, dim=1)
X = X.sort(dim=1, descending=True)[0]
# Xt = X.float().unsqueeze(0)  # (1, n, d)
print("X:", X.shape, X.stride(), storage_bytes(X))
Xt=X.unsqueeze(0)


t0 = time.time()
tkm = TorchKMeans(
    init_method=init_method ,
    num_init=num_init,
    max_iter=max_iter,
    p_norm=2,
    # tol=tol,
    normalize=None,
    n_clusters=k,
    verbose=False,
    seed=seed,
)
res = tkm(Xt.to(device), k=k)
print("type of results:", res.inertia)
t1 = time.time()
print(f"Torch KMeans took {t1-t0:.3f} seconds")
print("n_iter Torch KMeans:", tkm.n_iter)
# inertia_torch = float(tkm._result.inertia[0].cpu().numpy())
# centers_torch = tkm._result.centers[0].cpu().numpy()
# results["torch"] = (inertia_torch, centers_torch)
# print(f"[Torch] inertia = {inertia_torch:.6f}")
# print(f"[Torch] first {first_centers_to_show} centers:\n", centers_torch[:first_centers_to_show])


