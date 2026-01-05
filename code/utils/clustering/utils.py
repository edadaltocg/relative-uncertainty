#
from random import sample
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor

__all__ = ["ClusterResult", "group_by_label_mean", "first_nonzero", "rm_kwargs"]


class ClusterResult(NamedTuple):
    """Named and typed result tuple for kmeans algorithms

    Args:
        labels: label for each sample in x
        centers: corresponding coordinates of cluster centers
        inertia: sum of squared distances of samples to their closest cluster center
        x_org: original x
        x_norm: normalized x which was used for cluster centers and labels
        k: number of clusters
        soft_assignment: assignment probabilities of soft kmeans
    """

    labels: LongTensor
    centers: Tensor
    inertia: Tensor
    k: LongTensor

class SoftClusterResult(NamedTuple):
    """Named and typed result tuple for kmeans algorithms

    Args:
        resp: label for each sample in x
        centers: corresponding coordinates of cluster centers
        inertia: sum of squared distances of samples to their closest cluster center
        x_org: original x
        x_norm: normalized x which was used for cluster centers and labels
        k: number of clusters
        soft_assignment: assignment probabilities of soft kmeans
    """
    log_resp: Tensor
    weights: Tensor
    means: Tensor
    cov_diags: Tensor
    lower_bound: Tensor
    k: LongTensor

class SoftClusterOrigResult(NamedTuple):
    """Named and typed result tuple for kmeans algorithms

    Args:
        resp: label for each sample in x
        centers: corresponding coordinates of cluster centers
        inertia: sum of squared distances of samples to their closest cluster center
        x_org: original x
        x_norm: normalized x which was used for cluster centers and labels
        k: number of clusters
        soft_assignment: assignment probabilities of soft kmeans
    """

    log_resp: Tensor
    weights: Tensor
    means: Tensor
    cov_diags: Tensor
    lower_bound: Tensor
    k: LongTensor
    params_history: Dict[str, List[Tensor]]
    lower_bound_history: List
    lower_bound_test_history: List
    list_results_test : List[Dict]
    list_results_cal : List[Dict]
    list_results_res : List[Dict]
    lower_bound_cal_history: List[float]
    mutinfo_res_list : List[float]
    mutinfo_res_decomp_history : List[float]
    H_Z_history : List[float]
    H_Z_cond_history : List[float]
    H_S_history : List[float]
    H_S_cond_history : List[float]
    list_clusters_res : List[Tensor]
    # list_resp_dead : List[List[int]]

# class MutInfoResult(NamedTuple):
#     """Named and typed result tuple for kmeans algorithms

#     Args:
#         resp: label for each sample in x
#         centers: corresponding coordinates of cluster centers
#         inertia: sum of squared distances of samples to their closest cluster center
#         x_org: original x
#         x_norm: normalized x which was used for cluster centers and labels
#         k: number of clusters
#         soft_assignment: assignment probabilities of soft kmeans
#     """

#     means: Tensor
#     cov_diags: Tensor
#     best_mutinfo: float
#     mutinfo_history : List[float]
#     mutinfo_hard_history : List[float]
#     covs_history : List[Tensor]
#     best_iter: int
#     H_Z_history : List[float]
#     H_Z_giv_E_history : List[float]
#     list_results_cal : List[Dict]
#     list_results_test : List[Dict]
#     list_results_res : List[Dict]


class MutInfoResult(NamedTuple):
    """Named and typed result tuple for kmeans algorithms

    Args:
        resp: label for each sample in x
        centers: corresponding coordinates of cluster centers
        inertia: sum of squared distances of samples to their closest cluster center
        x_org: original x
        x_norm: normalized x which was used for cluster centers and labels
        k: number of clusters
        soft_assignment: assignment probabilities of soft kmeans
    """

    means: Tensor
    precisions: Tensor
    best_mutinfo: float
    best_iter: int
    precisions_history : List[Tensor]
    mutinfo_res_history : List[float]
    H_Z_history : List[float]
    H_Z_cond_history : List[float]
    H_E_history : List[float]
    H_E_cond_history : List[float]
    H_S_history : List[float]
    H_S_cond_history : List[float]
    classif_results : Dict

class SoftKmeansGDResult(NamedTuple):
    """Named and typed result tuple for kmeans algorithms

    Args:
        resp: label for each sample in x
        centers: corresponding coordinates of cluster centers
        inertia: sum of squared distances of samples to their closest cluster center
        x_org: original x
        x_norm: normalized x which was used for cluster centers and labels
        k: number of clusters
        soft_assignment: assignment probabilities of soft kmeans
    """

    means: Tensor
    precisions: Tensor
    weights: Tensor
    best_likelihood: float
    best_iter: int
    params_history : List[Tensor]
    likelihood_history : List[float]
    H_Z_history : List[float]
    H_Z_cond_history : List[float]
    H_E_history : List[float]
    H_E_cond_history : List[float]
    H_S_history : List[float]
    H_S_cond_history : List[float]
    classif_results : Dict
    list_clusters_res : List[Tensor]

def plot_metric_corr(classif_results, upper_types, splits=("res_val", "test"),
                     metric="fpr", save_folder="./", suffix=""):

    x = np.asarray(classif_results[splits[0]][upper_types[0]][metric].values, dtype=float)
    y = np.asarray(classif_results[splits[1]][upper_types[1]][metric].values, dtype=float)

    # Mask NaN/inf in a *pairwise* way
  
    mask = np.isfinite(x) & np.isfinite(y)

    # Handle degenerate case: not enough non-NaN points
    if mask.sum() < 2:
        corr = np.nan
        x_plot, y_plot = x[mask], y[mask]
    else:
        x_plot, y_plot = x[mask], y[mask]
        corr = np.corrcoef(x_plot, y_plot)[0, 1]

    plt.scatter(x_plot, y_plot, c="blue", alpha=0.7)
    plt.xlabel(f"{splits[0]}  {metric}")
    plt.ylabel(f"{splits[1]} {metric}")
    plt.title(f"Correlation (ignoring NaNs): {corr:.4f}" if np.isfinite(corr)
              else "Correlation: undefined (fewer than 2 valid points)")
    
    os.makedirs(save_folder, exist_ok=True)
    fname = f"split-{splits[0]}_upper-{upper_types[0]}_vs_split-{splits[1]}_upper-{upper_types[1]}_{metric}{suffix}.png"
    plt.savefig(os.path.join(save_folder, fname))
    plt.close()


@torch.jit.script
def group_by_label_mean(
    x: Tensor,
    labels: Tensor,
    k_max_range: Tensor,
) -> Tensor:
    """Group samples in x by label
    and calculate grouped mean.

    Args:
        x: samples (BS, N, D)
        labels: label per sample (BS, M, N)
        k_max_range: range of max number if clusters (BS, K_max)

    Returns:

    """
    # main idea: https://stackoverflow.com/a/56155805
    assert isinstance(x, Tensor)
    assert isinstance(labels, Tensor)
    assert isinstance(k_max_range, Tensor)
    n, d = x.size()
    bs, m, n_ = labels.size()
    assert  n == n_
    k_max = k_max_range.size(-1)
    M = (
        (
            labels[:, :, :, None].expand(bs, m, n, k_max)
            == k_max_range[:, None, None, :].expand(bs, m, n, k_max)
        )
        .permute(0, 1, 3, 2)
        .to(x.dtype)
    )
    M = F.normalize(M, p=1.0, dim=-1)
    return torch.matmul(M, x[:, None, :, :].expand(bs, m, n, d))


@torch.jit.script
def first_nonzero(x: Tensor, dim: int = -1) -> Tuple[Tensor, Tensor]:
    """Return idx of first positive (!) nonzero element
    of each row in 'dim' of tensor 'x'
    and a mask if such an element does exist.

    Returns:
        msk, idx
    """
    # from: https://discuss.pytorch.org/t/first-nonzero-index/24769/9
    assert isinstance(x, Tensor)
    if len(x.shape) > 1:
        assert dim == -1 or dim == len(x.shape) - 1
    nonz = x > 0
    return ((nonz.cumsum(dim) == 1) & nonz).max(dim)


def rm_kwargs(kwargs: Dict, keys: List):
    """Remove items corresponding to keys
    specified in 'keys' from kwargs dict."""
    keys_ = list(kwargs.keys())
    for k in keys:
        if k in keys_:
            del kwargs[k]
    return kwargs



## Other code
def plot_lower_bound(lower_bound_history, lower_bound_cal_history, lower_bound_test_history, save_folder, suffix=""):

    plt.plot(range(len(lower_bound_history)), lower_bound_history, label="Lower Bound")
    plt.plot(range(len(lower_bound_cal_history)), lower_bound_cal_history, label="Lower Bound (Calibration)")
    plt.plot(range(len(lower_bound_test_history)), lower_bound_test_history, label="Lower Bound (Test)")

    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Lower Bound")
    plt.title("Lower Bound Optimization")
    plt.tight_layout()
    save_path = os.path.join(save_folder, f"lower_bound_optimization{suffix}.png")
    plt.savefig(save_path, dpi=160)
    plt.close()

import torch

def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).to(mat_a.device)
    
    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)
    
    return res


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)

import os
import matplotlib.pyplot as plt

def plot_cov_trajectory(sigmas_list=None, precisions_list=None, save_folder=None, ylims=None, start = 0, suffix=""):

    if sigmas_list is not None:
        trajectories = torch.stack(sigmas_list)  # (iter, K, C)
    elif precisions_list is not None:
        trajectories = torch.stack(precisions_list).reciprocal()  # (iter, K, C)
    else:
        raise ValueError("Either sigmas_list or precision_list must be provided.")
    num_iter, K, C = trajectories.shape
    fig, axs = plt.subplots(10, figsize=(10, 15), sharex=True)

    # I want a plot for each dimension.
    for d in range(C):
        ax = axs[d]
        for k in range(K):
            ax.plot(range(start, num_iter), trajectories[start:, k, d].cpu().numpy())
        # ax.set_xlabel("Iteration")
        # ax.set_ylabel("Covariance Value")
        ax.set_title(f"Dim-{d}")
        # if ylims is not None:
        #     ax.set_ylim(ylims)
        # plt.tight_layout()

    save_path = os.path.join(save_folder, f"cov_trajectory_cluster{suffix}.png")
    plt.savefig(save_path, dpi=160)
    plt.close()

# def plot_cluster_sizes(clusters: LongTensor, save_folder: str, suffix="", means=None):
#     num_bins = int(clusters.max().item()) + 1
#     counts = torch.bincount(clusters.to('cpu'), minlength=num_bins)  # shape (K,)

#     # Sort counts (desc) but keep the original cluster labels for the x-tick text
#     sorted_counts, sort_idx = torch.sort(counts, descending=True)    # both on CPU

#     plt.figure(figsize=(7,4))
#     plt.bar(range(len(sorted_counts)), sorted_counts.numpy(), width=0.9)
#     plt.title("Cluster Sizes (sorted)")
#     plt.xlabel("Cluster (sorted by size)")
#     plt.ylabel("Number of Samples")

#     # Show original cluster labels along the x-axis (can be cluttered if K is large)

#     plt.tight_layout()
#     plt.savefig(os.path.join(save_folder, f"cluster_sizes_sorted{suffix}.png"), dpi=150)
#     plt.close()


import matplotlib as mpl
from typing import Optional

@torch.no_grad()
def reorder_by_error_threshold(
    detector_labels: torch.Tensor,   # (n,) in {0,1} or [0,1]
    clusters: torch.Tensor,          # (n,) long in [0, k-1]
    n_cluster,                               # int or 0-dim tensor
    tau: float = 0.01,               # gap tolerance on means
):
    """
    Single-pass 1-D threshold merging:
      1) compute per-cluster mean error \hat p_c
      2) sort clusters by \hat p_c
      3) sweep; start a new group when the mean gap > tau

    Returns:
        mapping: dict {old_cluster_id: new_group_id}
                 empty clusters get -1
    """
    device = clusters.device
    y = detector_labels.to(device=device, dtype=torch.float32)
    C = clusters.to(device=device, dtype=torch.long)

    # counts and sum of labels
    counts = torch.bincount(C, minlength=n_cluster).to(torch.float32)
    sum_y  = torch.zeros(n_cluster, device=device, dtype=torch.float32)
    sum_y.scatter_add_(0, C, y)

    # means; mark empties with +inf so they sort to the end
    means = torch.empty(n_cluster, device=device, dtype=torch.float32)
    nonempty = counts > 0
    means[nonempty] = sum_y[nonempty] / counts[nonempty]
    means[~nonempty] = float('inf')

    order = torch.argsort(means)  # ascending by mean

    # sweep to assign group ids
    group_id = torch.full((n_cluster,), -1, device=device, dtype=torch.long)  # -1 for empty clusters by default
    
    print("Means sorted:", means[order].tolist())
    current_group = -1
    prev_mean = None
    for cid in order.tolist():
        if not nonempty[cid]:
            continue  # keep -1
        m = float(means[cid])
        if (prev_mean is None) or (m - prev_mean > tau):
            current_group += 1
        group_id[cid] = current_group
        prev_mean = m

    return group_id

def get_clusters_info_2(detector_labels: torch.Tensor, clusters: torch.Tensor, k: torch.Tensor, alpha=0.05, bound="hoeffding"):
        """
        detector_labels: (n,) in {0,1} (or [0,1] probs)
        clusters:        (bs, n) long in [0, k_b-1] per batch
        k:               (bs,) number of clusters per batch

        Saves:
        self.cluster_counts       -> (bs, k_max)
        self.cluster_error_means  -> (bs, k_max)
        self.cluster_error_vars   -> (bs, k_max)
        self.cluster_intervals    -> (bs, k_max, 2)  # [lower, upper]
        """
        device = clusters.device
        dtype  = torch.float32

        if len(clusters.shape) == 1:
            clusters = clusters.unsqueeze(0)  # (1, n)
            k = k.unsqueeze(0)              # (1,)

        y = detector_labels.to(device=device, dtype=dtype)       # (n,)
        C = clusters.to(device=device, dtype=torch.long)         # (bs, n)
        k = k.to(device=device, dtype=torch.long)               # (bs,)
        bs, n = C.shape
        assert y.numel() == n, "detector_labels must have same n as clusters"

        k_max = int(k.max().item())

        # ---- counts per (batch, cluster) via scatter_add (no one-hot needed) ----
        counts = torch.zeros(bs, k_max, device=device, dtype=dtype)
        counts.scatter_add_(dim=1, index=C, src=torch.ones(bs, n, device=device, dtype=dtype))  # (bs, k_max)

        # ---- sums of labels and squared labels per cluster ----
        y_row  = y.expand(bs, n)                                   # (bs, n) view
        sum_y  = torch.zeros(bs, k_max, device=device, dtype=dtype)
        sum_y.scatter_add_(dim=1, index=C, src=y_row)              # (bs, k_max)

        y2_row = (y * y).expand(bs, n)
        sum_y2 = torch.zeros(bs, k_max, device=device, dtype=dtype)
        sum_y2.scatter_add_(dim=1, index=C, src=y2_row)            # (bs, k_max)

        # ---- means and population variance (ddof=0 to match unbiased=False) ----
        denom = counts.clamp_min(1.0)
        means = sum_y / denom
        vars_ = sum_y2 / denom - means.pow(2)
        vars_.clamp_(min=0.0)

    

        if bound.lower() == "hoeffding":
            # half = sqrt( log(2/alpha) / (2 n) )
            log_term = torch.log(torch.tensor(2.0 / alpha, device=device, dtype=dtype))
            half = torch.sqrt(log_term / (2.0 * denom))
        elif bound.lower() == "bernstein":
            # Empirical Bernstein (Audibert et al., 2009) for X in [0,1]:
            # |μ - μ̂| ≤ sqrt( 2 V_n log(3/α) / n ) + 3 log(3/α) / n
            # using empirical (population) variance vars_ above
            log_term = torch.log(torch.tensor(3.0 / alpha, device=device, dtype=dtype))
            half = torch.sqrt((2.0 * vars_ * log_term) / denom) + (3.0 * log_term) / denom
        else:
            raise ValueError("bound must be 'hoeffding' or 'bernstein'")
        lower = (means - half).clamp_(0.0, 1.0)
        upper = (means + half).clamp_(0.0, 1.0)

        empty   = (counts == 0.0)

        # mask = invalid | empty
        mask = empty
        if mask.any():
            counts[mask] = 0.0
            means[mask]  = 0.0
            vars_[mask]  = 0.0
            lower[mask]  = 0.0
            upper[mask]  = 1.0
        return means, upper, counts


def get_clusters_info(detector_labels: torch.Tensor, clusters: torch.Tensor, k: torch.Tensor, alpha=0.05, bound="hoeffding", return_lb=False):
        """
        detector_labels: (n,) in {0,1} (or [0,1] probs)
        clusters:        (bs, n) long in [0, k_b-1] per batch
        k:               (bs,) number of clusters per batch

        Saves:
        self.cluster_counts       -> (bs, k_max)
        self.cluster_error_means  -> (bs, k_max)
        self.cluster_error_vars   -> (bs, k_max)
        self.cluster_intervals    -> (bs, k_max, 2)  # [lower, upper]
        """
        device = clusters.device
        dtype  = torch.float32

        if len(clusters.shape) == 1:
            clusters = clusters.unsqueeze(0)  # (1, n)
            k = k.unsqueeze(0)              # (1,)

        y = detector_labels.to(device=device, dtype=dtype)       # (n,)
        C = clusters.to(device=device, dtype=torch.long)         # (bs, n)
        k = k.to(device=device, dtype=torch.long)               # (bs,)
        bs, n = C.shape
        assert y.numel() == n, "detector_labels must have same n as clusters"

        k_max = int(k.max().item())

        # ---- counts per (batch, cluster) via scatter_add (no one-hot needed) ----
        counts = torch.zeros(bs, k_max, device=device, dtype=dtype)
        counts.scatter_add_(dim=1, index=C, src=torch.ones(bs, n, device=device, dtype=dtype))  # (bs, k_max)

        # ---- sums of labels and squared labels per cluster ----
        y_row  = y.expand(bs, n)                                   # (bs, n) view
        sum_y  = torch.zeros(bs, k_max, device=device, dtype=dtype)
        sum_y.scatter_add_(dim=1, index=C, src=y_row)              # (bs, k_max)

        y2_row = (y * y).expand(bs, n)
        sum_y2 = torch.zeros(bs, k_max, device=device, dtype=dtype)
        sum_y2.scatter_add_(dim=1, index=C, src=y2_row)            # (bs, k_max)

        # ---- means and population variance (ddof=0 to match unbiased=False) ----
        denom = counts.clamp_min(1.0)
        means = sum_y / denom
        vars_ = sum_y2 / denom - means.pow(2)
        vars_.clamp_(min=0.0)

    

        if bound.lower() == "hoeffding":
            # half = sqrt( log(2/alpha) / (2 n) )
            log_term = torch.log(torch.tensor(2.0 / alpha, device=device, dtype=dtype))
            half = torch.sqrt(log_term / (2.0 * denom))
        elif bound.lower() == "bernstein":
            # Empirical Bernstein (Audibert et al., 2009) for X in [0,1]:
            # |μ - μ̂| ≤ sqrt( 2 V_n log(3/α) / n ) + 3 log(3/α) / n
            # using empirical (population) variance vars_ above
            log_term = torch.log(torch.tensor(3.0 / alpha, device=device, dtype=dtype))
            half = torch.sqrt((2.0 * vars_ * log_term) / denom) + (3.0 * log_term) / denom
        else:
            raise ValueError("bound must be 'hoeffding' or 'bernstein'")
        lower = (means - half).clamp_(0.0, 1.0)
        upper = (means + half).clamp_(0.0, 1.0)

        empty   = (counts == 0.0)

        # mask = invalid | empty
        mask = empty
        if mask.any():
            counts[mask] = 0.0
            means[mask]  = 0.0
            vars_[mask]  = 0.0
            lower[mask]  = 0.0
            upper[mask]  = 1.0
        if return_lb:
            return means, lower, upper
        return means, upper

def plot_cluster_sizes(
    clusters: LongTensor,
    save_folder: str,
    suffix: str = "",
    error_means: Optional[Tensor] = None,  # shape (K,), mean error per cluster,
    split_name: str = "res",
    sort=True
):
    # num_bins = int(clusters.max().item()) + 1
    num_bins = error_means.shape[0]
    counts = torch.bincount(clusters.to('cpu'), minlength=num_bins)  # (K,)

    # Sort by size (desc) and keep permutation
    if sort:
        sorted_counts, sort_idx = torch.sort(counts, descending=True)    # both on CPU
    else:
        sorted_counts = counts
        sort_idx = torch.arange(num_bins)

    fig, ax = plt.subplots(figsize=(7, 4))
    if error_means is None:
        # Original behavior: single color
        ax.bar(range(len(sorted_counts)), sorted_counts.numpy(), width=0.9)
    else:
        # Reorder error means to match the sorted clusters
        err = error_means.to('cpu').float()
        # if err.numel() != num_bins:
            
            # raise ValueError(f"error_means must have length K={num_bins}, got {err.numel()}")
        err_sorted = err[sort_idx]

        # Map error means to colors
        vmin = float(torch.min(err_sorted))
        vmax = float(torch.max(err_sorted))
        # Avoid degenerate norm if all equal
        if vmin == vmax:
            vmin, vmax = vmin - 1e-6, vmax + 1e-6

        cmap = mpl.cm.viridis  # choose any mpl colormap you like
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        colors = cmap(norm(err_sorted.numpy()))

        ax.bar(range(len(sorted_counts)), sorted_counts.numpy(), width=0.9, color=colors)

        # Add a colorbar keyed to error mean
        cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, pad=0.01)
        cbar.set_label("Mean error per cluster")

    ax.set_title(r"Cluster Sizes on $\mathcal{D}_\mathrm{res}$")
    ax.set_xlabel("Cluster (sorted by size)")
    ax.set_ylabel("Number of Samples")
    fig.tight_layout()
    fig.savefig(os.path.join(save_folder, f"cluster_sizes_sorted-split-{split_name}{suffix}.png"), dpi=150)
    plt.close(fig)

def plot_eval(
        results_test=None, 
        results_cal=None, 
        results_res=None, 
        results_upper_res=None,
         metric ="fpr", save_folder=None, suffix=""):
    plt.plot(range(len(results_test)), results_test[metric], label=f"{metric.upper()}(Test)")
    plt.plot(range(len(results_cal)), results_cal[metric], label=f"{metric.upper()} (Cal)")
    plt.plot(range(len(results_res)), results_res[metric], label=f"{metric.upper()}(Res)")
    if results_upper_res is not None:
        for split, values in results_upper_res.items():
            plt.plot(range(len(values)), values[metric], label=f"{metric.upper()}(Upper res - {split})")

    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()}")
    plt.tight_layout()
    save_path = os.path.join(save_folder, f"{metric}{suffix}.png")
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_entropy_and_cond_entropy_history(
        histories, 
        cond_entropy_histories, 
        save_folder, 
        supervised=None, 
        split="res",
        variable = "E",
        cond="Z",
        suffix="",
        n_clusters=None
        ):

    # cond_entrop_label = "H(Z|E)" if supervised else "H(Z|S)"
    # supervised_str = "supervised" if supervised else "unsupervised"

    colors = {
        "model_hard": "blue",
        "model_soft": "cyan",
        "emp_hard": "red",
        "emp_soft": "orange",
    }
    styles = {
        "uncond": "-",
        "cond": "--",
    }
    hard_soft_types = ["hard", "soft"]
    if variable == "Z":
        upper_bound = entropy(torch.ones(n_clusters) / n_clusters).item()
    plt.figure(figsize=(6,4))
    # for model_type in ["model", "emp"]:
    for model_type in ["emp"]:
    # model_type = "emp"
    # model_type = "emp"
        if model_type == "model":
            hard_soft_types = ["soft"]
        else:
            # hard_soft_types = ["hard", "soft"]
              hard_soft_types = ["hard", "soft"]
        for hard_soft in hard_soft_types:

            if variable == "Z":
                history = histories[cond][model_type][hard_soft]
                cond_history = cond_entropy_histories[cond][model_type][hard_soft]
            else:
                history = histories[model_type][hard_soft]
                cond_history = cond_entropy_histories[model_type][hard_soft]

            print("len history:", len(history))
            entropy_label = f"H({variable})"
            cond_entropy_label = f"H({variable}|{cond})"

            # label = f"{entropy_label} ({model_type}, {hard_soft})"
            # cond_label = f"{cond_entropy_label} ({model_type}, {hard_soft})"
            label = f"{entropy_label} ({hard_soft})"
            cond_label = f"{cond_entropy_label} ({hard_soft})"
            plt.plot(range(len(history)), history, label=label, linestyle=styles["uncond"], color=colors[f"{model_type}_{hard_soft}"])
            plt.plot(range(len(cond_history)), cond_history, label=cond_label, linestyle=styles["cond"], color=colors[f"{model_type}_{hard_soft}"])
            if variable == "Z":
                plt.hlines(upper_bound, 0, len(history)-1, colors='gray', linestyles='dotted', label="H_max(Z)")

    # entropy_history = entropy_histories["supervised"] if supervised else entropy_histories["unsupervised"]
    # cond_entropy_history = cond_entropy_histories["supervised"] if supervised else cond_entropy_histories["unsupervised"]
    # n_iter = len(entropy_history)
    # cond_entrop_label = "H(Z|E)" if supervised else "H(Z|S)"
    # plt.plot(range(n_iter), entropy_history, label="H(Z)")
    # plt.plot(range(n_iter), cond_entropy_history, label=cond_entrop_label)
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Entropy")
    # plt.title(f"Entropy H(Z) and {cond_entrop_label} on $\mathcal{{D}}_\mathrm{{{split}}}$")
    plt.tight_layout()
   # save_file = f"entropies_supervised_history{suffix}.png" if supervised else f"entropies_unsupervised_history{suffix}.png"
    save_file = f"entropies_{variable}_cond-{cond}_history{suffix}.png"
    save_path = os.path.join(save_folder, save_file)
    plt.savefig(save_path, dpi=160)
    plt.close()

def entropy(p):
    mask = p > 0
    p = p[mask]
    H = - torch.sum(p * torch.log(p))
    return H



def plot_cov_trajectory_vs_alpha(sigmas_list=None, precisions_list=None, save_folder=None, ylims=None, suffix=""):

    if sigmas_list is not None:
        trajectories = torch.stack(sigmas_list)  # (iter, K, C)
    elif precisions_list is not None:
        trajectories = torch.stack(precisions_list).reciprocal()  # (iter, K, C)
    else:
        raise ValueError("Either sigmas_list or precision_list must be provided.")
    num_iter, K, C = trajectories.shape
    fig, axs = plt.subplots(10, figsize=(10, 15), sharex=True)

    # I want a plot for each dimension.
    for d in range(C):
        ax = axs[d]
        for k in range(K):
            ax.plot(range(num_iter), trajectories[:, k, d].cpu().numpy())
        # ax.set_xlabel("Iteration")
        # ax.set_ylabel("Covariance Value")
        ax.set_title(f"Dim-{d}")
        # if ylims is not None:
        #     ax.set_ylim(ylims)
        # plt.tight_layout()

    save_path = os.path.join(save_folder, f"cov_trajectory_cluster{suffix}.png")
    plt.savefig(save_path, dpi=160)
    plt.close()

# def read_probits(latent_path, n_samples=None, order=True, subclasses=None, temperature=2.0):
#     pkg = torch.load(latent_path, map_location="cpu")
#     all_logits = pkg["logits"].to(torch.float32)        # (N, C)
#     all_labels = pkg["labels"]              # (N,)
#     all_model_preds  = pkg["model_preds"]# (N,)
#     all_detector_labels = (all_model_preds != all_labels).int()
    
#     if subclasses is not None:
#         sub_idx = [i for i, label in enumerate(all_labels)
#              if label in subclasses]
#         sub_idx = torch.tensor(sub_idx, dtype=torch.long)
#         all_logits = all_logits.index_select(0, sub_idx)
#         all_logits = all_logits[:, subclasses]
#         all_detector_labels = all_detector_labels.index_select(0, sub_idx)

#     probits = torch.softmax(all_logits / temperature, dim=1)
#     if order:
#         probits = probits.sort(dim=1, descending=True)[0]

#     if n_samples is None:
#         return probits, all_detector_labels
   
#     return probits[:n_samples], all_detector_labels[:n_samples]

def read_probits(latent_path, n_samples=None, order=True, subclasses=None, temperature=2.0, space="probits", n_dim=None):
    pkg = torch.load(latent_path, map_location="cpu")
    all_logits = pkg["logits"].to(torch.float32)        # (N, C)
    all_labels = pkg["labels"]              # (N,)
    all_model_preds  = pkg["model_preds"]# (N,)
    all_detector_labels = (all_model_preds != all_labels).int()
    
    if subclasses is not None:
        sub_idx = [i for i, label in enumerate(all_labels)
             if label in subclasses]
        sub_idx = torch.tensor(sub_idx, dtype=torch.long)
        all_logits = all_logits.index_select(0, sub_idx)
        all_logits = all_logits[:, subclasses]
        all_detector_labels = all_detector_labels.index_select(0, sub_idx)
    
    if space == "logits":
        all_logits = all_logits / (all_logits ** 2).sum(dim=1, keepdim=True).sqrt()
    elif space == "probits":
        all_logits = torch.softmax(all_logits / temperature, dim=1)
    else:
        raise ValueError(f"Unknown space: {space}")

    probits = all_logits
    if order:
        probits = probits.sort(dim=1, descending=True)[0]

    if n_dim is not None:
        probits = probits[:, :n_dim]

    if n_samples is None:
        return probits, all_detector_labels
   
    return probits[:n_samples], all_detector_labels[:n_samples]

def plot_means_covs_trajectory_per_cluster(trajectory, save_folder=None, start=0, suffix="", scale="normal"):
    """
    trajectory: List of covariances (T, K, D)
    """
    # trajectory = torch.stack(trajectory)
    #.reciprocal()  # (T, K, D)
    if trajectory.ndim == 3:
        trajectory = trajectory.mean(dim=2)    # (T, D)
    else:
        trajectory = trajectory   # (T, D)

    plt.figure(figsize=(6,4))
    for k in range(trajectory.shape[1]):
        if scale == "log":
            plt.semilogy(range(start, len(trajectory)), trajectory[start:, k].cpu().numpy(), label=f"Cluster-{k}")
        else:
            plt.plot(range(start, len(trajectory)), trajectory[start:, k].cpu().numpy(), label=f"Cluster-{k}")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Covariance")
    plt.title("Mean Covariance vs Iteration")
    # plt.ylim(bottom=0, top=0.005)
    plt.legend()
    log_str = "_logscale" if scale == "log" else ""
    save_fig = os.path.join(save_folder, f"mean_covs_per_cluster_trajectory{log_str}{suffix}.png")
    plt.savefig(save_fig, dpi=160)
    plt.close()

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import numpy as np

def plot_correlation_covs_n_clusters(
    covs_trajectory, clusters_trajectory, errors, n_clusters, save_folder, suffix="",
    ylim_top=1,
):
    """
    covs_trajectory: list/sequence; last item has shape (K, D) or (K, D, ...)
    clusters_trajectory: list/sequence of cluster assignments; we use the last one
    errors: (N,) tensor with {0,1} or floats in [0,1]
    n_clusters: int
    """
    clusters = clusters_trajectory[-1]  # (N,)
    # error_means: (K,) — mean error per cluster (your helper returns [0] as the vector)
    error_means = get_clusters_info(
        errors, clusters, torch.tensor([n_clusters], device=errors.device), bound=bound
    )[0].squeeze(0)  # (K,)

    # mean covariance per cluster (average over dimensions)
    last_covs = covs_trajectory[-1]              # (K, D[, ...])
    mean_covs = last_covs.mean(dim=1)            # (K,)

    # counts per cluster
    K = mean_covs.shape[0]
    num_samples_per_cluster = torch.bincount(clusters, minlength=K)  # (K,)

    # ---- colors from error_means
    err_cpu = error_means.detach().to(torch.float32).cpu()
    vmin = float(err_cpu.min())
    vmax = float(err_cpu.max())
    if vmin == vmax:
        eps = 1e-6
        vmin, vmax = vmin - eps, vmax + eps

    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    x = num_samples_per_cluster.cpu().numpy()
    y = mean_covs.detach().cpu().numpy()
    cvals = err_cpu.numpy()  # pass values, let mpl map via cmap+norm

    plt.figure(figsize=(6, 4))
    sc = plt.scatter(x, y, c=cvals, cmap=cmap, norm=norm, alpha=0.85, edgecolors="none")
    cbar = plt.colorbar(sc)
    cbar.set_label("Cluster error mean")

    plt.xlabel("Number of samples per cluster")
    plt.ylabel("Mean covariance")
    plt.title("Mean Covariance vs. Cluster Size (colored by error mean)")
    plt.ylim(bottom=0, top=ylim_top)
    os.makedirs(save_folder, exist_ok=True)
    save_fig = os.path.join(save_folder, f"correlation_covs_n_clusters_ylim-{ylim_top}_{suffix}.png")
    plt.tight_layout()
    plt.savefig(save_fig, dpi=160)
    plt.close()


def plot_covs_trajectory_per_dim(trajectory, save_folder, start=0, suffix=""):
    """
    trajectory: List of covariances (T, K, D)
    """
    covs_stack = torch.stack(trajectory).reciprocal()  # (T, K, D)
    mean_covs = covs_stack.mean(dim=1)    # (T, D)

    plt.figure(figsize=(6,4))
    for d in range(mean_covs.shape[1]):
        plt.plot(range(start, len(mean_covs)), mean_covs[start:, d].cpu().numpy(), label=f"Dim-{d}")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Covariance")
    plt.title("Mean Covariance vs Iteration")
    plt.legend()
    save_fig = os.path.join(save_folder, f"mean_covs_trajectory{suffix}.png")
    plt.savefig(save_fig, dpi=160)
    plt.close()

def plot_weights_trajectory(trajectory, save_folder, start=0,  suffix=""):
    """
    trajectory: List of weights (T, K)
    """
    weights_stack = torch.stack(trajectory)  # (T, K)

    plt.figure(figsize=(6,4))
    for k in range(weights_stack.shape[1]):
        plt.plot(range(start, len(weights_stack)), weights_stack[start:, k].cpu().numpy(), label=f"Cluster-{k}")
    plt.xlabel("Iteration")
    plt.ylabel("Weight")
    plt.title("Weights vs Iteration")
    plt.legend()
    save_fig = os.path.join(save_folder, f"weights_trajectory{suffix}.png")
    plt.savefig(save_fig, dpi=160)
    plt.close()

def plot_num_samples_per_cluster(trajectory_clusters, save_folder, start=0,  n_cluster =10, ylim=None,suffix="",
    sort=True, filter_zero_clusters=False):
    """
    trajectory: List of weights (T, K)
    """
    trajectory_cluster = torch.stack(trajectory_clusters)  # (T, N)

    num_samples_per_cluster = torch.zeros(trajectory_cluster.shape[0], n_cluster).to(trajectory_cluster.device)  # (T, K)

    for t in range(trajectory_cluster.shape[0]):
        num_samples_per_cluster[t] = torch.bincount(trajectory_cluster[t], minlength=n_cluster)

    plt.figure(figsize=(6,4))
    for k in range(n_cluster):
        if filter_zero_clusters and (num_samples_per_cluster[-1, k] == 0):
            continue
        plt.plot(range(start, len(num_samples_per_cluster)), num_samples_per_cluster[start:, k].cpu().numpy(), label=f"Cluster-{k}")
    plt.xlabel("Iteration")
    plt.ylabel("Number of Samples")
    plt.title("Number of Samples per Cluster vs Iteration")
    plt.legend()
    if ylim is not None:
        plt.ylim(top=ylim, bottom=0)
    save_fig = os.path.join(save_folder, f"num_samples_per_cluster_ylim-{ylim}_{suffix}.png")
    plt.savefig(save_fig, dpi=160)
    plt.close()


def plot_means_trajectory(trajectory, save_folder, start=0, suffix=""):
    """
    trajectory: List of means (T, K, D)
    """

    means_stack = torch.stack(trajectory).squeeze()  # (T, K, D)
    print("means_stack shape:", means_stack.shape)
    mean_means = means_stack.mean(dim=[2])    # (T, D)
    num_zero_means = means_stack.abs().sum(dim=[2]).eq(0).sum(dim=1)  # (T,)
    plt.figure(figsize=(6,4))
    for k in range(mean_means.shape[1]):
        plt.plot(range(start, len(mean_means)), mean_means[start:, k].cpu().numpy(), label=f"Cluster-{k}")
    # plt.plot(range(start, len(mean_means)), mean_means[start:].cpu().numpy(), label=f"Mean of Means")
    # plt.plot(range(start, len(mean_means)), mean_means[start:].cpu().numpy(), label=f"Mean of Means")
    #plt.plot(range(start, len(num_zero_means)), num_zero_means[start:].cpu().numpy(), label=f"Number of Zero Means")
    plt.xlabel("Iteration")
    plt.ylabel("Mean of Means")
    plt.title("Mean of Means vs Iteration")
    # plt.ylim(top=0.01)
    plt.legend()

    save_fig = os.path.join(save_folder, f"mean_of_means_trajectory{suffix}.png")
    plt.savefig(save_fig, dpi=160)
    plt.close()

def plot_means_axis_trajectory(trajectory, save_folder, start=0, dim=0, suffix="",
                                 filter_zero_clusters=False, list_clusters_result=None):
    """
    trajectory: List of means (T, K, D)
    """

    means_stack = torch.stack(trajectory).squeeze()  # (T, K, D)
    print("means_stack shape:", means_stack.shape)

    means_axis = means_stack[:, :, dim]  # (T, K)
    num_zero_means = means_stack.abs().sum(dim=[2]).eq(0).sum(dim=1)  # (T,)
    if filter_zero_clusters and (list_clusters_result is not None):
        # Get the clusters that are non-empty in the final iteration
        final_clusters = list_clusters_result[-1]
        non_empty_clusters = torch.unique(final_clusters).tolist()
        print("Non-empty clusters:", non_empty_clusters)
        means_axis = means_axis[:, non_empty_clusters]
    plt.figure(figsize=(6,4))
    for k in range(means_axis.shape[1]):
    
        plt.plot(range(start, len(means_axis)), means_axis[start:, k].cpu().numpy(), label=f"Cluster-{k}")
    # plt.plot(range(start, len(mean_means)), mean_means[start:].cpu().numpy(), label=f"Mean of Means")
    # plt.plot(range(start, len(mean_means)), mean_means[start:].cpu().numpy(), label=f"Mean of Means")
    #plt.plot(range(start, len(num_zero_means)), num_zero_means[start:].cpu().numpy(), label=f"Number of Zero Means")
    plt.xlabel("Iteration")
    plt.ylabel("Mean of Means")
    plt.title("Mean of Means vs Iteration")
    # plt.ylim(top=0.01)
    plt.legend()
    save_fig = os.path.join(save_folder, f"means_axis_trajectory_dim-{dim}{suffix}.png")
    plt.savefig(save_fig, dpi=160)
    plt.close()

def plot_num_nans(trajectory, save_folder, start=0, dim=0, suffix=""):
    """
    trajectory: List of means (T, K, D)
    """

    means_stack = torch.stack(trajectory).squeeze()  # (T, K, D)
    num_zero_means = means_stack.abs().sum(dim=[2]).isnan().sum(dim=1)  # (T,)
    plt.figure(figsize=(6,4))

    plt.plot(range(start, len(num_zero_means)), num_zero_means[start:].cpu().numpy(), label=f"Number of Zero Means")
    plt.xlabel("Iteration")
    plt.ylabel("Number of nans")
    plt.title("Number of nans vs Iteration")
    # plt.ylim(top=0.01)
    plt.legend()
    save_fig = os.path.join(save_folder, f"num_nans_trajectory_dim-{dim}{suffix}.png")
    plt.savefig(save_fig, dpi=160)
    plt.close()

def plot_dead_resp(list_resp_dead, save_folder, suffix=""):
    """
    list_resp_dead: List of tensors (T, K) with number of dead responsibilites per cluster
    """

    resp_dead_stack = [int(torch.as_tensor(l).numel()) for l in list_resp_dead]
    plt.figure(figsize=(6,4))
    plt.plot(range(len(resp_dead_stack)), resp_dead_stack, label=f"Number of Dead Responsibilites")
    plt.xlabel("Iteration")
    plt.ylabel("Number of Dead Responsibilites")
    plt.title("Number of Dead Responsibilites vs Iteration")
    plt.legend()
    save_fig = os.path.join(save_folder, f"num_dead_responsibilities_trajectory{suffix}.png")
    plt.savefig(save_fig, dpi=160)
    plt.close()

if __name__ == "__main__":
    gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    list_alpha = [0.0001, 0.001, 0.005, 0.01, 0.1, 0.7]

    seed = 2
    supervised = False
    n_clusters = 10
    max_iter = 1999
    lr = 0.0001
    lr_temp = 0.1
    lr_min = None
    l2_reg = 0.01
    reg_covar = 0.00001   

    loss_weights = [0, 2.0, 1.0]
    temp = 32
    train_temp = True
    

    # Parameterization settings
    precisions_type = "diag"
    init_scheme_means = "random"
    init_scheme_precision = "statistics"

    # Preprocessing settings
    order = True

    # Benchmark settings
    model_name = "resnet34" #timm-vit-tiny16
    data_name = "cifar10"
    n_res = 2000
    seed_split = 9
    bound ="bernstein"
    n_samples= 2000
    subclasses =  [5, 8] #    

    quantizer = "soft_kmeans"
    learn_precisions = True
    

    suffix = ""
    save_folder_base = f"./code/utils/clustering/{quantizer}_results/"
    if quantizer == "soft_kmeans_gd":
       if learn_precisions:
           save_folder_base = os.path.join(save_folder_base, "learn-precisions")
       else:
           save_folder_base = os.path.join(save_folder_base, "fix-precisions")
    elif quantizer == "tim":
         save_folder_base = f"./code/utils/quantizers/{quantizer}/{data_name}_{model_name}/seed-split-{seed_split}/"
         if train_temp:
            save_folder_base = os.path.join(save_folder_base, "train-temp")

    if subclasses is not None:
        subclasses_str = "-".join([str(c) for c in subclasses])
        # save_folder_base = f"./code/utils/clustering/{quantizer}_results/fix-precisions/subclass-{subclasses_str}"
        save_folder_base = os.path.join(save_folder_base, f"subclass-{subclasses_str}")
    
    if quantizer == "soft_kmeans":
        file = f"{data_name}_{model_name}_seed-split-{seed_split}_n-{n_samples}_iter-{max_iter}_K-{n_clusters}_bound-{bound}_init-{init_scheme_means}"
        if reg_covar != 1e-6:
            file += f"_regcovar-{reg_covar}"
        file += f"/seed_{seed}/"
    elif quantizer == "soft_kmeans_gd":
        file = f"{data_name}_{model_name}_seed-split-{seed_split}_n-{n_samples}_iter-{max_iter}_lr-{lr}_lr_min-{lr_min}_K-{n_clusters}-l2_reg-{l2_reg}/seed_{seed}/"
    elif quantizer == "tim":
        file = f"n_clusters-{n_clusters}_loss_weights-{loss_weights[1]}-{loss_weights[2]}-lr-{lr}_temp-{temp}_iter-{max_iter}_bound-{bound}/"
        if train_temp:
            file = f"n_clusters-{n_clusters}_loss_weights-{loss_weights[1]}-{loss_weights[2]}-lr_w-{lr}-lr_temp-{lr_temp}_temp-{temp}_iter-{max_iter}_bound-{bound}/"

    latent_path = f"./latent/{data_name}_{model_name}/seed-split-{seed_split}/res_n-samples-{n_res}_transform-test_n-epochs-1.pt"
    probits, errors = read_probits(latent_path, order=order, subclasses=subclasses)


    save_folder = os.path.join(save_folder_base, file)
    if quantizer == "tim":
        params_history = torch.load(os.path.join(save_folder, f"quantizer_params_history{suffix}.pt"), weights_only=False) # Dict of List of (N, K, D)
        list_clusters_res = torch.load(os.path.join(save_folder, f"quantizer_list_clusters_res{suffix}.pt"), weights_only=False) # List of (N,)
        soft_preds_res = torch.load(os.path.join(save_folder, f"quantizer_soft_preds_res{suffix}.pt"), weights_only=False).squeeze() # List of (N, K)

        print("list_clusters_res[-1][:20]:", list_clusters_res[-1][:20])
        # exit()
        n, K = soft_preds_res.shape
        samples = [i for i in range(20)]
        for sample in samples:
            print(f"Sample {sample} :", probits[sample])
            print("cluster assignment probs:", soft_preds_res[sample][[25, 30, 41]].cpu().numpy())
            print("cluster assignment:", list_clusters_res[-1][sample].item())
        # exit()
        print("weights:", params_history["weights"][-1][0, [25, 30, 41], :].cpu().numpy())
        print("weights:", params_history["weights"][-1][0, [25, 30, 41], :].sum(dim=1).cpu().numpy())
        exit()
       
        sample = 2
        # for sample in samples:
        #     print(f"Sample {sample} :", probits[sample])
        #     plt.plot(range(K), soft_preds_res[sample].cpu().numpy(), marker='o', label=f"Sample-{sample}")
        #     plt.xlabel("Cluster")
        #     plt.ylabel("Soft Assignment Probability")
        #     plt.title(f"Soft Assignment Probabilities for Sample-{sample}")
        print(f"Sample {sample} :", probits[sample])
        plt.plot(range(K), soft_preds_res[sample].cpu().numpy(), marker='o', label=f"Sample-{sample}")
        plt.xlabel("Cluster")
        plt.ylabel("Soft Assignment Probability")
        plt.title(f"Soft Assignment Probabilities for Sample-{sample}")
        plt.tight_layout()
        plt.legend()
        save_path = os.path.join(save_folder, f"soft_assignment_sample-{sample}{suffix}.png")
        plt.savefig(save_path, dpi=160)
        plt.close()
        
        plot_means_trajectory(
            trajectory=params_history["weights"],
            save_folder=save_folder,
            suffix=suffix,
            start=0
        )
        
        plot_num_samples_per_cluster(
            trajectory_clusters=list_clusters_res,
            save_folder=save_folder,
            suffix=suffix,
            n_cluster=n_clusters,
            start=0,
            ylim=None,
            filter_zero_clusters=True
        )

        plot_means_axis_trajectory(
            trajectory=params_history["weights"],
            save_folder=save_folder,
            suffix=suffix,
            start=0,
            dim=0,
            filter_zero_clusters=True,
            list_clusters_result=list_clusters_res
        )

    else:
        results = torch.load(os.path.join(save_folder, f"results{suffix}.pt"), weights_only=False) # List of (N, K, D)

        # print(list(results._asdict().keys()))

        # # save_folder  = f"./code/utils/clustering/{data_name}_{model_name}_seed-split-{seed_split}_n-{n_samples}_iter-{max_iter}_lr-{lr}_lr_min-{lr_min}_K-{n_clusters}_with_logdet_fix_cov-{sig}/seed_{seed}/"
        # supervised_str = "supervised" if supervised else "unsupervised"
        # if subclasses is not None:
        #     subclasses_str = "-".join([str(c) for c in subclasses])
        #     save_folder_base  = f"./code/utils/clustering/mutinfo_{supervised_str}_results/learn_diag/subclass-{subclasses_str}"
        #     save_folder_skmeans_base  = f"./code/utils/clustering/soft_kmeans_results/subclass-{subclasses_str}"
        # else:
        #     save_folder_base  = f"./code/utils/clustering/mutinfo_{supervised_str}_results/learn_diag"
        #     save_folder_skmeans_base  = f"./code/utils/clustering/soft_kmeans_results/"
        
        # precisions_dic = {}
        # for alpha in list_alpha:
        #     save_folder  = os.path.join(save_folder_base, f"{data_name}_{model_name}_seed-split-{seed_split}_n-{n_samples}_iter-{max_iter}_lr-{lr}_lr_min-{lr_min}_K-{n_clusters}_alpha-{alpha}/seed_{seed}/")
        #     results = torch.load(os.path.join(save_folder, "results.pt"), weights_only=False)
            
        #     precision_means_trajectory = torch.stack(results.precisions_history).mean(dim=(1,2))  # (D,)
        #     precisions_dic[alpha] = precision_means_trajectory

        # save_folder_soft_kmeans = os.path.join(save_folder_skmeans_base, f"{data_name}_{model_name}_seed-split-{seed_split}_n-{n_res}_K-{n_clusters}_bound-{bound}_init-{init_scheme_means}/seed_{seed}/")
        # results = torch.load(os.path.join(save_folder_soft_kmeans, "results.pt"), weights_only=False)
        # precision_means_trajectory_skmeans = torch.stack(results.params_history["covariances"]).reciprocal().mean(dim=(1,2))  # (D,)


        
        # plot_means_covs_trajectory_per_cluster(
        #     trajectory=results.params_history["covariances"],
        #     save_folder=save_folder,
        #     suffix=suffix,
        #     start=2,
        #     scale="log",
        # )
        

        # plot_correlation_covs_n_clusters(
        #     covs_trajectory=results.params_history["covariances"],
        #     clusters_trajectory=results.list_clusters_res,
        #     errors=errors,
        #     n_clusters=n_clusters,
        #     save_folder=save_folder,
        #     suffix=suffix,
        #     ylim_top=0.000005
        # )

        # plot_means_trajectory(
        #     trajectory=results.params_history["means"],
        #     save_folder=save_folder,
        #     suffix=suffix,
        #     start=0
        # )

        # plot_num_samples_per_cluster(
        #     trajectory_clusters=results.list_clusters_res,
        #     save_folder=save_folder,
        #     suffix=suffix,
        #     n_cluster=n_clusters,
        #     start=0
        # )
        print(results.list_resp_dead[50])

        plot_dead_resp(
            list_resp_dead=results.list_resp_dead,
            save_folder=save_folder,
            suffix=suffix
        )
        means = get_clusters_info(errors, results.list_clusters_res[-1], torch.tensor([n_clusters], device=device), bound=bound)[0].squeeze(0)
        plot_cluster_sizes(
            clusters=results.list_clusters_res[-1],
            save_folder=save_folder,
            suffix=suffix,
            error_means=means,
            split_name="res",
            sort=False
        )








        # plot_weights_trajectory(
        #     trajectory=results.params_history["weights"],
        #     save_folder=save_folder,
        #     suffix=suffix,
        #     start=0
        # )

        # plt.figure(figsize=(6,4))
        # for alpha in list_alpha:
        #     plt.plot(range(len(precisions_dic[alpha])), precisions_dic[alpha].cpu().numpy(), label=f"alpha={alpha}")
        # #plt.plot(range(len(precision_means_trajectory_skmeans)), precision_means_trajectory_skmeans.cpu().numpy(), label=f"soft k-means")
        # plt.xlabel("Iteration")
        # plt.ylabel("Precision")
        # plt.title("Precision vs Iteration")
        # plt.legend()
        # save_fig = os.path.join(save_folder_base, f"{data_name}_{model_name}_seed-split-{seed_split}_n-{n_samples}_iter-{max_iter}_lr-{lr}_lr_min-{lr_min}_K-{n_clusters}/seed_{seed}")
        # os.makedirs(save_fig, exist_ok=True)
        # plt.savefig(os.path.join(save_fig, f"precision_vs_alpha.png"), dpi=160)
        # plt.show()


