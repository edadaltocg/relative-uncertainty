#
# from turtle import pd
from typing import Any, Optional, Tuple, Union
from warnings import warn
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch import LongTensor, Tensor
from code.utils.clustering.utils import plot_means_covs_trajectory_per_cluster, read_probits, get_clusters_info, plot_eval, plot_cluster_sizes, plot_num_samples_per_cluster, plot_metric_corr
import pandas as pd
from code.utils.metrics import compute_all_metrics
import matplotlib.pyplot as plt
import os


from .distances import (
    BaseDistance,
    CosineSimilarity,
    DotProductSimilarity,
    LpDistance,
)
from .utils import ClusterResult, group_by_label_mean

# import numpy as np
# from sklearn.cluster._kmeans import _kmeans_plusplus, row_norms

__all__ = ["KMeans"]


#

class KMeans(nn.Module):
    """
    Implements k-means clustering in terms of
    pytorch tensor operations which can be run on GPU.
    Supports batches of instances for use in
    batched training (e.g. for neural networks).

    Partly based on ideas from:
        - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        - https://github.com/overshiki/kmeans_pytorch


    Args:
            init_method: Method to initialize cluster centers ['rnd', 'k-means++']
                            (default: 'rnd')
            num_init: Number of different initial starting configurations,
                        i.e. different sets of initial centers (default: 8).
            max_iter: Maximum number of iterations (default: 100).
        
      
            tol: Relative tolerance with regards to Frobenius norm of the difference
                        in the cluster centers of two consecutive iterations to
                        declare convergence. (default: 1e-4)
            normalize: String id of method to use to normalize input.
                        one of ['mean', 'minmax', 'unit'].
                        None to disable normalization. (default: None).
            n_clusters: Default number of clusters to use if not provided in call
                    (optional, default: 8).
            verbose: Verbosity flag to print additional info (default: True).
            seed: Seed to fix random state for randomized center inits
                    (default: True).
            **kwargs: additional key word arguments for the distance function.
    """

    INIT_METHODS = ["random", "k-means++", "kmeans", "supervised"]
    NORM_METHODS = ["mean", "minmax", "unit"]

    def __init__(
        self,
        init_method: str = "random",
        num_init: int = 8,
        max_iter: int = 300,
        tol: float = 1e-8,
        n_clusters: Optional[int] = 8,
        verbose: bool = True,
        seed: Optional[int] = 123,
        is_for_init=False,
        res_errors=None,
        val_errors=None,
        val_probits=None,
        cal_probits=None,
        cal_errors=None,
        test_probits=None,
        test_errors=None,
        bound="bernstein",
        device=torch.device("cpu"),
        collect_info: bool = True,
        **kwargs,
    ):
        super(KMeans, self).__init__()
        self.init_method = init_method.lower()
        self.num_init = num_init
        self.max_iter = max_iter
        self.tol = tol
        # self.normalize = normalize
        self.n_clusters = n_clusters
        self.verbose = verbose
        self.device = device
        self.seed = seed

        self._check_params()
        self.is_for_init = is_for_init

        self.eps = None
        self._k_max = None
        self.results = None
        self.n_iter = None  # number of iterations run in last fit
        
        self.res_errors = res_errors
        self.val_errors = val_errors
        self.val_probits = val_probits
        self.cal_probits = cal_probits
        self.cal_errors = cal_errors
        self.test_probits = test_probits
        self.test_errors = test_errors
        self.bound = bound

        self.collect_info = collect_info

        self._init_info()



    @property
    def is_fitted(self) -> bool:
        """True if model was already fitted."""
        return self.results is not None

    @property
    def num_clusters(self) -> Union[int, Tensor, Any]:
        """
        Number of clusters in fitted model.
        Returns a tensor with possibly different
        numbers of clusters per instance for whole batch.
        """
        if not self.is_fitted:
            return None
        return self.results.k

    def _check_params(self):
        if self.init_method not in self.INIT_METHODS:
            raise ValueError(
                f"unknown <init_method>: {self.init_method}. "
                f"Please choose one of {self.INIT_METHODS}"
            )
        if self.num_init <= 0:
            raise ValueError(f"num_init should be > 0, but got {self.num_init}.")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter should be > 0, but got {self.max_iter}.")
      
        # if self.tol < 0 or self.tol > 1:
        #     raise ValueError(f"tol should be > 0 and < 1, but got {self.tol}.")
        # # if isinstance(self.normalize, bool):
        # #     if self.normalize:
        # #         self.normalize = "mean"
        # #     else:
        # #         self.normalize = None
        # if self.normalize is not None and self.normalize not in self.NORM_METHODS:
        #     raise ValueError(
        #         f"unknown <normalize> method: {self.normalize}. "
        #         f"Please choose one of {self.NORM_METHODS}"
        #     )
        if self.n_clusters is not None and self.n_clusters < 2:
            raise ValueError(f"n_clusters should be > 1, but got {self.n_clusters}.")

    def _check_x(self, x) -> Tensor:
        """Check and (re-)format input samples x."""
        if not isinstance(x, Tensor):
            raise TypeError(f"x has to be a torch.Tensor but got {type(x)}.")
        shp = x.shape
        if len(shp) < 2:
            raise ValueError(
                f"input <x> should be at least of shape (N, D) "
                f"with number of points N and number of dimensions D but got {shp}."
            )
        elif len(shp) > 2:
            x = x.squeeze()
            x = self._check_x(x)
        self.eps = torch.finfo(x.dtype).eps
        return x

    def _check_k(
        self, k,  device: torch.device = torch.device("cpu")
    ) -> LongTensor:
        """Check and (re-)format number of clusters k."""
         
        if not isinstance(k, Tensor):
            if k is None:  # use specified default number of clusters
                if self.n_clusters is None:
                    raise ValueError(
                        "Did not provide number of clusters k on call and "
                        "did not specify default 'n_clusters' at initialization."
                    )
                k = self.n_clusters

            if isinstance(k, int):  # convert to tensor
                k = torch.tensor([k], dtype=torch.long)
                # print("k is int")
                # print('k shape:', k.shape)
            else:
                raise TypeError(
                    f"k has to be int, torch.Tensor or None " f"but got {type(k)}."
                )
        if len(k.shape) > 1:
            k = k.squeeze()
            assert len(k.shape) == 1
        # if k.shape[0] == 1:
        #     k = k.repeat(bs)
        if (k >= self.n).any():
            raise ValueError(
                f"Specified 'k' must be smaller than "
                f"number of samples n={self.n}, but got: {k}."
            )
        if (k <= 1).any():
            raise ValueError("Clustering for k=1 is ambiguous.")
        self._k_max = int(k.max())
        return k.to(dtype=torch.long, device=device)

    def _check_centers(
        self,
        centers,
        dims: Tuple,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        if not isinstance(centers, Tensor):
            raise TypeError(
                f"centers has to be a torch.Tensor " f"but got {type(centers)}."
            )
        bs, n, d = dims
        if len(centers.shape) == 3:
            if (
                centers.size(0) != bs
                or centers.size(1) != self._k_max
                or centers.size(2) != d
            ):
                raise ValueError(
                    f"centers needs to be of shape "
                    f"({bs}, {self._k_max}, {d}),"
                    f"but got {tuple(centers.shape)}."
                )
            if self.num_init > 1:
                warn(
                    f"Specified num_init={self.num_init} > 1 but provided "
                    f"only 1 center configuration per instance. "
                    f"Using same center configuration for all {self.num_init} runs."
                )
                # expand to num_init size
                centers = centers[:, None, :, :].expand(
                    centers.size(0), self.num_init, centers.size(1), centers.size(2)
                )
            else:
                centers = centers.unsqueeze(1)
        elif len(centers.shape) == 4:
            if (
                centers.size(0) != bs
                or centers.size(1) != self.num_init
                or centers.size(2) != self._k_max
                or centers.size(3) != d
            ):
                raise ValueError(
                    f"centers needs to be of shape "
                    f"({bs}, {self.num_init}, {self._k_max}, {d}),"
                    f"but got {tuple(centers.shape)}."
                )
        else:
            raise ValueError(
                f"centers have unsupported shape of "
                f"{tuple(centers.shape)} "
                f"instead of "
                f"({bs}, {self.num_init}, {self._k_max}, {d})."
            )
        return centers.contiguous().to(dtype=dtype, device=device)

    def forward(
        self,
        x: Tensor,
        k: Optional[Union[LongTensor, Tensor, int]] = None,
        centers: Optional[Tensor] = None,
        **kwargs,
    ) -> ClusterResult:
        """torch.nn like forward pass.

        Args:
            x: input features/coordinates (BS, N, D)
            k: optional batch of (possibly different)
                numbers of clusters per instance (BS, )
            centers: optional batch of initial centers to use (BS, K, D)
            **kwargs: additional kwargs for initialization or cluster procedure

        Returns:
            ClusterResult tuple

        """
        x = self._check_x(x)
        self.n, self.d = x.shape
        x_ = x
        
        k = self._check_k(k,  device=x.device)
        self.bs = k.shape[0]

        # normalize input
        # if self.normalize is not None:
        #     x = self._normalize(x, self.normalize, self.eps)
        # init centers
        if centers is None:
            centers = self._center_init(x, k, **kwargs)
        centers = self._check_centers(
            centers, dims=(self.bs, self.n, self.d), dtype=x.dtype, device=x.device
        )

        if not self.is_for_init:
            self._cluster(
                x, centers, k, **kwargs
            )
      
        else:
            centers = self._cluster(
                x, centers, k, **kwargs
            )
            return centers

    def fit(
        self,
        x: Tensor,
        k: Optional[Union[LongTensor, Tensor, int]] = None,
        centers: Optional[Tensor] = None,
        **kwargs,
    ) -> nn.Module:
        """Compute cluster centers and predict cluster index for each sample.

        Args:
            x: input features/coordinates (N, D)
            k: optional batch of (possibly different)
                numbers of clusters per instance (BS, )
            centers: optional batch of initial centers to use (BS, K, D)
            **kwargs: additional kwargs for initialization or cluster procedure

        Returns:
            KMeans model
        """
        self(x, k=k, centers=centers, **kwargs)

    
    @torch.no_grad()
    def predict(self, x: Tensor, **kwargs) -> LongTensor:
        """Hard assignments (argmax over responsibilities)."""
        assert self.is_fitted, "Call fit() first."
        x = self._check_x(x)  # (bs, n, d)
        # if self.normalize is not None:
        #     x = self._normalize(x, self.normalize, self.eps)

        # Fitted params (one set per batch)
        x   = x.to(self.device)                            # (bs, n, d)
        centers    = self.results.centers.to(self.device)        # (bs, k_max, d)
        k_vec    = self.results.k.to(self.device)            # (bs,)

        n, d = x.shape
        bs, k_max, d2 = centers.shape
        assert  d == d2

        # Build invalid-component mask from k (don’t rely on any saved mask)
        k_range = torch.arange(k_max, device=self.device).expand(bs, -1)    # (bs, k_max)
        invalid = (k_range >= k_vec.unsqueeze(1)).unsqueeze(1)  

        labels = self._e_step(
            x, 
            centers.unsqueeze(1), 
            invalid
            )
        return labels.squeeze(1)  # type: ignore
        
        # def predict(self, x: Tensor, **kwargs) -> LongTensor:
        #     """Predict the closest cluster each sample in X belongs to.

        #     Args:
        #         x: input features/coordinates (BS, N, D)
        #         **kwargs: additional kwargs for assignment procedure

        #     Returns:
        #         batch tensor of cluster labels for each sample (BS, N)

        #     """
        #     assert self.is_fitted
        #     x = self._check_x(x)
        #     return self._assign(
        #         x, centers=self._result.centers[:, None, :, :], **kwargs
        #     ).squeeze(1)
    @torch.no_grad()
    def fit_predict(
        self,
        x: Tensor,
        k: Optional[Union[LongTensor, Tensor, int]] = None,
        centers: Optional[Tensor] = None,
        **kwargs,
    ) -> LongTensor:
        """Compute cluster centers and predict cluster index for each sample.

        Args:
            x: input features/coordinates (BS, N, D)
            k: optional batch of (possibly different)
                numbers of clusters per instance (BS, )
            centers: optional batch of initial centers to use (BS, K, D)
            **kwargs: additional kwargs for initialization or cluster procedure

        Returns:
            batch tensor of cluster labels for each sample (BS, N)

        """
        self(x, k=k, centers=centers, **kwargs)
        return self.results.labels.to(self.device)

    @torch.no_grad()
    def _center_init(self, x: Tensor, k: LongTensor, **kwargs) -> Tensor:
        """Wrapper to apply different methods for
        initialization of initial centers (centroids)."""
        if self.init_method == "random":
            return self._init_rnd(x, k)
        elif self.init_method == "k-means++":
            return self._init_plus(x, k)
        elif self.init_method == "supervised":
            return self._init_supervised(x, k)
        else:
            raise ValueError(f"unknown initialization method: {self.init_method}.")

    # @staticmethod
    # def _normalize(x: Tensor, normalize: str, eps: float = 1e-8):
    #     """Normalize input samples x according to specified method:

    #     - mean: subtract sample mean
    #     - minmax: min-max normalization subtracting sample min and divide by sample max
    #     - unit: normalize x to lie on D-dimensional unit sphere

    #     """
    #     if normalize == "mean":
    #         x -= x.mean(dim=1)[:, None, :]
    #     elif normalize == "minmax":
    #         x -= x.min(-1, keepdims=True).values  # type: ignore
    #         x /= x.max(-1, keepdims=True).values  # type: ignore
    #     elif normalize == "unit":
    #         # normalize x to unit sphere
    #         z_msk = x == 0
    #         x = x.clone()
    #         x[z_msk] = eps
    #         x = torch.diag_embed(1.0 / (torch.norm(x, p=2, dim=-1))) @ x
    #     else:
    #         raise ValueError(f"unknown normalization type {normalize}.")
    #     return x

    def _init_rnd(self, x: torch.Tensor, k: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            x: (N, D)  -- shared dataset (no batch dim)
            k: (BS,)   -- clusters per batch item (we’ll use k_max)

        Returns:
            centers: (BS, num_init, k_max, D)  -- same init for each batch item
        """
        N, D = x.shape
        bs = int(k.shape[0])
        k_max = int(k.max())

        if self.seed is not None:
            gen = torch.Generator(device=x.device)
            gen.manual_seed(self.seed)
        else:
            gen = None

        # (num_init, k_max) distinct row indices per init
        # probs is uniform over N rows
        probs = torch.full((self.num_init, N), 1.0 / N, device=x.device, dtype=x.dtype)

        idx   = torch.multinomial(probs, num_samples=k_max, replacement=False, generator=gen)  # (num_init, k_max), Long

        # Select those rows from x
        base_centers = x.index_select(0, idx.reshape(-1)).view(self.num_init, k_max, D).contiguous()
        # Make per-batch copies (don’t use expand if you’ll modify centers)
        centers = base_centers.unsqueeze(0).repeat(bs, 1, 1, 1).contiguous()  # (bs, num_init, k_max, D)

        return centers
    def _init_supervised(self, x, k):


        if self.seed is not None:
            gen = torch.Generator(device=x.device)
            gen.manual_seed(self.seed)
        else:
            gen = None
        
        mean_error_res =  self.res_errors.float().mean()
        n_cluster_err = int(k[0].item() * mean_error_res)
        if n_cluster_err == 0:
            n_cluster_err = 1
        n_cluster_noerr = int(k[0].item() - n_cluster_err)
        n_clusters_dic = {'err': n_cluster_err, 'noerr': n_cluster_noerr}

        n_errors = int(self.res_errors.sum().item())
        n_noerrors = int(self.res_errors.shape[0] - n_errors)
        n_samples_dic = {'err': n_errors, 'noerr': n_noerrors}
        x_dic = {'err': x[self.res_errors==1], 'noerr': x[self.res_errors==0]}
        base_centers = []
        for err_type in n_clusters_dic.keys():
            n = n_samples_dic[err_type]
            k = n_clusters_dic[err_type]
            # print("k for", err_type, ":", k)

            probs = torch.full((self.num_init, n), 1.0 / n, device=x.device, dtype=x.dtype)
            idx   = torch.multinomial(probs, num_samples=k, replacement=False, generator=gen)  # (num_init, k_max), Long
            centers = x_dic[err_type].index_select(0, idx.reshape(-1)).view(self.num_init, k, self.d).contiguous()
           # print("shape of centers:", centers.size())
            base_centers.append(centers)
        base_centers = torch.cat(base_centers, dim=1).unsqueeze(0).repeat(self.bs, 1, 1, 1).contiguous()
        return base_centers

            
        

        

    
    @torch.no_grad()
    def _init_plus(self, x: torch.Tensor, k: torch.LongTensor) -> torch.Tensor:
        """
        x: (n, d) shared dataset (no batch dim)
        k: (bs,) number of clusters per batch
        return: centers (bs, num_init, k_max, d) with the same init for all bs
        """
        n, d = x.shape
        bs,  = k.shape
        m     = self.num_init
        k_max = int(k.max())

        # RNG
        gen = None
        if self.seed is not None:
            gen = torch.Generator(device=x.device)
            gen.manual_seed(self.seed)

        # We build a single base init (m, k, d) and repeat for bs
        centers = torch.empty((m, k_max, d), dtype=x.dtype, device=x.device)

        # Keep just a 2D "picked" mask and the current min squared distances
        picked = torch.zeros((m, n), dtype=torch.bool, device=x.device)

        # Precompute x^2 once
        x2 = x.square().sum(dim=1)                    # (n,)

        # ---- 1) First center: uniform over rows per init ----
        assert n > m, f"n={n} must be > num_init={m}"
        probs0 = torch.full((m, n), 1.0 / n, dtype=torch.float32, device=x.device)
        idx = torch.multinomial(probs0, num_samples=1, replacement=False, generator=gen).squeeze(1)  # (m,)

        centers[:, 0, :] = x.index_select(0, idx)    # (m, d)
        picked[torch.arange(m, device=x.device), idx] = True

        # Initialize best min-squared-distance to chosen set: distance to first center
        c = centers[:, 0, :]                          # (m, d)
        # d2_to_c[r, i] = ||x_i - c_r||^2 = ||x_i||^2 + ||c_r||^2 - 2 x_i·c_r
        c2 = c.square().sum(dim=1)                    # (m,)
        xTc = torch.einsum('nd,md->mn', x, c)         # (m, n)
        best_d2 = (x2[None, :] + c2[:, None] - 2.0 * xTc).clamp_min_(0)  # (m, n)
        best_d2.masked_fill_(picked, 0)               # don't resample picked points

        eps = torch.finfo(x.dtype).eps

        # ---- 2) Remaining centers: incremental D^2 sampling ----
        for nc in range(1, k_max):
            # Sample next index using current best_d2
            pot = best_d2.clamp_min(eps)              # (m, n)
            next_idx = torch.multinomial(pot, num_samples=1, generator=gen).squeeze(1)  # (m,)

            # Add new center
            centers[:, nc, :] = x.index_select(0, next_idx)
            picked[torch.arange(m, device=x.device), next_idx] = True

            # Update best_d2 with distance to the newly picked center only (O(m*n))
            c = centers[:, nc, :]                     # (m, d)
            c2 = c.square().sum(dim=1)                # (m,)
            xTc = torch.einsum('nd,md->mn', x, c)     # (m, n)
            new_d2 = (x2[None, :] + c2[:, None] - 2.0 * xTc).clamp_min_(0)
            best_d2 = torch.minimum(best_d2, new_d2)
            best_d2.masked_fill_(picked, 0)

        # Repeat to all batch items
        return centers.unsqueeze(0).repeat(bs, 1, 1, 1).contiguous()   # (bs, m, k, d)
    
    @torch.no_grad()
    def _m_step(self, x, labels, k_max):

        n,d = x.shape
        bs, m, n_ = labels.shape
        K = int(k_max)
        M = torch.nn.functional.one_hot(labels, num_classes=K).to(x.dtype)  # (bs, m, N, K)
        M = M.permute(0, 1, 3, 2)                                            # (bs, m, K, N)

        nk = M.sum(dim=-1) + 1e-12                                           # (bs, m, K)
        means = torch.einsum('bmkN,Nd->bmkd', M, x)                          # (bs, m, K, D)
        means = means / nk[..., None]                                         # broadcast divide

        return means

    def _init_info(self):
        """Record additional info before clustering."""

        self.best_init = None
        self.inertia_history = torch.empty((self.num_init, self.max_iter), device='cpu')
        self.classif_results = {split: [[] for _ in range(self.num_init)] for split in ['res', 'cal', 'test']}
        self.classif_results_upper_res = {split: [[] for _ in range(self.num_init)] for split in ['res', 'val', 'test']}
        self.params_history = {"weights": []}
        self.obj_history = torch.empty((self.num_init, self.max_iter), device='cpu')
        self.list_clusters_res = []

    def _record_info(self, x, centers, invalid, k, iter):
        """Record additional info after clustering."""
        

        clusters_res = self._e_step(x, centers, invalid=invalid)
        clusters_cal = self._e_step(self.cal_probits, centers, invalid=invalid)
        clusters_test = self._e_step(self.test_probits, centers, invalid=invalid)
        clusters_val = self._e_step(self.val_probits, centers, invalid=invalid)

        clusters_dic = {"res": clusters_res,  "val": clusters_val, "cal": clusters_cal, "test": clusters_test}
        errors_dic = {"res": self.res_errors, "val": self.val_errors, "cal": self.cal_errors, "test": self.test_errors}
    

        _, upper = get_clusters_info(self.cal_errors, clusters_cal.squeeze(0), k, bound=self.bound)
        _, upper_res = get_clusters_info(self.res_errors, clusters_res.squeeze(0), k, bound=self.bound)
        
        for split in ["res", "val", "test"]:
            clusters = clusters_dic[split].squeeze(0)
            errors = errors_dic[split]
     
            scores =  upper_res.gather(1, clusters)
        
            for n_init in range(self.num_init):
                
                results = compute_all_metrics(
                    conf=scores[n_init].cpu(),
                    detector_labels=errors.cpu(),
                )
                results = pd.DataFrame([results])

            
                    
                self.classif_results_upper_res[split][n_init].append(results)

        

        for split in ["res", "cal", "test"]:
            clusters = clusters_dic[split].squeeze(0)
            errors = errors_dic[split]
     
            scores =  upper.gather(1, clusters)
        
            for n_init in range(self.num_init):
                
                results = compute_all_metrics(
                    conf=scores[n_init].cpu(),
                    detector_labels=errors.cpu(),
                )
                results = pd.DataFrame([results])

            
                    
                self.classif_results[split][n_init].append(results)



        self.list_clusters_res.append(clusters_res.detach().cpu().clone().squeeze(0))
        self.params_history["weights"].append(centers.detach().cpu().clone())
        inertia = self._calculate_inertia(x, centers, clusters_res)
        self.inertia_history[:, iter] = inertia.squeeze(0).cpu()
            

        

    @torch.no_grad()
    def _cluster(
        self, x: Tensor, centers: Tensor, k: LongTensor, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor, Union[Tensor, Any]]:
        """
        Run Lloyd's k-means algorithm.

        Args:
            x: (N, D)
            centers: (BS, num_init, k_max, D)
            k: (BS, )

        """
     
        n, d = x.size()
        bs, = k.shape
        # mask centers for which  k < k_max with inf to get correct assignment
        k_max = torch.max(k).cpu().item()
        k_max_range = torch.arange(k_max, device=x.device)[None, :].expand(bs, -1)
        k_mask = k_max_range >= k[:, None]
        k_mask = k_mask[:, None, :].expand(bs, self.num_init, -1)
        # print("k shape", k_mask.size())
                # Build invalid-component mask from k (don’t rely on any saved mask)
        k_max = int(k.max())
        k_max_range = torch.arange(k_max, device=x.device)[None, :].expand(bs, -1)
        invalid = (k_max_range >= k[:, None])[:, None, :].expand(bs, self.num_init, k_max)  # (bs,m,k)
  

        for i in tqdm.tqdm(range(self.max_iter), desc="KMeans clustering"):
            # print("centers size:", centers.size())
            # centers[k_mask] = float("inf")
            # print("centers size:", centers.size())
            old_centers = centers.clone()
            # get cluster assignments
            c_assign = self._e_step(x, centers, invalid=invalid)
            # update cluster centers
            # print("labels shape:", c_assign.size())
            centers = self._m_step(x, c_assign, k_max)

            if self.collect_info:
                self._record_info(x, centers, invalid=invalid, k=k, iter=i)


            if self.tol is not None:
                # calculate center shift
                shift = self._calculate_shift(centers, old_centers, p=2)
                if (shift < self.tol).all():
                    if self.verbose:
                        print(
                            f"Full batch converged at iteration "
                            f"{i+1}/{self.max_iter} "
                            f"with center shifts = "
                            f"{shift.view(-1, self.num_init).mean(-1)}."
                        )
                    self.n_iter = {i + 1}
                    break

        # select best rnd restart according to inertia
        if self.n_iter is None:
            self.n_iter = {self.max_iter}
        # centers[k_mask] = float("inf")
        c_assign = self._e_step(x, centers, invalid=invalid)

        if self.is_for_init:
            return centers 
        else:

            inertia = self._calculate_inertia(x, centers, c_assign)
            # best_init = torch.argmin(inertia, dim=-1)
            # b_idx = torch.arange(bs, device=x.device)
            self.results = ClusterResult(
                centers=centers.squeeze(0).cpu(),
                labels=c_assign.squeeze(0).cpu(),
                inertia=inertia.squeeze(0).cpu(),
                k=k,
                )

            # return (
            #     c_assign[b_idx, best_init],
            #     centers[b_idx, best_init],
            #     inertia[b_idx, best_init],
            # )

    def storage_bytes(t): 
        GIB = 1024 ** 3
        return t.untyped_storage().nbytes() / GIB
    
   

    def _pairwise_distance(self, x: Tensor, centers: Tensor, **kwargs):
        def storage_bytes(t): 
            GIB = 1024 ** 3
            return t.untyped_storage().nbytes() / GIB
        """
        x:       (bs, n, d)
        centers: (bs, num_init, k, d)
        retourne: (bs, num_init, n, k) distances L2
        """
        n, d = x.shape
        bs, num_init, k, d2 = centers.shape

        # Réplique x le long de num_init sans copie "réelle"
        # x_rep = x[:, None, :, :].expand(bs, num_init, n, d)  # (bs, num_init, n, d)
       # print(x_rep.shape, x_rep.is_contiguous(), x_rep.stride(), storage_bytes(x_rep))  # small

        # Aplatis les dims (bs, num_init) pour un matmul batched propre
        # X_flat = x_rep.reshape(-1, n, d)      # (B, n, d), B = bs*num_init
        # print(X_flat.is_contiguous(), storage_bytes(X_flat))                             
        # X = X_flat.contiguous()                     # (B, n, d)
        # print(X.is_contiguous(), storage_bytes(X))
        # C = centers.reshape(-1, k, d).contiguous()     # (B, k, d)

        # Normes au carré
        X2 = x.square().sum(dim=1, keepdim=True)           # (n, 1)
        C2 = centers.square().sum(dim=3)         # (bs, m , k)

        # Produits scalaires
        # (B, n, d) @ (B, d, k) -> (B, n, k)
        # XC = X @ C.transpose(1, 2)
        XC = torch.einsum('nd,bmkd->bmnk', x, centers) # (bs, m , n, k)


        # Distances au carré (numériquement sûr)
        dist2 = X2[None, None, :, :] + C2[:, :, None, :] - 2.0 * XC # (bs, m, n, k)
        dist2.clamp_(min=0.0)
        dist = dist2.sqrt_()                           # retire sqrt si tu veux l'inertie
        return dist
        # return dist.view(bs, num_init, n, k)

    # def _pairwise_distance(self, x: Tensor, centers: Tensor, **kwargs):
    #     """Calculate pairwise distances between samples in x and all centers."""
    #     # expand tensors to calculate pairwise distance over (d) dimensions
    #     # of each point (n) to each center (k_max)
    #     # for each random restart (num_init) in each batch instance (bs)
        
    #     bs, n, d = x.size()
    #     bs, num_init, k_max, d = centers.size()
    #     x = x[:, None, :, None, :].expand(bs, num_init, n, k_max, d).reshape(-1, d)
    #     centers = (
    #         centers[:, :, None, :, :].expand(bs, num_init, n, k_max, d).reshape(-1, d)
    #     )
    #     return self.distance.pairwise_distance(x, centers, **kwargs).view(
    #         bs, num_init, n, k_max
    #     )

    def _e_step(self, x: Tensor, centers: Tensor, invalid, **kwargs) -> LongTensor:
        """Infer cluster assignment for each sample in x."""
        # dist: (bs, num_init, n, k_max)
        dist = self._pairwise_distance(x, centers)
        # get cluster assignments (center with minimal distance)
      
        dist = dist.masked_fill(invalid[:, :, None, :], float('inf'))
        return torch.argmin(dist, dim=-1)  # type: ignore

    @staticmethod
    @torch.jit.script
    def _calculate_shift(centers: Tensor, old_centers: Tensor, p: int = 2) -> Tensor:
        """Calculate center shift w.r.t. centers from last iteration."""
        # calculate euclidean distance while replacing inf with 0 in sum
        d = torch.norm((centers - old_centers), p=p, dim=-1)
        d[d == float("inf")] = 0
        # sum(d, dim=-1)**2 -> use mean to be independent of number of points
        return torch.mean(d, dim=-1)

    @staticmethod
    @torch.jit.script
    def _calculate_inertia(x: Tensor, centers: Tensor, labels: Tensor) -> Tensor:
        """Compute sum of squared distances of samples
        to their closest cluster center."""
        n, d = x.size()
        bs, m, k, d = centers.shape
        assert m == labels.size(1)
        # select assigned center by label and calculate squared distance
        assigned_centers = centers.gather(
            index=labels[:, :, :, None].expand(
                labels.size(0), labels.size(1), labels.size(2), d
            ),
            dim=2,
        )
        # squared distance to closest center
        d = (
            torch.norm(
                (x[None, None, :, :].expand(bs, m, n, d) - assigned_centers), p=2, dim=-1
            )
            ** 2
        )
        d[d == float("inf")] = 0
        return torch.sum(d, dim=-1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"init: '{self.init_method}', "
            f"num_init: {self.num_init}, "
            f"max_iter: {self.max_iter}, "
            # f"distance: {self.distance}, "
            f"tolerance: {self.tol}, "
            f"normalize: {self.normalize}"
            f")"
        )




if __name__ == "__main__":

    gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    seed = 2
    cov_type = "diag"
    n_clusters = 92
    order = True
    init_scheme = "k-means++"  # random, k-means++, kmeans
    bound = "bernstein"
    max_iter = 300
    num_init = 20
 
    
    model_name = "timm-vit-tiny16" #timm-vit-tiny16
    data_name = "imagenet"
    n_res = 10000
    n_cal = 15000
    n_test = 25000
    seed_split = 9
    subclasses = None#  [5, 8] # [5, 4, 6, 8]  # [5, 8]
    temperature = 6.9

    suffix = ""

    # Real probits
    # latent_path = f"./latent/ablation/{data_name}_{model_name}_n_cal_with-res-{n_res}/seed-split-{seed_split}/res_n-samples-{n_res}_transform-test_n-epochs-1.pt"
    latent_path = f"./latent/{data_name}_{model_name}/seed-split-{seed_split}/res_n-samples-{n_res}_transform-test_n-epochs-1.pt"
    probits, errors = read_probits(latent_path, order=order, subclasses=subclasses, temperature=temperature)
    latent_path_cal = f"./latent/{data_name}_{model_name}/seed-split-{seed_split}/cal_n-samples-{n_cal}_transform-test_n-epochs-1.pt"
    cal_probits, cal_errors = read_probits(latent_path_cal, order=order, subclasses=subclasses, temperature=temperature)
    latent_path_test = f"./latent/{data_name}_{model_name}/seed-split-{seed_split}/test_n-samples-{n_test}.pt"
    test_probits, test_errors = read_probits(latent_path_test, order=order, subclasses=subclasses, temperature=temperature)
    print("Number of res samples:", probits.size(0))
    print("Number of cal samples:", cal_probits.size(0))
    print("Number of test samples:", test_probits.size(0))
    # print("Shape probits:", probits.size())
    # exit()

    probits = probits.to(device)
    errors = errors.to(device)
    test_probits = test_probits.to(device)
    test_errors = test_errors.to(device)
    cal_probits = cal_probits.to(device)
    cal_errors = cal_errors.to(device)

    n_tr = int(0.98 * n_res)
    tr_idx = np.arange(n_tr)
    va_idx = np.arange(n_tr, n_res)
    probits_res_tr = probits[tr_idx]
    errors_res_tr = errors[tr_idx]
    probits_res_va = probits[va_idx]
    errors_res_va = errors[va_idx]
    # print(errors)

    # save_folder  = f"./code/utils/clustering/{data_name}_{model_name}_seed-split-{seed_split}_n-{n_samples}_iter-{max_iter}_lr-{lr}_lr_min-{lr_min}_K-{n_clusters}_with_logdet_fix_cov-{sig}/seed_{seed}/"
   # save_folder  = f"./code/utils/clustering/soft_kmeans_results/{data_name}_{model_name}_seed-split-{seed_split}_n-{n_res}_K-{n_clusters}_bound-{bound}_init-{init_scheme}/seed_{seed}/"
    if subclasses is not None:
        subclasses_str = "-".join([str(c) for c in subclasses])
        save_folder  = f"./code/utils/clustering/kmeans_results/subclass-{subclasses_str}"
    else:
        save_folder  = f"./code/utils/clustering/kmeans_results"
    file =f"{data_name}_{model_name}_seed-split-{seed_split}/n-{n_res}_iter-{max_iter}_K-{n_clusters}_bound-{bound}_init-{init_scheme}_numinit-{num_init}"
    if temperature != 2:
        file += f"_temp-{temperature}"
    save_folder  = os.path.join(save_folder, file, f"seed_{seed}")
    # save_folder  = f"./code/utils/clustering/toy_example_with_logdet_campled_min-{var_min}_cov/seed_{seed}/"
    os.makedirs(save_folder, exist_ok=True)
    print("save filer:", save_folder)

   

    quantizer = KMeans(
                seed=seed,
                init_method=init_scheme,
                max_iter = max_iter, 
                num_init=num_init, 
                verbose=0, 
                cal_errors=cal_errors,
                cal_probits=cal_probits,
                test_errors=test_errors,
                test_probits=test_probits,
                res_errors=errors_res_tr,
                val_probits=probits_res_va,
                val_errors=errors_res_va,
                bound=bound,
                tol=None,
                device=device,
            
            )
    clusters_res_tr = quantizer.fit_predict(
        probits_res_tr,
        k=n_clusters).squeeze(0)


    # if True:
    #     torch.save(quantizer.results, os.path.join(save_folder, f"results{suffix}.pt"))

    ################################################################################

    k = torch.tensor(n_clusters, device=device)
    clusters_res_val = quantizer.predict(probits_res_va).squeeze(0)
    clusters_cal = quantizer.predict(cal_probits).squeeze(0)
    clusters_test = quantizer.predict(test_probits).squeeze(0)
    clusters_dic = {"res_tr": clusters_res_tr, "res_val": clusters_res_val, "cal": clusters_cal, "test": clusters_test}
    errors_dic = {"res_tr": errors_res_tr, "res_val": errors_res_va, "cal": cal_errors, "test": test_errors}

    
    _, upper_res_tr = get_clusters_info(errors_res_tr, clusters_res_tr, k, bound=bound)
    _, upper_res_val = get_clusters_info(errors_res_va, clusters_res_val, k, bound=bound)

    _, upper_cal = get_clusters_info(cal_errors, clusters_cal, k, bound=bound)
    
    upper_dic = {"res_tr": upper_res_tr, "cal": upper_cal,  "res_val": upper_res_val}
    

    classif_results = {split: {upper_type: None for upper_type in upper_dic.keys()} for split in ['res_tr', 'res_val', 'cal', 'test']}

    for split in ["res_tr", "res_val", "cal", "test"]:
        for upper_type, upper in upper_dic.items():
            # print(f"Processing split: {split}, upper_type: {upper_type}")
            clusters = clusters_dic[split].to(device)
            errors = errors_dic[split]
            
            scores =  upper.gather(1, clusters)
        
           
            # print("score shape:", scores.shape)
            results = compute_all_metrics(
            conf=scores.cpu(),
            detector_labels=errors.cpu(),
        )
            # print("fpr shape:", np.shape(fpr))

            results = pd.DataFrame([results])
       
            
            classif_results[split][upper_type] = results

    print("MEan Test FPR:", np.mean(classif_results["test"]["cal"]["fpr"]))

    # Plot res_va vs test results
  
    for metric in ["fpr", "roc_auc", "aurc", "aupr_err", "aupr_success"]:


        plot_metric_corr(classif_results, upper_types=["res_tr", "cal"], splits=["res_val", "test"], metric=metric, save_folder=save_folder, suffix=suffix)
        
        plot_metric_corr(classif_results, upper_types=["res_tr", "res_tr"], splits=["cal", "test"], metric=metric, save_folder=save_folder, suffix=suffix)
        
        plot_metric_corr(classif_results, upper_types=["res_val", "cal"], splits=["res_tr", "test"], metric=metric, save_folder=save_folder, suffix=suffix)
        

    # print("quantizer.classif_results['res'][n_init],", quantizer.classif_results["res"][0])
    for n_init in [0]:
        quantizer.classif_results["res"][n_init] = pd.concat(quantizer.classif_results["res"][n_init], ignore_index=True)
        quantizer.classif_results["cal"][n_init] = pd.concat(quantizer.classif_results["cal"][n_init], ignore_index=True)
        quantizer.classif_results["test"][n_init] = pd.concat(quantizer.classif_results["test"][n_init], ignore_index=True)
        plot_eval(
            results_test=quantizer.classif_results["test"][n_init],
            results_cal=quantizer.classif_results["cal"][n_init],
            results_res=quantizer.classif_results["res"][n_init],
            metric="fpr",
            save_folder=save_folder,
            suffix= f"_init-{n_init}" + suffix,
        )

        # PLot Inertia
        plt.figure()
        for n_init in range(quantizer.num_init):
            plt.plot(quantizer.inertia_history[n_init], label=f"init {n_init}")
        plt.legend()    
        plt.xlabel("Iteration")
        plt.ylabel("Inertia")
        plt.title("KMeans Inertia over Iterations")
        plt.grid()
        plt.savefig(os.path.join(save_folder, f"inertia_history_{suffix}.png"))
        plt.close()
   



    # # clusters: 1D LongTensor of shape (N,), possibly on GPU
    # # Ensure we have a bin for every cluster id up to max()
    means = get_clusters_info(errors_res_tr, clusters_res_tr[0], torch.tensor([n_clusters], device=device), bound=bound)[0].squeeze(0)
    # print('clusters shape', clusters.shape)
    # print('means shape', means.shape)
    plot_cluster_sizes(clusters_res_tr[0], save_folder, suffix=suffix, error_means=means, sort=False)

    # cal_clusters = quantizer.predict(cal_probits).squeeze(0).to(torch.long)
    # print("cal clusters shape:", cal_clusters.shape)
    # means_cal = get_clusters_info(cal_errors, cal_clusters, torch.tensor([n_clusters], device=device), bound=bound)[0].squeeze(0)
    # plot_cluster_sizes(
    #     cal_clusters, save_folder, suffix=suffix, error_means=means_cal, split_name="cal",
    #     sort=False)

    # test_clusters = quantizer.predict(test_probits).squeeze(0).to(torch.long)
    # means_test = get_clusters_info(test_errors, test_clusters, torch.tensor([n_clusters], device=device), bound=bound)[0].squeeze(0)
    # plot_cluster_sizes(
    #     test_clusters, save_folder, suffix=suffix, error_means=means_test, split_name="test",
    #     sort=False)

    # for n_init in range(quantizer.num_init):
    #     quantizer.classif_results["res"][n_init] = pd.concat(quantizer.classif_results["res"][n_init], ignore_index=True)
    #     quantizer.classif_results["cal"][n_init] = pd.concat(quantizer.classif_results["cal"][n_init], ignore_index=True)
    #     quantizer.classif_results["test"][n_init] = pd.concat(quantizer.classif_results["test"][n_init], ignore_index=True)
    #     plot_eval(
    #         results_test=quantizer.classif_results["test"][n_init],
    #         results_cal=quantizer.classif_results["cal"][n_init],
    #         results_res=quantizer.classif_results["res"][n_init],
    #         metric="fpr",
    #         save_folder=save_folder,
    #         suffix= f"_init-{n_init}" + suffix,
    #     )

    #     # PLot Inertia
    #     plt.figure()
    #     for n_init in range(quantizer.num_init):
    #         plt.plot(quantizer.inertia_history[n_init], label=f"init {n_init}")
    #     plt.legend()    
    #     plt.xlabel("Iteration")
    #     plt.ylabel("Inertia")
    #     plt.title("KMeans Inertia over Iterations")
    #     plt.grid()
    #     plt.savefig(os.path.join(save_folder, f"inertia_history_{suffix}.png"))
    #     plt.close()

    # # Compute perason Correlation betweeen inertia and fpr
    # metric_value = [quantizer.classif_results["test"][n_init]["fpr"].values[-1] for n_init in range(quantizer.num_init)]
    # inertia_value = [quantizer.inertia_history[n_init, -1].item() for n_init in range(quantizer.num_init)]
    # pearson_corr = np.corrcoef(metric_value, inertia_value)[0, 1]
    # print(f"Pearson correlation between final inertia and FPR: {pearson_corr:.4f}")
    # print("Best FPR:", min(metric_value))

    #
    # # for supervised in [True, False]:
    # #     if supervised:
    # #        upper_bound = entropy(
    # #         p=torch.bincount(errors) / errors.size(0)
    # #         ).item()
    # #     else:
    # #         upper_bound = entropy(
    # #         p=torch.ones((probits.size(0), ), device=probits.device) / probits.size(0)
    # #         ).item()




    # plot_num_samples_per_cluster(
    #     trajectory_clusters=quantizer.results.list_clusters_res,
    #     save_folder=save_folder,
    #     suffix=suffix,
    #     n_cluster=n_clusters,
    #     start=0
    # )




    # plot_dead_resp(
    #         list_resp_dead=results.list_resp_dead,
    #         save_folder=save_folder,
    #         suffix=suffix
    #     )
    
        
    # plot_mutual_info(
    #     mutinfo_res_list=model.results.mutinfo_res_decomp_history,
    #     save_folder=save_folder,
    #     dataset="res",
    #     supervised=True,
    #     suffix= "decomp" + suffix ,
    # )

    # mapping_sup_clusters = reorder_by_error_threshold(
    #     detector_labels=errors,   # (n,) in {0,1} or [0,1]
    #     clusters=clusters,          # (n,) long in [0, k-1]
    #     n_cluster=n_clusters,                               # int or 0-dim tensor
    #     tau=0.00000000000001,               # gap tolerance on means
    # )
    # clusters_cal = model.predict(cal_probits).squeeze(0).to(torch.long)
    # for tau in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
    #     print("\nMerging with tau =", tau)
    
    #     means, upper_original, counts = get_clusters_info_2(cal_errors, clusters_cal, torch.tensor(n_clusters, device=device), bound=bound)
    #     mask = counts[0] > 0
    #     # print("mask shape:", mask.shape)
    #     # print('maslk:', mask)
    #     order_mean = torch.argsort(means.squeeze(0))  # ascending by mean error
        

    #     sorted_means = means.squeeze(0)[order_mean].tolist()
    #     sorted_upper = upper_original.squeeze(0)[order_mean].tolist()
    #     sorted_counts = counts.squeeze(0)[order_mean].tolist()
    #     # str_print = ["{mean:.6f} ({count}), U:{upper:.6f}".format(mean=m, count=c, upper=u) for m, c, u in zip(sorted_means, sorted_counts, sorted_upper)]
                    
    #     # print("Clusters (mean (count), U):", ", ".join(str_print))
    #     upper_original = upper_original.squeeze(0)  # shape (n_clusters,)
    #     order = torch.argsort(upper_original)  # ascending by upper bound
    #     mapping_sup_clusters = torch.full((n_clusters,), -1, device=device, dtype=torch.long)

    #     current_group = -1
    #     prev_val = None
    #     for cid in order.tolist():
    #         if not bool(mask[cid].item()):
    #             continue                     # leave empty clusters for the post-pass
    #         u = float(upper_original[cid])
    #         if (prev_val is None) or (u - prev_val > tau):
    #             current_group += 1
    #         mapping_sup_clusters[cid] = current_group
    #         prev_val = u
        
    #     if (~mask).any() and current_group >= 0:
    #     # find the non-empty cluster with the smallest upper bound → its group id
    #         highest_nonempty_cid = torch.argmax(upper_original.masked_fill(~mask, -float('inf')))
    #         lowest_gid = mapping_sup_clusters[highest_nonempty_cid].item()
    #         mapping_sup_clusters[~mask] = lowest_gid  # join empties to the safest group

    #     # --- determine merged k and guard against the "all empty" corner case ---
    #     if current_group < 0:
    #         # no non-empty clusters: fall back to a single group 0
    #         mapping_sup_clusters.fill_(0)
    #         k_merged = 1
    #     else:
    #         k_merged = int(mapping_sup_clusters.max().item()) + 1

    #     print("k (before):", n_clusters, "  k (after merge):", k_merged)

    #     # --- relabel calibration and recompute bounds with the merged labels ---
        
    #     new_clusters_cal = mapping_sup_clusters[clusters_cal]                  # fancy indexing
    #     _, upper = get_clusters_info(cal_errors, new_clusters_cal, torch.tensor(k_merged, device=device), bound=bound)
    #     upper = upper.squeeze(0)                                              # now shape (k_merged,)

    #     # --- relabel test and score with merged bounds ---
    #     clusters_test = model.predict(test_probits).squeeze(0).to(torch.long)
    #     # print("ok")
    #     # exit()
    #     new_clusters_test = mapping_sup_clusters[clusters_test]
    #     preds_test = upper.gather(0, new_clusters_test)    

    #     fpr_test, tpr_test, thr_test, auroc_test, accuracy_test, aurc_value_test, aupr_err_test, aupr_success_test = compute_all_metrics(
    #     conf=preds_test.cpu(),
    #     detector_labels=test_errors.cpu(),
    # )

    #     results_test = pd.DataFrame([{
    #     "fpr": fpr_test,
    #     "tpr": tpr_test,
    #     "thr": thr_test,
    #     "roc_auc": auroc_test,
    #     "model_acc": accuracy_test,
    #     "aurc": aurc_value_test,
    #     "aupr_err": aupr_err_test,
    #     "aupr_success": aupr_success_test,
    # }])
    #     print(results_test)


