#
import os
import pandas as pd
from typing import Any, Optional, Tuple, Union
from warnings import warn
from copy import deepcopy
from tqdm import tqdm

from code.utils.clustering.mutinfo_optimizer import MutInfoOptimizer
import torch
from torch import LongTensor, Tensor

from .distances import BaseDistance, CosineSimilarity
from .kmeans import KMeans
from .utils import SoftClusterResult, plot_cov_trajectory, plot_cluster_sizes, get_clusters_info, plot_eval, plot_entropy_and_cond_entropy_history, entropy, reorder_by_error_threshold, get_clusters_info_2, read_probits, plot_means_covs_trajectory_per_cluster, plot_num_samples_per_cluster, plot_weights_trajectory, plot_metric_corr
import matplotlib.pyplot as plt
from code.utils.metrics import compute_all_metrics

__all__ = ["SoftKMeans"]


class SoftKMeans(KMeans):
    """
    Implements differentiable soft k-means clustering.
    Method adapted from https://github.com/bwilder0/clusternet
    to support batches.

    Paper:
        Wilder et al., "End to End Learning and Optimization on Graphs" (NeurIPS'2019)

    Args:
        init_method: Method to initialize cluster centers: ['rnd', 'topk']
                        (default: 'rnd')
        num_init: Number of different initial starting configurations,
                    i.e. different sets of initial centers.
                    If >1 selects the best configuration before
                    propagating through fixpoint (default: 1).
        max_iter: Maximum number of iterations (default: 100).
        distance: batched distance evaluator (default: CosineSimilarity).
        p_norm: norm for lp distance (default: 1).
        normalize: id of method to use to normalize input. (default: 'unit').
        tol: Relative tolerance with regards to Frobenius norm of the difference
                    in the cluster centers of two consecutive iterations to
                    declare convergence. (default: 1e-4)
        n_clusters: Default number of clusters to use if not provided in call
                (optional, default: 8).
        verbose: Verbosity flag to print additional info (default: True).
        seed: Seed to fix random state for randomized center inits
                (default: True).
        temp: temperature for soft cluster assignments (default: 5.0).
        **kwargs: additional key word arguments for the distance function.

    """

    def __init__(
        self,
        temp: float = 5.0,
        reg_covar = 1e-06,
        init_scheme_covs: str = "statistics",
        init_kmeans_method: str = "k-means++",
        cov_momentum: float = 0.0,
        **kwargs,
    ):
        super(SoftKMeans, self).__init__(
            **kwargs,
        )
        self.temp = temp
        if self.temp <= 0.0:
            raise ValueError(f"temp should be > 0, but got {self.temp}.")
      
        self.reg_covar = reg_covar
        self.init_scheme_covs = init_scheme_covs
        self.cov_momentum = cov_momentum
        self.init_kmeans_method = init_kmeans_method



    def _cov_diag_init(self, x, k, **kwargs):
        """Choose k random nodes as initial centers.

        Args:
            x: (BS, N, D)
            k: (BS, )

        Returns:
            cov_diag: (BS, num_init, k_max, D)

        """

        n, d = x.size()
        bs, = k.shape
        k_max = torch.max(k).cpu().item()
        if self.init_scheme_covs == "statistics":
            # Initialize to data variance
            data_var = torch.var(x, dim=0, unbiased=True)  # (D,)
            cov_diag = data_var.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(bs, self.num_init, k_max, d).contiguous() 
        else:
            cov_diag = torch.ones((bs, self.num_init, k_max, d), device=x.device, dtype=x.dtype)

        return cov_diag

    def _weights_init(self, k: torch.LongTensor, device=None, dtype=None, **kwargs) -> torch.Tensor:
        """
        k: (bs,) number of clusters per batch item
        returns: weights (bs, num_init, k_max) with uniform weights over valid components
        """
        if device is None: device = k.device
        if dtype  is None: dtype  = torch.float32

        bs = k.shape[0]
        k_max = int(k.max())

        k_range = torch.arange(k_max, device=device).unsqueeze(0).expand(bs, -1)  # (bs, k_max)
        mask = (k_range < k.unsqueeze(1))                                         # True where component is valid

        # Row-wise divide by k: valid entries become 1/k[b], padded stay 0
        w = mask.to(dtype) / k.to(dtype).unsqueeze(1)                              # (bs, k_max)

        # replicate across num_init (same weights for each restart)
        w = w.unsqueeze(1).expand(bs, self.num_init, k_max).contiguous()           # (bs, num_init, k_max)
        return w

    @torch.no_grad()
    def _init_kmeans(self, x: Tensor, k: LongTensor, **kwargs) -> Tensor:
        """Wrapper to apply different methods for
        initialization of initial centers (centroids)."""
        km = KMeans(
            init_method=self.init_kmeans_method,
            num_init=self.num_init,
            max_iter=self.max_iter,
            # normalize=self.normalize,
            tol=self.tol,
            verbose=False,
            seed=self.seed,
            collect_info=False,
        )
        km.fit(x, k=k, **kwargs)
        # means = km.results.centers.unsqueeze(1).expand(-1, self.num_init, -1, -1).contiguous()
        means = km.results.centers.unsqueeze(0).contiguous()
        # print("means shape:", means.shape)
     
        return means

    
    @torch.no_grad()
    def _center_init(self, x: Tensor, k: LongTensor, **kwargs) -> Tensor:
        """Wrapper to apply different methods for
        initialization of initial centers (centroids)."""
        if self.init_method == "random":
            return self._init_rnd(x, k)
        elif self.init_method == "k-means++":
            return self._init_plus(x, k)
        elif self.init_method == "kmeans":
            return self._init_kmeans(x, k, **kwargs)
        else:
            raise ValueError(f"unknown initialization method: {self.init_method}.")

      
    def _estimate_log_gaussian_prob_diag(
        self,
        x: Tensor,                 # (bs, n, d)
        means: Tensor,             # (bs, m, k, d)
        cov_diag: Tensor,          # (bs, m, k, d)  (variances)
    ) -> Tensor:
        """
        Returns log N(x; mean_k, diag(cov_k)) for all (bs, m, n, k).
        Returns p_S_giv_Z : (bs, m, n, k)
        No big (n,k,d) intermediates are materialized.
        means : (bs, num_init, k, d)
        cov_diag : (bs, num_init, k, d)
        x : (n, d)

        """
        n, d = x.shape
        bs, m, k, d2 = means.shape
        assert d == d2

        #cov_diag = cov_diag.clamp(min=self.reg_covar)


        MU = means
   
        TAU = cov_diag.reciprocal()                              # (bs, m, k, d)
        MU_TAU = MU * TAU                                   # (bs, m, k, d)
        MU2_TAU_SUM = (MU.square() * TAU).sum(dim=3)        # (bs, m, k)
        X2_TAU   = torch.einsum("nd,bmkd->bmnk", x.square(), TAU)         # (b, m, n, k)
        X_MU_TAU = torch.einsum("nd,bmkd->bmnk", x, MU_TAU)               # (b, m, n, k)
        quad = X2_TAU - 2.0 * X_MU_TAU + MU2_TAU_SUM.unsqueeze(2)   # (bs, m, n, k)
        LOGDET = cov_diag.log().sum(dim=3)                       # (bs, m, k)

        # Final log-pdf per component
        log_prob = -0.5 * (d * torch.log(torch.tensor(2.0 * torch.pi, dtype=x.dtype, device=x.device))
                        + LOGDET.unsqueeze(2) + quad)    # (bs, m, n, k)

        return log_prob
    
    # def _estimate_log_weights(self, weights):
    #     return torch.log(weights)
    
    def _estimate_weighted_log_prob(self, x, weights, means, cov_diag):
       # weights = weights.clamp(min=self.eps)
        return self._estimate_log_gaussian_prob_diag(x, means, cov_diag) + torch.log(weights)[:, :, None, :]

    def _estimate_log_prob_resp(self, x, weights, means, cov_diag, invalid):

        weighted_log_prob = self._estimate_weighted_log_prob(x, weights, means, cov_diag)
        weighted_log_prob = weighted_log_prob.masked_fill(invalid[:, :, None, :], float('-inf'))
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=-1)

        log_resp = weighted_log_prob - log_prob_norm.unsqueeze(-1)
      
        return log_prob_norm, log_resp

    def _e_step(self, x, weights, means, cov_diag, invalid):

        log_prob_norm, log_resp = self._estimate_log_prob_resp(x, weights, means, cov_diag, invalid)
        return torch.mean(log_prob_norm, dim=-1), log_resp 

    def _m_step(self, X, log_resp, invalid, iter=None):

        weights, means, covariances, cluster_dead = self._estimate_gaussian_parameters(
            X,
            resp=log_resp.exp(),
            reg_covar=self.reg_covar,
            iter=iter,
        )
        weights = weights.masked_fill(invalid, 0)
        weights /= weights.sum(dim=2, keepdim=True)

        return weights, means, covariances, cluster_dead

    def _estimate_gaussian_parameters(self, x, resp, reg_covar, covariance_type="diag", iter=None):
        """
        
        Estimate the Gaussian distribution parameters.
        Parameters
        ----------
        x : array-like of shape (bs, m, n, d)

        resp : array-like of shape (b, m, n, k)


        """

        n, d = x.size()
        bs, num_init, n2, k = resp.size()
        assert n == n2

        nk = resp.sum(dim=2) + 10 * torch.finfo(resp.dtype).eps # (bs, m, k)
        #nk = resp.sum(dim=2).clamp(min=self.eps)  # (bs, m, k)
      
        cluster_dead = torch.nonzero((nk[0,0] < 1e-4)).squeeze().tolist()

                  
       #int(f"Dead clusters iter {iter}: {dead.sum().item()} / {bs * num_init * k}")
        means = torch.einsum('bmnk, nd -> bmkd', resp, x)  # (b, m, k, d)
        means = means * nk[:,:, :, None].reciprocal()  # (B, k, d)
        covariances = {
        # "full": _estimate_gaussian_covariances_full,
        # "tied": _estimate_gaussian_covariances_tied,
        "diag": self._estimate_gaussian_covariances_diag,
        # "spherical": _estimate_gaussian_covariances_spherical,
        }[covariance_type](resp, x, nk, means, reg_covar)
        return nk, means, covariances, cluster_dead


    def _estimate_gaussian_covariances_diag(self, resp, x, nk, means, reg_covar):
        """Estimate the diagonal covariance vectors.

        Parameters
        ----------
        responsibilities : array-like of shape (B, n_samples, n_components)

        X : array-like of shape (n, d)

        nk : array-like of shape (bs, m, k)

        resp : (bs, m, n, k)

        means : array-like of shape (bs, m, k, d)

        reg_covar : float

        Returns
        -------
        covariances : array, shape (n_components, d)
            The covariance vector of the current components.
        """
        avg_X2 = torch.einsum('bmnk, nd -> bmkd', resp, x.square()) / nk[:, :, :, None]
        # avg_X2 = resp.transpose(1, 2) @  torch.square(x_rep) / nk[:, :, None]
        avg_means2 = torch.square(means)
        # return (avg_X2 - avg_means2).clamp(min=reg_covar)
        return (avg_X2 - avg_means2) + reg_covar

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _hard_conditional_proba(self, cond_log_probs):
        """
        cond_log_probs : (n, k)
        returns: hard_cond_log_probs : (n, k)
        """
        hard_cond_log_probs = torch.zeros_like(cond_log_probs).scatter_(1, cond_log_probs.argmax(dim=1, keepdim=True), 1.0)
        hard_cond_log_probs = hard_cond_log_probs.clamp(
                min=torch.finfo(hard_cond_log_probs.dtype).tiny # For safe log(0)
                ).log()
        return hard_cond_log_probs

    # def _mutual_info_hard(self, clusters, errors, n_cluster):
    #     errors = errors.long()
    #     # print(errors[:3])
    #     p_EZ = torch.zeros((2, n_cluster), device=clusters.device)
    #     p_EZ[1, :] = torch.bincount(clusters[errors==1], minlength=n_cluster).to(torch.float32) / self.n
    #     p_EZ[0, :] = torch.bincount(clusters[errors==0], minlength=n_cluster).to(torch.float32) / self.n
    #     p_Z = p_EZ.sum(dim=0)
    #     p_E = p_EZ.sum(dim=1)
    #     assert torch.allclose(p_E, torch.bincount(errors, minlength=2).to(torch.float32) / self.n)
    #     mask = p_EZ > 0
    #     # mi = torch.sum(p_EZ * (torch.log(p_EZ + 1e-12) - torch.log(p_Z[None, :] + 1e-12) - torch.log(p_E[:, None] + 1e-12)))
    #     # mask = p_EZ > 0
    #     mi = torch.sum( p_EZ[mask] * ( torch.log(p_EZ[mask])
    #                                 - torch.log(p_Z).unsqueeze(0).expand_as(p_EZ)[mask]
    #                                 - torch.log(p_E).unsqueeze(1).expand_as(p_EZ)[mask]) )

    #     # p_Z_giv_S = torch.zeros_like(log_resp).scatter_(1, log_resp.argmax(dim=1, keepdim=True), 1.0)
    #     # p_Z_giv_S = p_Z_giv_S.clamp(
    #     #     min=torch.finfo(p_Z_giv_S.dtype).tiny # For safe log(0)
    #     # ).log()

    #     return mi

    def _mutual_info_hard(self,  probits=None, weights=None, means=None, sigmas=None, errors=None):
        errors = errors.long()
        n_cluster = means.shape[2]
        # print(errors[:3])
        _, p_Z_giv_S_log = self._e_step(probits, weights, means, sigmas, self.invalid) # (bs, m, n, K)
        clusters = p_Z_giv_S_log[0, 0, :, :].argmax(dim=1)  # (n,)
        p_EZ = torch.zeros((2, n_cluster), device=clusters.device)
        p_EZ[1, :] = torch.bincount(clusters[errors==1], minlength=n_cluster).to(torch.float32) / self.n
        p_EZ[0, :] = torch.bincount(clusters[errors==0], minlength=n_cluster).to(torch.float32) / self.n
        p_Z = p_EZ.sum(dim=0)
        p_E = p_EZ.sum(dim=1)
        assert torch.allclose(p_E, torch.bincount(errors, minlength=2).to(torch.float32) / self.n)
        mask = p_EZ > 0
        # mi = torch.sum(p_EZ * (torch.log(p_EZ + 1e-12) - torch.log(p_Z[None, :] + 1e-12) - torch.log(p_E[:, None] + 1e-12)))
        # mask = p_EZ > 0
        mi = torch.sum( p_EZ[mask] * ( torch.log(p_EZ[mask])
                                    - torch.log(p_Z).unsqueeze(0).expand_as(p_EZ)[mask]
                                    - torch.log(p_E).unsqueeze(1).expand_as(p_EZ)[mask]) )

        # p_Z_giv_S = torch.zeros_like(log_resp).scatter_(1, log_resp.argmax(dim=1, keepdim=True), 1.0)
        # p_Z_giv_S = p_Z_giv_S.clamp(
        #     min=torch.finfo(p_Z_giv_S.dtype).tiny # For safe log(0)
        # ).log()

        return mi

    def _mutual_info_decomposed(self, probits=None, weights=None, means=None, sigmas=None, errors=None, hard=False, alpha=1):
        p_E = torch.bincount(errors) / self.n
        n_clusters = means[0,0, :, :].shape[0]
        _, p_Z_giv_S_log = self._e_step(probits, weights, means, sigmas, self.invalid) # (bs, m, n, K)

        if hard:
            p_Z_giv_S_log = self._hard_conditional_proba(p_Z_giv_S_log[0, 0, :, :])  # (n, k)
    
        p_Z_giv_E_log = torch.zeros((2, n_clusters), device=means.device)
        p_Z_giv_E_log[1, :] = torch.logsumexp(p_Z_giv_S_log[errors == 1, :], dim=0) - torch.log(p_E[1] * self.n)
        p_Z_giv_E_log[0, :] = torch.logsumexp(p_Z_giv_S_log[errors == 0, :], dim=0) - torch.log(p_E[0] * self.n)

        p_Z_log = torch.logsumexp(p_Z_giv_E_log + torch.log(p_E.unsqueeze(-1)), dim=0)
        H_Z = - torch.sum(p_Z_log.exp() * p_Z_log)
        H_Z_giv_E = - torch.sum(
            p_E.unsqueeze(-1).expand(-1, n_clusters) *
            p_Z_giv_E_log.exp() *
            p_Z_giv_E_log
            )
        mi = H_Z - alpha * H_Z_giv_E
        return mi, H_Z, H_Z_giv_E


    
    def _mutual_info_unsup_model_decomp_S(self, probits=None, weights=None, means=None, sigmas=None,  alpha=1):
        """
        Compute I(S,Z) = H(S) - H(S|Z) where the model is fully specified and P_Z|S is soft assignment.
        """
        p_S_giv_Z_log = self._estimate_log_gaussian_prob_diag(probits, means, sigmas)[0, 0]  # (n,k)
        p_Z_log = weights[0, 0].log()                                                          # (k,)

        # log p_S(s_i) = logsumexp_k [ log p(s_i|k) + log pi_k ]
        p_S_log = torch.logsumexp(p_S_giv_Z_log + p_Z_log[None, :], dim=1)                     # (n,)

        # responsibilities r_{ik} = p(z=k | s_i)
        p_Z_giv_S_log = p_S_giv_Z_log + p_Z_log[None, :] - p_S_log[:, None]                            # (n,k)
        

        # H(S) = - E[log p_S(S)]
        H_S = - p_S_log.mean()

        # H(S|Z) = - E_S[ sum_k r_{ik} log p(s_i|k) ]
        H_S_giv_Z = - (p_Z_giv_S_log.exp() * p_S_giv_Z_log).sum(dim=1).mean()

        mi = H_S - alpha * H_S_giv_Z
        return mi, H_S, H_S_giv_Z
        
        # p_S_giv_Z_log = self._estimate_log_gaussian_prob_diag(probits, means, sigmas)[0, 0]  # (n, k)
        # p_Z_log = weights[0, 0, None, :].log()
        # p_S_log = torch.logsumexp(p_S_giv_Z_log + p_Z_log, dim=1)
  
        # H_S = - torch.sum(p_S_log.exp() * p_S_log)
        # H_S_giv_Z = - torch.sum(
        #     p_Z_log.exp() *
        #     p_S_giv_Z_log.exp() *
        #     p_S_giv_Z_log
        #     )
        # mi = H_S - alpha * H_S_giv_Z
        # return mi, H_S, H_S_giv_Z
    
    def _mutual_info_unsup_model_decomp_Z(self, probits=None, weights=None, means=None, sigmas=None,  alpha=1):
        """
        Compute I(S,Z) = H(S) - H(S|Z) where the model is fully specified and P_Z|S is soft assignment.
        """
        
        p_S_giv_Z_log = self._estimate_log_gaussian_prob_diag(probits, means, sigmas)[0, 0]  # (n, k)
        p_Z_log = weights[0, 0, None, :].log()                                                          # (k,)
        p_SZ_log = p_S_giv_Z_log + p_Z_log # (n, k)
        # p_S_log = torch.logsumexp(p_SZ_log, dim=1) # (n,)
        p_Z_giv_S_log = p_SZ_log - torch.logsumexp(p_SZ_log, dim=1, keepdim=True)  # (n, k)
         

        H_Z = - torch.sum(p_Z_log.exp() * p_Z_log)
        # H_Z_giv_S = - torch.sum(
        #     p_S_log[:, None].exp() *
        #     p_Z_giv_S_log.exp() *
        #     p_Z_giv_S_log
        #     )
        H_Z_giv_S = - torch.sum(
            p_Z_giv_S_log.exp() *
            p_Z_giv_S_log, dim=1
            ).mean()
        mi = H_Z - alpha * H_Z_giv_S
        return mi, H_Z, H_Z_giv_S

    
    
    # def _mutual_info_hard_unsup_model_decomp_S(self, probits=None, weights=None, means=None, sigmas=None, hard=True,  alpha=1):
    #     """
    #     Compute I(S,Z) = H(S) - H(S|Z) where the model is fully specified and P_Z|S is hard assignment.:
    #     1) Compute P_S|Z and P_S from the model
    #     2) Compute P_Z|S from P_S|Z, P_S and P_Z
    #     3) Transform P_Z|S to hard assignments
    #     4) Recompute P_S|Z from hard P_Z|S, P_S and P_Z
         
    #     """
        
    #     p_S_giv_Z_log = self._estimate_log_gaussian_prob_diag(probits, means, sigmas)[0, 0]  # (n, k)

    #     p_SZ_log = p_S_giv_Z_log + weights[0,0, None, :].log()  # (n, k)
    #     p_S_log = torch.logsumexp(p_SZ_log, dim=1) # (n,)
    #     p_Z_giv_S_log = p_SZ_log - torch.logsumexp(p_SZ_log, dim=1, keepdim=True)  # (n, k)
    #     p_Z_giv_S_log = self._hard_conditional_proba(p_Z_giv_S_log)  # (n, k)

    #     p_S_giv_Z_log = p_Z_giv_S_log + p_S_log[:, None] - weights[0, 0, None, :].log()  # (n, k)

    #     H_S = - torch.sum(p_S_log.exp() * p_S_log)
    #     H_S_giv_Z = - torch.sum(
    #         weights[0, 0, None, :] *
    #         p_S_giv_Z_log.exp() *
    #         p_S_giv_Z_log
    #         )
    #     mi = H_S - alpha * H_S_giv_Z
    #     return mi, H_S, H_S_giv_Z
    
    
    def _mutual_info_unsup_decomp_S(self, probits=None, weights=None, means=None, sigmas=None, hard=True,  alpha=1):
        """
        Compute I(S,Z) = H(S) - H(S|Z) where
            P_S is uniform over samples
            P_Z|S is hard or soft assignment from current model
        """
        
        p_S = torch.ones((self.n, ), device=means.device) / self.n
        p_S_giv_Z_log = self._estimate_log_gaussian_prob_diag(probits, means, sigmas)[0, 0]  # (n, k)

        p_SZ_log = p_S_giv_Z_log + weights[0,0, None, :].log()  # (n, k)
        p_Z_giv_S_log = p_SZ_log - torch.logsumexp(p_SZ_log, dim=1, keepdim=True)  # (n, k)

        if hard:
            p_Z_giv_S_log = self._hard_conditional_proba(p_Z_giv_S_log)  # (n, k)

        # clusters = p_Z_giv_S_log.argmax(dim=1)  # (n,)
        p_Z_new_log = torch.logsumexp(p_Z_giv_S_log + p_S[:, None].log(), dim=0)  # (k,)

        # new_weights = torch.bincount(clusters, minlength=weights.shape[2]).to(weights.dtype) / self.n
        p_S_giv_Z_log = p_Z_giv_S_log + p_S[:, None].log() - p_Z_new_log[None, :]  # (n, k)
        H_S = - torch.sum(p_S * p_S.log())
        H_S_giv_Z = - torch.sum(
            p_Z_new_log[None, :].exp() *
            p_S_giv_Z_log.exp() *
            p_S_giv_Z_log
            )
        mi = H_S - alpha * H_S_giv_Z
        return mi, H_S, H_S_giv_Z

    def _mutual_info_unsup_decomp_Z(self, probits=None, weights=None, means=None, sigmas=None, hard=False, alpha=1):
        p_S = torch.ones((self.n, ), device=means.device) / self.n
        p_S_giv_Z_log = self._estimate_log_gaussian_prob_diag(probits, means, sigmas)[0, 0]  # (n, k)

        p_SZ_log = p_S_giv_Z_log + weights[0,0, None, :].log()  # (n, k)
        p_Z_giv_S_log = p_SZ_log - torch.logsumexp(p_SZ_log, dim=1, keepdim=True)  # (n, k)

        if hard:
            p_Z_giv_S_log = self._hard_conditional_proba(p_Z_giv_S_log)  # (n, k)

        # clusters = p_Z_giv_S_log.argmax(dim=1)  # (n,)
        p_Z_new_log = torch.logsumexp(p_Z_giv_S_log + p_S[:, None].log(), dim=0)  # (k,)
        H_Z = - torch.sum(p_Z_new_log.exp() * p_Z_new_log)
        H_Z_giv_S = - torch.sum(
            p_S[:, None] *
            p_Z_giv_S_log.exp() *
            p_Z_giv_S_log
            )
        mi = H_Z - alpha * H_Z_giv_S
        return mi, H_Z, H_Z_giv_S
        
        # p_S = torch.ones((self.n, ), device=means.device) / self.n
        # n_clusters = means[0,0, :, :].shape[0]
        # _, p_Z_giv_S_log = self._e_step(probits, weights, means, sigmas, self.invalid) # (bs, m, n, K)

        # if hard:
        #     p_Z_giv_S_log = self._hard_conditional_proba(p_Z_giv_S_log[0, 0, :, :])  # (n, k)
    
        # p_Z_log = torch.logsumexp(p_Z_giv_S_log + torch.log(p_S.unsqueeze(-1)), dim=0)
        # H_Z = - torch.sum(p_Z_log.exp() * p_Z_log)
        # H_Z_giv_S = - torch.sum(
        #     # p_S.unsqueeze(-1).expand(-1, n_clusters) *
        #     p_S[:, None] *
        #     p_Z_giv_S_log.exp() *
        #     p_Z_giv_S_log
        #     )
        # mi = H_Z - alpha * H_Z_giv_S
        # return mi, H_Z, H_Z_giv_S

        


     
        

        # log_resp = weighted_log_prob - log_prob_norm.unsqueeze(-1)
        # p_S_log = torch.logsumexp(p_S_giv_Z_log + torch.log(weights[0, 0, None, :]), dim=1)
  
        # H_S = - torch.sum(p_S_log.exp() * p_S_log)
        # H_S_giv_Z = - torch.sum(
        #     weights[0, 0, None, :] *
        #     p_S_giv_Z_log.exp() *
        #     p_S_giv_Z_log
        #     )
        # mi = H_S - alpha * H_S_giv_Z
        # return mi, H_S, H_S_giv_Z
    def _init_info(self):
        """Record additional info before clustering."""

        self.best_init = None
        self.likelihood_history = torch.empty((self.num_init, self.max_iter), device='cpu')
        self.classif_results = {split: [[] for _ in range(self.num_init)] for split in ['res', 'cal', 'test']}
        self.classif_results_upper_res = {split: [[] for _ in range(self.num_init)] for split in ['res', 'val', 'test']}
        self.params_history = {"weights": []}
        self.list_clusters_res = []

    @torch.no_grad()
    def _record_info(self, x, weights, means, cov_diag, invalid=None, iter=0, k=None):
        """Record additional info after clustering."""
        
        likelihood, log_resp_res = self._e_step(x, weights, means, cov_diag, invalid)
        clusters_res = log_resp_res.argmax(dim=-1)
        _, log_resp_val = self._e_step(self.val_probits, weights, means, cov_diag, invalid)
        clusters_val = log_resp_val.argmax(dim=-1)
        _, log_resp_cal = self._e_step(self.cal_probits, weights, means, cov_diag, invalid)
        clusters_cal = log_resp_cal.argmax(dim=-1)
        _, log_resp_test = self._e_step(self.test_probits, weights, means, cov_diag, invalid)
        clusters_test = log_resp_test.argmax(dim=-1)

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
        self.params_history["weights"].append(weights.detach().cpu().clone())
        
        self.likelihood_history[:, iter] = likelihood.squeeze(0).cpu()

        

    
    @torch.no_grad()
    def _cluster(
        self, x: Tensor, weights, means: Tensor, cov_diag,  k: LongTensor, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor, Union[Tensor, Any]]:
        """
        Run Lloyd's k-means algorithm.

        Args:
            x: (BS, N, D)
            weights: (BS, num_init, k_max)
            means: (BS, num_init, k_max, D)
            cov_diag: (BS, num_init, k_max, D)
            k: (BS, )

        """
        n, d = x.size()

        bs, = k.shape
        k_max = int(k.max())
        k_max_range = torch.arange(k_max, device=x.device)[None, :].expand(bs, -1)
        self.invalid = (k_max_range >= k[:, None])[:, None, :].expand(bs, self.num_init, k_max)  # (bs,m,k)

        # Track per-init lower bound
        lower_bound = torch.full((bs, self.num_init), -float('inf'), device=x.device, dtype=x.dtype)
        self.n_iter = None
        # covs_history = []
        # weights_history = []
        params_history = {"weights": [], "means": [], "covariances": []}
        lower_bound_history = []
        lower_bound_test_history = []
        lower_bound_cal_history = []
        # mutinfo_model = MutInfoOptimizer()
        list_results_test = []
        list_results_cal = []
        list_results_res = []

        list_clusters_res = []
        list_resp_dead = []
        # mutinfo_res_list = []
        # H_Z_history = {"supervised": [], "unsupervised": []}
        # H_Z_cond_history = {"supervised": [], "unsupervised": []}
        # H_E_history = {"supervised": [], "unsupervised": []}
        # H_E_cond_history = {"supervised": [], "unsupervised": []}
        H_Z_history = {
            cond:  {
                mod:  {
                    "hard": [], "soft": []
                } for mod in ["model", "emp"]
            } for cond in ["E", "S"]
        }
        H_Z_cond_history = deepcopy(H_Z_history)
        mutinfo_res_decomp_history = {
            decomp:  {
                mod:  {
                    "supervised": {"hard": [], "soft": []},
                    "unsupervised": {"hard": [], "soft": []}
                    } for mod in ["model", "emp"]
                } for decomp in ["Z", "E", "S"]
            }
        H_E_history =  {
            mod:  {"hard": [], "soft": []} for mod in ["model", "emp"]
            }
        H_E_cond_history = deepcopy(H_E_history)
        H_S_history = deepcopy(H_E_history)
        H_S_cond_history = deepcopy(H_E_history)
        

        # mutinfo_res_decomp_history = {"supervised": [], "unsupervised": []}

        for i in tqdm(range(self.max_iter), desc="GMM EM clustering", leave=False):
            
       
            
            prev_lower_bound = lower_bound

            log_prob_norm, log_resp = self._e_step(x, weights, means, cov_diag, self.invalid) # (bs, num_init), (bs, num_init, n, k_max)

           # num_zero_resp = log_resp.exp()[0,0,:,:].eq(0).all(dim=1)

            weights, means, cov_diag, cluster_dead = self._m_step(x, log_resp, invalid=self.invalid, iter=i)
            #cov_diag = (1 - self.cov_momentum) * cov_diag + self.cov_momentum * new_cov_diag
            #cov_diag = (1- self.cov_momentum) * (cov_diag.sum(dim=-1, keepdim=True) / self.d) + self.cov_momentum * new_cov_diag
            #list_resp_dead.append(cluster_dead)
           # print(f"Iter {i}",  cov_diag[0,0].eq(self.reg_covar).all(dim=-1).sum())


            lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)
            #lower_bound_history.append(lower_bound[0].detach().cpu().item())
            if self.collect_info:
                self._record_info(x, weights, means, cov_diag, invalid=self.invalid, iter=i, k=k)
            

            change = lower_bound - prev_lower_bound
            if (torch.abs(change) < self.tol).all():
                converged = True
                self.n_iter = {i + 1}
                break
            if False:
                # Parameters history
                params_history["covariances"].append(cov_diag.detach().cpu().squeeze([0, 1]))
                params_history["weights"].append(weights.detach().cpu().squeeze([0, 1]))
                params_history["means"].append(means.detach().cpu().squeeze([0, 1]))

                log_prob_norm_cal, log_resp_cal = self._e_step(self.cal_probits, weights, means, cov_diag, self.invalid)
                clusters_cal = log_resp_cal.argmax(dim=-1).squeeze(0).squeeze(0)  # (n,)
                _, upper = get_clusters_info(self.cal_errors, clusters_cal, k, bound=self.bound)

                # Test results
                log_prob_norm_test, log_resp_test = self._e_step(self.test_probits, weights, means, cov_diag, self.invalid)
                clusters_test = log_resp_test.argmax(dim=-1).squeeze(0).squeeze(0)  # (n,)
                preds_test = upper.squeeze(0).gather(0, clusters_test)

                fpr_test, tpr_test, thr_test, auroc_test, accuracy_test, aurc_value_test, aupr_err_test, aupr_success_test = compute_all_metrics(
                conf=preds_test.cpu(),
                detector_labels=self.test_errors.cpu(),
            )
   
                results_test = pd.DataFrame([{
                "fpr": fpr_test,
                "tpr": tpr_test,
                "thr": thr_test,
                "roc_auc": auroc_test,
                "model_acc": accuracy_test,
                "aurc": aurc_value_test,
                "aupr_err": aupr_err_test,
                "aupr_success": aupr_success_test,
            }])
                # Cal results

                preds_cal = upper.squeeze(0).gather(0, clusters_cal)

                fpr_cal, tpr_cal, thr_cal, auroc_cal, accuracy_cal, aurc_value_cal, aupr_err_cal, aupr_success_cal = compute_all_metrics(
                conf=preds_cal.cpu(),
                detector_labels=self.cal_errors.cpu(),
            )
   
                results_cal = pd.DataFrame([{
                "fpr": fpr_cal,
                "tpr": tpr_cal,
                "thr": thr_cal,
                "roc_auc": auroc_cal,
                "model_acc": accuracy_cal,
                "aurc": aurc_value_cal,
                "aupr_err": aupr_err_cal,
                "aupr_success": aupr_success_cal,
            }])

                # Res results
                clusters_res = log_resp.argmax(dim=-1).squeeze(0).squeeze(0)  # (n,)
                preds_res = upper.squeeze(0).gather(0, clusters_res)

                fpr_res, tpr_res, thr_res, auroc_res, accuracy_res, aurc_value_res, aupr_err_res, aupr_success_res = compute_all_metrics(
                conf=preds_res.cpu(),
                detector_labels=self.res_errors.cpu(),
            )
   
                results_res = pd.DataFrame([{
                "fpr": fpr_res,
                "tpr": tpr_res,
                "thr": thr_res,
                "roc_auc": auroc_res,
                "model_acc": accuracy_res,
                "aurc": aurc_value_res,
                "aupr_err": aupr_err_res,
                "aupr_success": aupr_success_res,
            }])
                
                mi_unsup_soft_model_decomp_Z, h_Z_unsup_soft_model, h_Z_cond_unsup_soft_model = self._mutual_info_unsup_model_decomp_Z(
                    probits=x,
                    weights=weights, 
                    means=means, 
                    sigmas=cov_diag, 
                    alpha=1
                )

                mi_unsup_hard_emp_decomp_Z, h_Z_unsup_emp_model, h_Z_cond_unsup_emp_model = self._mutual_info_unsup_decomp_Z(
                    probits=x,
                    weights=weights, 
                    means=means, 
                    sigmas=cov_diag, 
                    hard=True, 
                    alpha=1
                )

                mi_unsup_soft_emp_decomp_Z, h_Z_unsup_soft_emp_model, h_Z_cond_unsup_soft_emp_model = self._mutual_info_unsup_decomp_Z(
                    probits=x,
                    weights=weights, 
                    means=means, 
                    sigmas=cov_diag, 
                    hard=False, 
                    alpha=1
                )

                mi_unsup_soft_model_decomp_S, h_S_unsup_soft_model, h_S_cond_unsup_soft_model = self._mutual_info_unsup_model_decomp_Z(
                    probits=x,
                    weights=weights, 
                    means=means, 
                    sigmas=cov_diag, 
                    alpha=1
                )

                mi_unsup_hard_emp_decomp_S, h_S_unsup_hard_emp_model, h_S_cond_unsup_hard_emp_model = self._mutual_info_unsup_decomp_S(
                    probits=x,
                    weights=weights, 
                    means=means, 
                    sigmas=cov_diag, 
                    hard=True, 
                    alpha=1
                )

                mi_unsup_soft_emp_decomp_S, h_S_unsup_soft_emp_model, h_S_cond_unsup_soft_emp_model = self._mutual_info_unsup_decomp_S(
                    probits=x,
                    weights=weights, 
                    means=means, 
                    sigmas=cov_diag, 
                    hard=False, 
                    alpha=1
                )

                mutinfo_res_decomp_history["Z"]["model"]["unsupervised"]["soft"].append(mi_unsup_soft_model_decomp_Z.detach().cpu().item())
                mutinfo_res_decomp_history["Z"]["emp"]["unsupervised"]["hard"].append(mi_unsup_hard_emp_decomp_Z.detach().cpu().item())
                mutinfo_res_decomp_history["Z"]["emp"]["unsupervised"]["soft"].append(mi_unsup_soft_emp_decomp_Z.detach().cpu().item())
                mutinfo_res_decomp_history["S"]["model"]["unsupervised"]["soft"].append(mi_unsup_soft_model_decomp_S.detach().cpu().item())
                mutinfo_res_decomp_history["S"]["emp"]["unsupervised"]["hard"].append(mi_unsup_hard_emp_decomp_S.detach().cpu().item())
                mutinfo_res_decomp_history["S"]["emp"]["unsupervised"]["soft"].append(mi_unsup_soft_emp_decomp_S.detach().cpu().item())

                H_Z_history["S"]["model"]["soft"].append(h_Z_unsup_soft_model.detach().cpu().item())
                H_Z_cond_history["S"]["model"]["soft"].append(h_Z_cond_unsup_soft_model.detach().cpu().item())
                H_Z_history["S"]["emp"]["hard"].append(h_Z_unsup_emp_model.detach().cpu().item())
                H_Z_cond_history["S"]["emp"]["hard"].append(h_Z_cond_unsup_emp_model.detach().cpu().item())
                H_Z_history["S"]["emp"]["soft"].append(h_Z_unsup_soft_emp_model.detach().cpu().item())
                H_Z_cond_history["S"]["emp"]["soft"].append(h_Z_cond_unsup_soft_emp_model.detach().cpu().item())

                H_S_history["model"]["soft"].append(h_S_unsup_soft_model.detach().cpu().item())
                H_S_cond_history["model"]["soft"].append(h_S_cond_unsup_soft_model.detach().cpu().item())
                H_S_history["emp"]["soft"].append(h_S_unsup_soft_emp_model.detach().cpu().item())
                H_S_cond_history["emp"]["soft"].append(h_S_cond_unsup_soft_emp_model.detach().cpu().item())
                H_S_history["emp"]["hard"].append(h_S_unsup_hard_emp_model.detach().cpu().item())
                H_S_cond_history["emp"]["hard"].append(h_S_cond_unsup_hard_emp_model.detach().cpu().item())

                list_clusters_res.append(clusters_res.detach().cpu())
                # mutinfo_res = self._mutual_info_hard(
                #     clusters_res, 
                #     self.res_errors, 
                #     n_cluster = k.max().item()
                #     )
                # mutinfo_res = self._mutual_info_hard(
                #     probits=x, 
                #     weights=weights, 
                #     means=means,
                #     sigmas=cov_diag,
                #     errors=self.res_errors
                #     )
              
                # mi_sup, h_z_sup, h_z_giv_e = self._mutual_info_decomposed(
                #     probits=x,
                #     errors=self.res_errors, 
                #     weights=weights, 
                #     means=means, 
                #     sigmas=cov_diag, 
                #     hard=True, 
                #     alpha=1
                # )
            
                # self._mutual_info_hard_unsup_model_decomp_S(
                #     probits=x,
                #     weights=weights, 
                #     means=means, 
                #     sigmas=cov_diag,
                # )

                # mi_unsup, h_z_unsup, h_z_giv_s = self._mutual_info_unsup_decomposed(
                #     probits=x,
                #     weights=weights, 
                #     means=means, 
                #     sigmas=cov_diag, 
                #     hard=True, 
                #     alpha=1
                # )

                # mi_unsup_2, h_s, h_s_giv_z = self._mutual_info_unsup_decomposed_E(
                #     probits=x,
                #     weights=weights, 
                #     means=means, 
                #     sigmas=cov_diag, 
                #     hard=True, 
                #     alpha=1
                # )

                
                list_results_test.append(results_test)
                list_results_cal.append(results_cal)
                list_results_res.append(results_res)
                lower_bound_test_history.append(log_prob_norm_test[0].detach().cpu().item())
                lower_bound_cal_history.append(log_prob_norm_cal[0].detach().cpu().item())

                # mutinfo_res_list.append(mutinfo_res.detach().cpu().item())
                # H_Z_history["supervised"].append(h_z_sup.detach().cpu().item())
                # H_Z_cond_history["supervised"].append(h_z_giv_e.detach().cpu().item())
                # mutinfo_res_decomp_history["supervised"].append(mi_sup.detach().cpu().item())
                # H_Z_history["unsupervised"].append(h_z_unsup.detach().cpu().item())
                # H_Z_cond_history["unsupervised"].append(h_z_giv_s.detach().cpu().item())
                # mutinfo_res_decomp_history["unsupervised"].append(mi_unsup.detach().cpu().item())

        # select best rnd restart according to inertia
        
        if self.n_iter is None:
            self.n_iter = {self.max_iter}
        # means[k_mask] = float("inf")
        # print("lower_bound", lower_bound.shape)
        log_prob_norm, log_resp = self._e_step(x, weights, means, cov_diag, self.invalid)
        # print("lorg resp shape :", log_resp.size())
        # print("shape log_prob_norm:", log_prob_norm.size())
        # inertia = self._calculate_inertia(x, means, c_assign)
        best_init = torch.argmax(lower_bound, dim=-1)
        b_idx = torch.arange(bs, device=x.device)
        # print("hello")
        if False:
            list_results_res = pd.concat(list_results_res, axis=0, )
            list_results_cal = pd.concat(list_results_cal, axis=0)
            list_results_test = pd.concat(list_results_test, axis=0)
        # print("log_resp[b_idx, best_init]", log_resp[b_idx, best_init])
        self.results = SoftClusterResult(
            log_resp=log_resp.squeeze(0).cpu(),
            weights=weights.squeeze(0).cpu(),
            means=means.squeeze(0).cpu(),
            cov_diags=cov_diag.squeeze(0).cpu(),
            lower_bound=log_prob_norm.squeeze(0).cpu(),
            # params_history=params_history,
            # lower_bound_history=lower_bound_history,
            k=k,
            # list_results_test=list_results_test,
            # list_results_cal=list_results_cal,
            # list_results_res=list_results_res,
            # lower_bound_test_history=lower_bound_test_history,
            # lower_bound_cal_history=lower_bound_cal_history,
            # mutinfo_res_list= [],
            # H_Z_history=H_Z_history,
            # H_Z_cond_history=H_Z_cond_history,
            # H_S_history=H_S_history,
            # H_S_cond_history=H_S_cond_history,
            # mutinfo_res_decomp_history=mutinfo_res_decomp_history,
            # list_clusters_res=list_clusters_res,
            # list_resp_dead=list_resp_dead,
        )


    @torch.no_grad()
    def predict(self, x: Tensor, **kwargs) -> LongTensor:
        """Hard assignments (argmax over responsibilities)."""
        assert self.is_fitted, "Call fit() first."
        x = self._check_x(x).to(self.device) # (bs, n, d)
        # if self.normalize is not None:
        #     x = self._normalize(x, self.normalize, self.eps)

        # Fitted params (one set per batch)
        means    = self.results.means.to(self.device)        # (bs, k_max, d)
        cov_diags = self.results.cov_diags.to(self.device)    # (bs, k_max, d)
        weights  = self.results.weights.to(self.device)      # (bs, k_max)
        k_vec    = self.results.k.to(self.device)            # (bs,)

        n, d = x.shape
        bs, k_max, d2 = means.shape
        assert  d == d2

        # Build invalid-component mask from k (don’t rely on any saved mask)
        k_range = torch.arange(k_max, device=self.device).expand(bs, -1)    # (bs, k_max)
        invalid = (k_range >= k_vec.unsqueeze(1)).unsqueeze(1)  

        _, log_resp = self._e_step(
            x, 
            weights.unsqueeze(1), 
            means.unsqueeze(1), 
            cov_diags.unsqueeze(1), 
            invalid
            )
        return log_resp.squeeze(1).argmax(dim=-1)  # type: ignore

    
    def fit_predict(
        self,
        x: Tensor,
        k: Optional[Union[LongTensor, Tensor, int]] = None,
        **kwargs,
    ) -> LongTensor:
        """Compute cluster centers and predict cluster index for each sample.

        Args:
            x: input features/coordinates (N, D)
            k: (bs,)
            centers: optional batch of initial centers to use (BS, K, D)
            **kwargs: additional kwargs for initialization or cluster procedure

        Returns:
            batch tensor of cluster labels for each sample (BS, N)

        """
        self(x, k=k,  **kwargs)
        return self.results.log_resp.argmax(dim=-1)  # type: ignore


    def forward(
        self,
        x: Tensor,
        weights: Optional[Tensor] = None,
        means: Optional[Tensor] = None,
        cov_diag: Optional[Tensor] = None,
        k: Optional[Union[LongTensor, Tensor, int]] = None,
        **kwargs,
    ) -> SoftClusterResult:
        """torch.nn like forward pass.

        Args:
            x: input features/coordinates (N, D)
            k: (bs,)
            centers: optional batch of initial centers to use (BS, K, D)
            **kwargs: additional kwargs for initialization or cluster procedure

        Returns:
            ClusterResult tuple

        """
        # print("x shape:", x.size())
        x = self._check_x(x)
        self.n, self.d = x.shape
        self.eps = torch.finfo(x.dtype).tiny
        # x_ = x
        k = self._check_k(k,  device=x.device)
        self.bs = k.shape[0]
      
        # # normalize input
        # if self.normalize is not None:
        #     x = self._normalize(x, self.normalize, self.eps)
        # init centers
        if means is None:
            means = self._center_init(x, k, **kwargs) # (bs, num_init, k_max, d)
        if cov_diag is None:
            cov_diag = self._cov_diag_init(x, k, **kwargs) # (bs, num_init, k_max, d)
        if weights is None:
            weights = self._weights_init(k, device=x.device, dtype=x.dtype) # (bs, num_init, k_max)
        means = self._check_centers(
            means, dims=(self.bs, self.n, self.d), dtype=x.dtype, device=x.device
        )
        self._cluster(x, weights, means, cov_diag, k, **kwargs)
      



# def read_probits(latent_path, order=True):
#     pkg = torch.load(latent_path, map_location="cpu")
#     all_logits = pkg["logits"].to(torch.float32)        # (N, C)
#     all_labels = pkg["labels"]              # (N,)
#     all_model_preds  = pkg["model_preds"]# (N,)
#     all_detector_labels = (all_model_preds != all_labels).int()
#     probits = torch.softmax(all_logits / 2, dim=1)
#     if order:
#         probits = probits.sort(dim=1, descending=True)[0]
  
#     return probits, all_detector_labels

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




# def plot_mutual_info(mutinfo_histories, save_folder, supervised =True, dataset= "res", upper_bound = None, suffix=""):

#     history = mutinfo_histories["supervised"] if supervised else mutinfo_histories["unsupervised"]
#     mi_label = "I(E,Z)" if supervised else "I(S,Z)"

#     plt.plot(range(len(history)), history, label=mi_label)
#     if upper_bound is not None:
#         label_upper = "H(E)" if supervised else "H(S)"
#         plt.hlines(upper_bound, 0, len(history)-1, colors='r', linestyles='dashed', label=label_upper)
#     plt.legend()
#     plt.xlabel("Iteration")
#     plt.ylabel("Mutual Information")
#     plt.title(f"{mi_label} on $\mathcal{{D}}_\mathrm{{{dataset}}}$")
#     plt.tight_layout()
#     save_file = f"mutual_info_supervised_history{suffix}.png" if supervised else f"mutual_info_unsupervised_history{suffix}.png"
#     save_path = os.path.join(save_folder, save_file)
#     plt.savefig(save_path, dpi=160)
#     plt.close()


def plot_mutual_info(histories, save_folder, supervised =True, dataset= "res", upper_bound = None, deconp="S",  suffix=""):

    mi_label = "I(E,Z)" if supervised else "I(S,Z)"
    supervised_str = "supervised" if supervised else "unsupervised"
    colors = {
        "model_hard": "blue",
        "model_soft": "cyan",
        "emp_hard": "red",
        "emp_soft": "orange",
    }

    styles = {
        "hard": "-",
        "soft": ":",
    }

    plt.figure(figsize=(6,4))
    for model_type in ["model", "emp"]:
    # model_type = "emp"
        for hard_soft in ["hard", "soft"]:
            history = histories[deconp][model_type][supervised_str][hard_soft]
            print("len history", len(history))
            label = f"{mi_label} ({model_type}, {hard_soft})"
            plt.plot(range(len(history)), history, label=label, linestyle=styles[hard_soft], color=colors[f"{model_type}_{hard_soft}"])

    # plt.plot(range(len(history)), history, label=mi_label)
    # if upper_bound is not None:
    #     label_upper = "H(E)" if supervised else "H(S)"
    #     plt.hlines(upper_bound, 0, len(history)-1, colors='r', linestyles='dashed', label=label_upper)
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Mutual Information")
    plt.title(f"{mi_label} on $\mathcal{{D}}_\mathrm{{{dataset}}}$")
    plt.tight_layout()
    
    save_file = f"mutual_info_{supervised_str}_deconp-{deconp}_history{suffix}.png"
    save_path = os.path.join(save_folder, save_file)
    plt.savefig(save_path, dpi=160)
    plt.close()
    
DATA_SPLITS = {
    "cifar10": (2000, 3000, 5000),
    "cifar100": (3000, 2000, 5000),
    "imagenet": (10000, 15000, 25000)
}

if __name__ == "__main__":

    gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    seed = 1
    cov_type = "diag"
    n_clusters = 80
    order = True
    init_scheme = "random"  # random, k-means++, kmeans
    init_scheme_covs = "statistics"  # statistic, uniform
    bound = "bernstein"
    max_iter = 200
    reg_covar = 1e-6
    cov_momentum = 0.
    space = "probits"  # "logits"  # "probits"


    

    model_name = "resnet34" #timm-vit-tiny16
    data_name = "cifar10"
    n_res = 2000
    n_cal = 3000
    n_test = 5000
    ratio_res = 0.5
    # n_samples = 2000
    seed_split = 9
    subclasses = None#  [5, 8] # [5, 4, 6, 8]  # [5, 8]
    num_init = 100
    temperature= 2
    metric = "fpr"
    suffix = ""

    # Real probits
    # latent_path = f"./latent/ablation/{data_name}_{model_name}_n_cal_with-res-{n_res}/seed-split-{seed_split}/res_n-samples-{n_res}_transform-test_n-epochs-1.pt"
    latent_path = f"./latent/{data_name}_{model_name}/seed-split-{seed_split}/res_n-samples-{n_res}_transform-test_n-epochs-1.pt"
    probits, errors = read_probits(latent_path, order=order, subclasses=subclasses,  temperature=temperature, space=space)
    latent_path_cal = f"./latent/{data_name}_{model_name}/seed-split-{seed_split}/cal_n-samples-{n_cal}_transform-test_n-epochs-1.pt"
    cal_probits, cal_errors = read_probits(latent_path_cal, order=order, subclasses=subclasses,  temperature=temperature, space=space)
    latent_path_test = f"./latent/{data_name}_{model_name}/seed-split-{seed_split}/test_n-samples-{n_test}.pt"
    test_probits, test_errors = read_probits(latent_path_test, order=order, subclasses=subclasses, temperature=temperature, space=space)
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

    import numpy as np
    n_tr = int(ratio_res * n_res)
    tr_idx = np.arange(n_tr)
    va_idx = np.arange(n_tr, n_res)
    probits_res_tr = probits[tr_idx]
    errors_res_tr = errors[tr_idx]
    probits_res_va = probits[va_idx]
    errors_res_va = errors[va_idx]
    # print(errors)
    # print(errors)

    # save_folder  = f"./code/utils/clustering/{data_name}_{model_name}_seed-split-{seed_split}_n-{n_samples}_iter-{max_iter}_lr-{lr}_lr_min-{lr_min}_K-{n_clusters}_with_logdet_fix_cov-{sig}/seed_{seed}/"
   # save_folder  = f"./code/utils/clustering/soft_kmeans_results/{data_name}_{model_name}_seed-split-{seed_split}_n-{n_res}_K-{n_clusters}_bound-{bound}_init-{init_scheme}/seed_{seed}/"
    if subclasses is not None:
        subclasses_str = "-".join([str(c) for c in subclasses])
        save_folder  = f"./code/utils/clustering/soft_kmeans_corr_results/subclass-{subclasses_str}"
    else:
        save_folder  = f"./code/utils/clustering/soft_kmeans_corr_results"
    file =f"{data_name}_{model_name}_n-res-{n_res}_n-cal-{n_cal}_seed-split-{seed_split}/iter-{max_iter}_K-{n_clusters}_bound-{bound}_init-{init_scheme}_initcovs-{init_scheme_covs}_ratiores-{ratio_res}"
    if reg_covar != 1e-6:
        file += f"_regcovar-{reg_covar}" 
    
    if num_init !=1:
        file += f"_numinit-{num_init}"
    if temperature !=2:
        file += f"_T-{temperature}"
    if cov_momentum != 0.0:
        file += f"_covmomentum-{cov_momentum}"
    if space != "probits":
        file += f"_{space}"
    save_folder  = os.path.join(save_folder, file, f"seed_{seed}")
    # save_folder  = f"./code/utils/clustering/toy_example_with_logdet_campled_min-{var_min}_cov/seed_{seed}/"
    os.makedirs(save_folder, exist_ok=True)
    print("save filer:", save_folder)

   

    model = SoftKMeans(
                seed=seed,
                init_method=init_scheme,
                init_scheme_covs=init_scheme_covs,
                max_iter = max_iter, 
                num_init=num_init, 
                verbose=0, 
        
                reg_covar=reg_covar,
                cal_errors=cal_errors,
                cal_probits=cal_probits,
                test_errors=test_errors,
                test_probits=test_probits,
                res_errors=errors_res_tr,
                val_probits=probits_res_va,
                val_errors=errors_res_va,
                bound=bound,
                cov_momentum=cov_momentum,
            
            )
    clusters_res_tr = model.fit_predict(probits_res_tr, k=n_clusters)
    #print("shape clusters res tr:", clusters_res_tr.shape)
    # print("Iterations:", model.n_iter)
    # print("FPR test:", model.results.list_results_test.iloc[-1]["fpr"])
    # print("Thr test:", model.results.list_results_test.iloc[-1]["thr"])

    ################################################################################

    k = torch.tensor(n_clusters, device=device)
    clusters_res_val = model.predict(probits_res_va).squeeze(0)
    clusters_cal = model.predict(cal_probits).squeeze(0)
    clusters_test = model.predict(test_probits).squeeze(0)
    clusters_dic = {"res_tr": clusters_res_tr, "res_val": clusters_res_val, "cal": clusters_cal, "test": clusters_test}
    errors_dic = {"res_tr": errors_res_tr, "res_val": errors_res_va, "cal": cal_errors, "test": test_errors}

    
    _, upper_res_tr = get_clusters_info(errors_res_tr, clusters_res_tr, k, bound=bound)
    _, upper_res_val = get_clusters_info(errors_res_va, clusters_res_val, k, bound=bound)

    _, upper_cal = get_clusters_info(cal_errors, clusters_cal.squeeze(0), k, bound=bound)
    
    upper_dic = {"res_tr": upper_res_tr, "cal": upper_cal,  "res_val": upper_res_val}
    

    classif_results = {split: {upper_type: None for upper_type in upper_dic.keys()} for split in ['res_tr', 'res_val', 'cal', 'test']}

    for split in ["res_tr", "res_val", "cal", "test"]:
        for upper_type, upper in upper_dic.items():
            # print(f"Processing split: {split}, upper_type: {upper_type}")
            clusters = clusters_dic[split].squeeze(0).to(device)
            errors = errors_dic[split]
          
            scores =  upper.to(device).gather(1, clusters)
        
           
            # print("score shape:", scores.shape)
            results = compute_all_metrics(
            conf=scores.cpu(),
            detector_labels=errors.cpu(),
        )
            # print("fpr shape:", np.shape(fpr))
            results = pd.DataFrame(results)
          
    
            
            classif_results[split][upper_type] = results
    

    
    best_init = np.argmin(classif_results["res_val"]["res_tr"][metric])
                   

                                         
    # Plot res_va vs test results
    print("MEan Test FPR:", np.mean(classif_results["test"]["cal"]["fpr"]))
    print("Min Test FPR:", np.min(classif_results["test"]["cal"]["fpr"]))
    print("Best init FPR test:", classif_results["test"]["cal"].iloc[best_init]["fpr"])
   
    # corr_likelihood_fpr_test = np.corrcoef(classif_results["test"]["cal"]["fpr"], model.likelihood_history[:,-1])
    # print("Correlation likelihood vs FPR test (cal upper):", corr_likelihood_fpr_test[0,1])
    # for metric in ["fpr", "roc_auc", "aurc", "aupr_in", "aupr_out"]:


    #     plot_metric_corr(classif_results, upper_types=["res_tr", "cal"], splits=["res_val", "test"], metric=metric, save_folder=save_folder, suffix=suffix)
        
    #     plot_metric_corr(classif_results, upper_types=["res_tr", "res_tr"], splits=["cal", "test"], metric=metric, save_folder=save_folder, suffix=suffix)
        
    #     plot_metric_corr(classif_results, upper_types=["res_val", "cal"], splits=["res_tr", "test"], metric=metric, save_folder=save_folder, suffix=suffix)
    #     plot_metric_corr(classif_results, upper_types=["res_tr", "cal"], splits=["res_tr", "test"], metric=metric, save_folder=save_folder, suffix=suffix)
        


    for n_init in [best_init]:
            model.classif_results["res"][n_init] = pd.concat(model.classif_results["res"][n_init], ignore_index=True)
            model.classif_results["cal"][n_init] = pd.concat(model.classif_results["cal"][n_init], ignore_index=True)
            model.classif_results["test"][n_init] = pd.concat(model.classif_results["test"][n_init], ignore_index=True)
            model.classif_results_upper_res["val"][n_init] = pd.concat(model.classif_results_upper_res["val"][n_init], ignore_index=True)
            model.classif_results_upper_res["res"][n_init] = pd.concat(model.classif_results_upper_res["res"][n_init], ignore_index=True)
            model.classif_results_upper_res["test"][n_init] = pd.concat(model.classif_results_upper_res["test"][n_init], ignore_index=True)
            
            plot_eval(
                results_test=model.classif_results["test"][n_init],
                results_cal=model.classif_results["cal"][n_init],
                results_res=model.classif_results["res"][n_init],
                results_upper_res={split: model.classif_results_upper_res[split][n_init] for split in model.classif_results_upper_res.keys()},
                
                metric="fpr",
                save_folder=save_folder,
                suffix= f"_init-{n_init}" + suffix,
            )
            plot_eval(
                results_test=model.classif_results["test"][n_init],
                results_cal=model.classif_results["cal"][n_init],
                results_res=model.classif_results["res"][n_init],
                results_upper_res={split: model.classif_results_upper_res[split][n_init] for split in model.classif_results_upper_res.keys()},
                
                metric="roc_auc",
                save_folder=save_folder,
                suffix= f"_init-{n_init}" + suffix,
            )
        

    # # if True:
    # #     torch.save(model.results, os.path.join(save_folder, f"results{suffix}.pt"))



    # plot_cov_trajectory(
    #     sigmas_list=model.results.params_history["covariances"],
    #     save_folder=save_folder,
    #     suffix=suffix,
    #     ylims=(0, 0.1),
    #     start=2
    # )
    # # clusters: 1D LongTensor of shape (N,), possibly on GPU
    # # Ensure we have a bin for every cluster id up to max()
    means = get_clusters_info(errors_res_tr, clusters_res_tr[best_init], torch.tensor([n_clusters], device=device), bound=bound)[0].squeeze(0)
    # print('clusters shape', clusters.shape)
    # print('means shape', means.shape)
    plot_cluster_sizes(clusters_res_tr[best_init], save_folder, suffix=suffix, error_means=means, sort=False)

    # cal_clusters = model.predict(cal_probits).squeeze(0).to(torch.long)
    # means_cal = get_clusters_info(cal_errors, cal_clusters, torch.tensor([n_clusters], device=device), bound=bound)[0].squeeze(0)
    # plot_cluster_sizes(
    #     cal_clusters, save_folder, suffix=suffix, error_means=means_cal, split_name="cal",
    #     sort=False)

    # test_clusters = model.predict(test_probits).squeeze(0).to(torch.long)
    # means_test = get_clusters_info(test_errors, test_clusters, torch.tensor([n_clusters], device=device), bound=bound)[0].squeeze(0)
    # plot_cluster_sizes(
    #     test_clusters, save_folder, suffix=suffix, error_means=means_test, split_name="test",
    #     sort=False)

    # plot_lower_bound(
    #     lower_bound_history=model.results.lower_bound_history,
    #     lower_bound_cal_history=model.results.lower_bound_cal_history,
    #     lower_bound_test_history=model.results.lower_bound_test_history,

    #     save_folder=save_folder,
    #     suffix=suffix,
    # )
    # plot_eval(
    #     results_test=model.results.list_results_test,
    #     results_cal=model.results.list_results_cal,
    #     results_res=model.results.list_results_res,
    #     metric="fpr",
    #     save_folder=save_folder,
    #     suffix=suffix,
    # )

    # # for supervised in [True, False]:
    # #     if supervised:
    # #        upper_bound = entropy(
    # #         p=torch.bincount(errors) / errors.size(0)
    # #         ).item()
    # #     else:
    # #         upper_bound = entropy(
    # #         p=torch.ones((probits.size(0), ), device=probits.device) / probits.size(0)
    # #         ).item()
    # plot_mutual_info(
    #     histories=model.results.mutinfo_res_decomp_history,
    #     save_folder=save_folder,
    #     dataset="res",
    #     supervised=False,
    #     deconp="S",
    #     # upper_bound=upper_bound,
    #     suffix=suffix,
    # )

    # plot_mutual_info(
    #     histories=model.results.mutinfo_res_decomp_history,
    #     save_folder=save_folder,
    #     dataset="res",
    #     supervised=False,
    #     deconp="Z",
    #     # upper_bound=upper_bound,
    #     suffix=suffix,
    # )

    # plot_entropy_and_cond_entropy_history(
    #     histories=model.results.H_S_history,
    #     cond_entropy_histories=model.results.H_S_cond_history,
    #     save_folder=save_folder,
    #     suffix=suffix , 
    #     # supervised=supervised,
    #     variable="S",
    #     cond="Z",
    #     split="res"
    #     )
    # plot_entropy_and_cond_entropy_history(
    #     histories=model.results.H_Z_history,
    #     cond_entropy_histories=model.results.H_Z_cond_history,
    #     save_folder=save_folder,
    #     suffix=suffix , 
    #     # supervised=supervised,
    #     variable="Z",
    #     cond="S",
    #     split="res",
    #     n_clusters=n_clusters
    #     )
    
    
    # plot_means_covs_trajectory_per_cluster(
    #     trajectory=torch.stack(model.results.params_history["covariances"]).reciprocal(),
    #     save_folder=save_folder,
    #     suffix=suffix,
    #     start=2
    # )

    # plot_num_samples_per_cluster(
    #     trajectory_clusters=model.results.list_clusters_res,
    #     save_folder=save_folder,
    #     suffix=suffix,
    #     n_cluster=n_clusters,
    #     start=0
    # )





    # plot_weights_trajectory(
    #     trajectory=model.results.params_history["weights"],
    #     save_folder=save_folder,
    #     suffix=suffix,
    #     start=0
    # )

    # # plot_dead_resp(
    # #         list_resp_dead=results.list_resp_dead,
    # #         save_folder=save_folder,
    # #         suffix=suffix
    # #     )
    
        
    # # plot_mutual_info(
    # #     mutinfo_res_list=model.results.mutinfo_res_decomp_history,
    # #     save_folder=save_folder,
    # #     dataset="res",
    # #     supervised=True,
    # #     suffix= "decomp" + suffix ,
    # # )

    # # mapping_sup_clusters = reorder_by_error_threshold(
    # #     detector_labels=errors,   # (n,) in {0,1} or [0,1]
    # #     clusters=clusters,          # (n,) long in [0, k-1]
    # #     n_cluster=n_clusters,                               # int or 0-dim tensor
    # #     tau=0.00000000000001,               # gap tolerance on means
    # # )
    # # clusters_cal = model.predict(cal_probits).squeeze(0).to(torch.long)
    # # for tau in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
    # #     print("\nMerging with tau =", tau)
    
    # #     means, upper_original, counts = get_clusters_info_2(cal_errors, clusters_cal, torch.tensor(n_clusters, device=device), bound=bound)
    # #     mask = counts[0] > 0
    # #     # print("mask shape:", mask.shape)
    # #     # print('maslk:', mask)
    # #     order_mean = torch.argsort(means.squeeze(0))  # ascending by mean error
        

    # #     sorted_means = means.squeeze(0)[order_mean].tolist()
    # #     sorted_upper = upper_original.squeeze(0)[order_mean].tolist()
    # #     sorted_counts = counts.squeeze(0)[order_mean].tolist()
    # #     # str_print = ["{mean:.6f} ({count}), U:{upper:.6f}".format(mean=m, count=c, upper=u) for m, c, u in zip(sorted_means, sorted_counts, sorted_upper)]
                    
    # #     # print("Clusters (mean (count), U):", ", ".join(str_print))
    # #     upper_original = upper_original.squeeze(0)  # shape (n_clusters,)
    # #     order = torch.argsort(upper_original)  # ascending by upper bound
    # #     mapping_sup_clusters = torch.full((n_clusters,), -1, device=device, dtype=torch.long)

    # #     current_group = -1
    # #     prev_val = None
    # #     for cid in order.tolist():
    # #         if not bool(mask[cid].item()):
    # #             continue                     # leave empty clusters for the post-pass
    # #         u = float(upper_original[cid])
    # #         if (prev_val is None) or (u - prev_val > tau):
    # #             current_group += 1
    # #         mapping_sup_clusters[cid] = current_group
    # #         prev_val = u
        
    # #     if (~mask).any() and current_group >= 0:
    # #     # find the non-empty cluster with the smallest upper bound → its group id
    # #         highest_nonempty_cid = torch.argmax(upper_original.masked_fill(~mask, -float('inf')))
    # #         lowest_gid = mapping_sup_clusters[highest_nonempty_cid].item()
    # #         mapping_sup_clusters[~mask] = lowest_gid  # join empties to the safest group

    # #     # --- determine merged k and guard against the "all empty" corner case ---
    # #     if current_group < 0:
    # #         # no non-empty clusters: fall back to a single group 0
    # #         mapping_sup_clusters.fill_(0)
    # #         k_merged = 1
    # #     else:
    # #         k_merged = int(mapping_sup_clusters.max().item()) + 1

    # #     print("k (before):", n_clusters, "  k (after merge):", k_merged)

    # #     # --- relabel calibration and recompute bounds with the merged labels ---
        
    # #     new_clusters_cal = mapping_sup_clusters[clusters_cal]                  # fancy indexing
    # #     _, upper = get_clusters_info(cal_errors, new_clusters_cal, torch.tensor(k_merged, device=device), bound=bound)
    # #     upper = upper.squeeze(0)                                              # now shape (k_merged,)

    # #     # --- relabel test and score with merged bounds ---
    # #     clusters_test = model.predict(test_probits).squeeze(0).to(torch.long)
    # #     # print("ok")
    # #     # exit()
    # #     new_clusters_test = mapping_sup_clusters[clusters_test]
    # #     preds_test = upper.gather(0, new_clusters_test)    

    # #     fpr_test, tpr_test, thr_test, auroc_test, accuracy_test, aurc_value_test, aupr_err_test, aupr_success_test = compute_all_metrics(
    # #     conf=preds_test.cpu(),
    # #     detector_labels=test_errors.cpu(),
    # # )

    # #     results_test = pd.DataFrame([{
    # #     "fpr": fpr_test,
    # #     "tpr": tpr_test,
    # #     "thr": thr_test,
    # #     "roc_auc": auroc_test,
    # #     "model_acc": accuracy_test,
    # #     "aurc": aurc_value_test,
    # #     "aupr_err": aupr_err_test,
    # #     "aupr_success": aupr_success_test,
    # # }])
    # #     print(results_test)


