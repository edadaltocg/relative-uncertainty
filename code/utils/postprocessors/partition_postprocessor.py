
import os
import numpy as np
import joblib
import torch
from .base_postprocessor import BasePostprocessor
from code.utils.clustering.kmeans import KMeans as TorchKMeans
from code.utils.clustering.my_soft_kmeans import SoftKMeans as TorchSoftKMeans
from code.utils.clustering import TreeQuantizer
from code.utils.clustering.mutinfo_optimizer import MutInfoOptimizer
from code.utils.clustering.divergences import (
    euclidean,
    kullback_leibler,
    itakura_saito,
    alpha_divergence_factory,
)

# from sklearn.cluster import KMeans, MiniBatchKMeans
# from sklearn.mixture import GaussianMixture

def gini(logits, temperature=1.0, normalize=False):
    g =torch.sum(torch.softmax(logits / temperature, dim=1) ** 2, dim=1, keepdim=True)
    if normalize:
        return  (1 - g) / g 
    else:
        return 1 - g

class PartitionPostprocessor(BasePostprocessor):
    def __init__(self, model, cfg, result_folder, class_subset=None, device=torch.device('cpu')):
        """
        Args:
            classifier (nn.Module): A PyTorch model that takes an input tensor of shape [1, dim] and returns (logits, probs).
            weights (torch.Tensor): Tensor of shape [n_classes] (e.g., [7]).
            means (torch.Tensor): Tensor of shape [n_classes, dim] (e.g., [7, 10]).
            stds (torch.Tensor): Tensor of shape [n_classes, dim] (e.g., [7, 10]).
            n_cluster (int): Number of clusters to partition the error probability into.
            alpha (float): Confidence level parameter for interval widths.
            method (str): The method to compute the cluster. (Currently only "uniform" is supported.)
            seed (int): Random seed for data generation.
            device (torch.device): Device on which to run the classifier.
        """
        super().__init__(model, cfg, result_folder, device)

        ## Meta Parameters
        self.alpha = cfg["alpha"]
        self.method = cfg["method"]
        
        self.bound = cfg["bound"]
        self.n_classes = cfg["n_classes"]
        self.class_subset = torch.tensor(class_subset, dtype=torch.long) if class_subset is not None else None

        ## Quantizer Space
        self.quantiz_space = cfg["space"]
        self.reorder_embs = cfg["reorder_embs"]
        self.n_dim = cfg.get("n_dim", None)
        self.temperature = cfg["temperature"]
        self.normalize_gini = False
        self.pred_weight = cfg["pred_weights"]
        self.reducer = None  # No reducer for now

        ## Quantize Parameters
        self.n_clusters = cfg["n_clusters"]
        self.cov_type = cfg.get("cov_type", None)
        self.init_scheme = cfg.get("init_scheme", None)
        self.n_init = cfg.get("n_init", None)
        self.max_iter = cfg.get("max_iter", None)
        self.quantiz_seed = cfg["clustering_seed"]
        self.sig = cfg.get("sig", None)
        self.lr = cfg.get("lr", None)
        self.with_logdet = cfg.get("with_logdet", None)
        self.cov_proj_type = cfg.get("cov_proj_type", None)
        self.alpha = cfg.get("alpha", None)
        self.mutual_computation = cfg.get("mutual_computation", None)
        self.score = cfg.get("score", "upper")

        ### Bregman Divergence Parameters
        self.divergence = None

        

        self.__init__quantizer()


    def __init__quantizer(self):

        if self.method == "kmeans_torch":
            
            self.clustering_algo = TorchKMeans(
                # n_clusters=self.list_n_cluster[0], 
                n_clusters=self.n_clusters,
                seed=self.quantiz_seed, 
                init_method=self.init_scheme,
                num_init=self.n_init, 
                verbose=0, 
                max_iter = self.max_iter,
                collect_info=False,
                # reg_covar=1e-3
            )
        elif self.method == "soft-kmeans_torch":
            
            self.clustering_algo = TorchSoftKMeans(
                seed=self.quantiz_seed, 
                init_method=self.init_scheme,
                max_iter = self.max_iter, 
                num_init=self.n_init, 
                verbose=0, 
                save_folder=self.result_folder,
                collect_info=False,
                # reg_covar=1e-3
            )

        elif self.method == "mutinfo_opt":
            self.clustering_algo = MutInfoOptimizer(
                n_clusters=self.n_clusters,
                cov_type=self.cov_type,
                sigmas=self.sig * torch.ones((self.n_clusters, self.n_classes), device=self.device),
                max_iter=self.max_iter,
                lr=self.lr,
                init_scheme=self.init_scheme,
                n_init=self.n_init,
                seed=self.quantiz_seed,
                device=self.device,
                with_logdet=self.with_logdet,
                cov_proj_type=self.cov_proj_type,
                sig_init = self.sig,
                alpha=self.alpha,
                mutual_computation=self.mutual_computation,
            )
        # elif self.method == "tim_gd":
        elif self.method == "decision-tree":
            self.clustering_algo = TreeQuantizer(
                seed=self.quantiz_seed,
                verbose=0, 
                n_clusters=self.n_clusters,
                splitter="best",
                max_depth=None,
                class_weight="balanced",
                device=self.device,
            
            )

        elif self.method in ["unif-width", "unif-mass"]:
            pass
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        pass

    def _extract_embeddings(self, x=None, logits=None):
        """
        Extract embeddings from the model.
        This function is used to create a feature extractor.
        """

        # Should implement the reducer here!
        # if self.reducer is not None:
        #     all_embs = torch.tensor(self.reducer.fit_transform(all_embs.cpu().numpy()), device=self.device)

       
        if logits is not None:
            logits = logits.to(self.device)
            if self.class_subset is not None:
                logits = logits[:, self.class_subset]
            if self.quantiz_space == "gini":
                embs = gini(logits, temperature=self.temperature, normalize=self.normalize_gini)
                return embs
            elif self.quantiz_space == "probits":
                embs = torch.softmax(logits / self.temperature, dim=1)
            elif self.quantiz_space == "logits":
                embs = logits
            elif self.quantiz_space == "max_proba":
                probs = torch.softmax(logits / self.temperature, dim=1)
                embs, _ = torch.max(probs, dim=1)

                return embs
        else:
            self.model.to(self.device)
            logits = self.model(x)
            if self.class_subset is not None:
                logits = logits[:, self.class_subset]
            self.model.to(torch.device('cpu'))
            if self.quantiz_space == "gini":
                embs = gini(logits, temperature=self.temperature, normalize=self.normalize_gini)
            elif self.quantiz_space == "probits":
                embs = torch.softmax(logits / self.temperature, dim=1)
            elif self.quantiz_space == "logits":
                embs = logits
            else:
                raise ValueError("Unsupported quantiz_space")


        # Reorder embeddings if needed
        if self.reorder_embs:
            embs, idx = embs.sort(dim=1, descending=True)  # idx: shape (B, N)
            self._perm_idx = idx
            # print("idx" ,idx.shape)                                   # save per-batch permutation
       
            # embs: (N, D)
            # scores = embs.norm(dim=1)                 # example key, shape (N,)
            # idx = torch.argsort(scores, descending=True)  # permutation σ on [0..N-1]

            # embs = embs[idx]                   # apply σ to rows
            # self._perm_idx = idx                        # save σ for later

        if self.n_dim is not None:
            embs = embs[:, :self.n_dim]

        if self.pred_weight is not None:
            if self.pred_weight > 0:
                preds = torch.argmax(logits, dim=1)
                preds_onehot = torch.nn.functional.one_hot(preds, num_classes=self.n_classes).float()
                scale = embs.detach().abs().amax()
                W = float(self.pred_weight) * (float(scale) + 1e-8)
                embs = torch.cat([embs, W * preds_onehot.to(embs.device)], dim=1)
        return embs

    
    def predict_clusters(self, x=None, logits=None):

        embs = self._extract_embeddings(x, logits)

        if self.method == "unif-width":
            cluster = torch.floor(embs * self.n_clusters).long()
            cluster[cluster == self.n_clusters] = self.n_clusters - 1 # Handle edge case when proba_error == 1
            return cluster
        elif self.method == "unif-mass":

            bin_edges = self.bin_edges.to(embs.device)
            # bucketize returns integers in [0, n_clusters-1]
            cluster = torch.bucketize(embs, bin_edges)
            return cluster  # (N,)

        
        else:
            if self.reducer is not None:
                embs = self.reducer.transform(embs.cpu().numpy())
                cluster = torch.tensor(self.clustering_algo.predict(embs), 
                                    device=self.device)
            else:
                if self.method in ["kmeans_torch", "soft-kmeans_torch", "mutinfo_opt"]:
                    embs = embs.to(self.device)
                    cluster = self.clustering_algo.predict(embs).cpu()
                    if self.clustering_algo.best_init is not None:
                        cluster = cluster[self.clustering_algo.best_init, :].unsqueeze(0)
                    # print("cluster predict shape", cluster.shape)
                else:
                    cluster = torch.tensor(self.clustering_algo.predict(embs.cpu().numpy()), 
                                   device=self.device)
            return cluster

    def save_results(self, experiment_folder, clusters=None, embs=None):

            np.savez_compressed(
                os.path.join(experiment_folder, "cluster_results.npz"),
                    cluster_counts=self.cluster_counts,
                    cluster_error_means=self.cluster_error_means,
                    cluster_error_vars=self.cluster_error_vars,
                    cluster_intervals=self.cluster_intervals,
                    clusters = clusters,
                    embs = embs,
                )
            
            joblib.dump(self.clustering_algo, os.path.join(experiment_folder, 'clustering_algo.pkl'))
        # else:
        #     raise ValueError("Unsupported method")
    def fit_quantizer(self, all_embs,  detector_labels, logits=None):
        if self.method in ["kmeans_torch", "soft-kmeans_torch"]:
            all_embs = all_embs.to(self.device)
            #print("All embs shape", all_embs.shape)
            clusters = self.clustering_algo.fit_predict(all_embs, self.n_clusters).squeeze(0)
            self.n_iter = self.clustering_algo.n_iter

        elif self.method == "mutinfo_opt":
            clusters = self.clustering_algo.fit_predict(
                probits=all_embs, 
                errors=detector_labels
                )  
        
            self.n_iter = self.clustering_algo.results.best_iter
        elif self.method == "decision-tree":
            clusters = self.clustering_algo.fit_predict(
                x=all_embs, 
                y=detector_labels
                )  
        elif self.method == "unif-width":
            clusters = self.predict_clusters(logits=logits)
        elif self.method == "unif-mass":
             # internal quantiles: (n_clusters - 1) edges
            q = torch.linspace(0.0, 1.0, self.n_clusters + 1, device=all_embs.device)[1:-1]
            bin_edges = torch.quantile(all_embs, q)
            self.bin_edges = bin_edges.detach().cpu()  # store on CPU
            clusters = torch.bucketize(all_embs, self.bin_edges.to(all_embs.device))
            clusters = clusters.to(self.device, dtype=torch.long)
        else:
            raise ValueError("Unsupported method")
        return clusters
        

    def fit(self, logits, detector_labels, dataloader=None, fit_clustering=True):

        all_embs = self._extract_embeddings(logits=logits)
    
        if fit_clustering:
            clusters = self.fit_quantizer(all_embs=all_embs, detector_labels=detector_labels, logits=logits)
        else:
            clusters = self.predict_clusters(logits=logits)
            
        self.clustering(detector_labels, clusters, k=torch.tensor([self.n_clusters]))

        torch.save(
            {
                "cluster_counts": self.cluster_counts,
                "cluster_error_means": self.cluster_error_means,
                "cluster_error_vars": self.cluster_error_vars,
                "cluster_intervals": self.cluster_intervals,
            },
            os.path.join(self.result_folder, f"partition_cluster_stats_n-clusters-{self.n_clusters}.pt"),
        )


    #    # statistics = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # cifar10 resnet34
    #     statistics = ((0.5000, 0.5000, 0.5000), (0.5000, 0.5000, 0.5000)) # imagenet base
    #     mean, std = statistics
    #     out_dir = f"./cluster_examples_imagenet_7_fit_cluster_{fit_clustering}"
        # N = self._perm_idx.numel()
        # inv = torch.empty_like(self._perm_idx)
        # inv[self._perm_idx] = torch.arange(N, device=self._perm_idx.device)  # build σ^{-1}

        # embs_recovered = all_embs[inv]        # back to original row order
        # embs_recovered = all_embs.gather(1, self._perm_idx.argsort(1))

        # save_cluster_examples(
        #     dataloader=dataloader,   # (N,3,32,32), already normalized by your dataset
        #     embeddings=all_embs,                      # (N,D)
        #     clusters=clusters,                 # (N,)
        #     out_dir=out_dir,
        #     per_cluster=10,
        #     pick="first",
        #     mean=mean, std=std                    # <-- ensures correct de-normalization for PNGs
        # )
        # self.experiment_folder = out_dir
      
       # torch.save(self.clustering_algo.results, os.path.join(self.result_folder, 'clustering_results.pt'))
        # if self.result_folder is not None:
        #     self.save_results(self.result_folder, clusters=clusters.squeeze(0).cpu().numpy(), embs = all_embs.cpu().numpy())

    

 
    @torch.no_grad()
    def clustering(self, detector_labels: torch.Tensor, clusters: torch.Tensor, k: torch.Tensor):
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
        device = self.device
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
        means = sum_y / denom                                      # (bs, k_max)
        vars_ = sum_y2 / denom - means.pow(2)
        vars_.clamp_(min=0.0)

    

        if self.bound.lower() == "hoeffding":
            # half = sqrt( log(2/alpha) / (2 n) )
            log_term = torch.log(torch.tensor(2.0 / self.alpha, device=device, dtype=dtype))
            half = torch.sqrt(log_term / (2.0 * denom))
        elif self.bound.lower() == "bernstein":
            # Empirical Bernstein (Audibert et al., 2009) for X in [0,1]:
            # |μ - μ̂| ≤ sqrt( 2 V_n log(3/α) / n ) + 3 log(3/α) / n
            # using empirical (population) variance vars_ above
            log_term = torch.log(torch.tensor(3.0 / self.alpha, device=device, dtype=dtype))
            half = torch.sqrt((2.0 * vars_ * log_term) / denom) + (3.0 * log_term) / denom
        else:
            raise ValueError("bound must be 'hoeffding' or 'bernstein'")
        lower = (means - half).clamp_(0.0, 1.0)
        upper = (means + half).clamp_(0.0, 1.0)

        # ---- mask components j >= k_b and truly empty clusters ----
                                    # (bs, k_max)
        empty   = (counts == 0.0)

        # mask = invalid | empty
        mask = empty
        if mask.any():
            counts[mask] = 0.0
            means[mask]  = 0.0
            vars_[mask]  = 0.0
            lower[mask]  = 0.0
            upper[mask]  = 1.0

        # ---- store ----
        self.cluster_counts       = counts.cpu()                         # (bs, k_max)
        self.cluster_error_means  = means.cpu()                          # (bs, k_max)
        self.cluster_error_vars   = vars_.cpu()                          # (bs, k_max)
        self.cluster_intervals    = torch.stack([lower, upper], dim=-1).cpu()  # (bs, k_max, 2)

        # print("shape intervals", self.cluster_intervals.shape)


    def __call__(self, x=None, logits=None, save_embs=True):
        """
        Returns the cluster **upper bound(s)** for the predicted cluster(s).
        If a single sample is provided, output is (bs,).
        If many samples, output is (bs, n).
        """
        # cluster: (bs,) for one sample or (bs, n) for many
        # cluster = self.predict_clusters(x, logits)
        cluster = self.predict_clusters(x, logits).squeeze(0)
        # print("cluster shape in call", cluster.shape)

        # (bs, Kmax) of upper bounds
        # upper = self.cluster_intervals[..., 1]  # take the upper bound
        # print("cluster shape in call", cluster.shape)
        if self.score == "upper":
            scores = self.cluster_intervals[..., 1].squeeze(0).to(self.device)  # take the upper bound
        elif self.score == "mean":
            scores = self.cluster_error_means.squeeze(0).to(self.device)
        else:
            raise ValueError("Unsupported score type")
        if cluster.dim() == 1:          # (bs,)
            preds = scores.gather(0, cluster) # (bs,)  
        else:
            preds = scores.gather(1, cluster)  

        torch.save(
            cluster, os.path.join(self.result_folder, f"clusters_test_n-clusters-{self.n_clusters}.pt")
        )

       # print("preds shape in call", preds.shape)   

        # Gather per batch along cluster index
        # if cluster.dim() == 1:          # (bs,)
        #     preds = upper.gather(1, cluster.unsqueeze(1)).squeeze(1)  # (bs,)
        # else:                           # (bs, n)
        #     preds = upper.gather(1, cluster)                          # (bs, n)

       # # Optionally stick NaNs on invalid components (if ever present)
       # # preds = preds.masked_fill( ... , float('nan'))
        if preds.dim() == 1:
            return preds.unsqueeze(0)  # (bs, 1)
        return preds  # (bs, 1) or (bs, n, 1)