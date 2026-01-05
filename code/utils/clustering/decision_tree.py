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


from sklearn.tree import DecisionTreeClassifier

# import numpy as np
# from sklearn.cluster._kmeans import _kmeans_plusplus, row_norms

__all__ = ["Decision_Tree"]


#

class TreeQuantizer(nn.Module):
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

   

    def __init__(
        self,
  
        n_clusters: Optional[int] = None,
        splitter="best",
        max_depth=None,
        class_weight="balanced",
        min_samples_leaf=1,
        seed: Optional[int] = 123,
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
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__() 
    
        self.n_clusters = n_clusters
        self.splitter = splitter
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.verbose = verbose
        self.device = device
        self.seed = seed
        self.min_samples_leaf = min_samples_leaf

      
        self.results = None
   
        
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

        self.tree = DecisionTreeClassifier(
            random_state=self.seed,
            splitter=self.splitter,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            max_leaf_nodes=self.n_clusters,
            min_samples_leaf=self.min_samples_leaf,
            monotonic_cst=[-1] 
        )



        self.MATCHING = None
        self.actual_n_clusters = None

    @torch.no_grad()
    def forward(self, x: Tensor, y: Tensor):
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        if isinstance(y, torch.Tensor):
            y_np = y.detach().cpu().numpy()
        else:
            y_np = y
        self.tree.fit(x_np, y_np)

        # Build mapping using training data (or any reference set)
        leafs = self.tree.apply(x_np)                # node indices
        unique_leafs_ids = np.unique(leafs)
        self.MATCHING = {leaf_id: j
                         for j, leaf_id in enumerate(unique_leafs_ids)}
        self.n_clusters = len(unique_leafs_ids)
        return self

    @torch.no_grad()
    def predict(self, x: Tensor) -> torch.LongTensor:
        if self.MATCHING is None:
            raise RuntimeError("TreeQuantizer.predict called before fit")
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        leafs = self.tree.apply(x_np)  # node indices

        compact_ids = np.array([self.MATCHING[l] for l in leafs], dtype=int)
        clusters = torch.from_numpy(compact_ids).long().to(self.device)
        # 
        return clusters  # shape (n_samples,)

    @torch.no_grad()
    def fit_predict(self, x: Tensor, y: Tensor, **kwargs) -> torch.LongTensor:
        self.forward(x, y)
        return self.predict(x)
        # return self.results.labels.to(self.device)

 
    def _init_info(self):
        """Record additional info before clustering."""

        self.best_init = None


    def _record_info(self, x, centers, invalid, k, iter):
        """Record additional info after clustering."""
        

        clusters_res = self.predict(x)
        clusters_cal = self.predict(self.cal_probits)
        clusters_test = self.predict(self.test_probits)
        clusters_val = self.predict(self.val_probits)

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




        

   


if __name__ == "__main__":

  
    gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    n_clusters = None
    splitter="best"
    max_depth= None
    class_weight="balanced"
    min_samples_leaf=20
    
    order = True
    init_scheme_covs = "statistics"  # statistic, uniform
    bound = "bernstein"

    space = "probits"  # "logits"  # "probits"


    seed = 3
    n_split_val = 1
    weight_std = 0
    ratio_res = 0.99
    mode = "search_res"   # controls which split you use as *validation* for selection
    # NEW: controls on which split you do the counting for upper bounds
    # choose one of: "res_tr", "res_val"
    upper_source = "res_tr"

    model_name = "timm-vit-base16"
    data_name = "imagenet"
    n_res = 10000
    n_cal = 15000
    n_test = 25000

    seed_split = 9
    subclasses = None
    temperature = 1.1
    n_dim = 1
    quantizer_metric = "fpr"  # only used for naming

    # ------ Read Latent ------
    latent_rooth = f"./latent/{data_name}_{model_name}_n-res-{n_res}_n-cal-{n_cal}/seed-split-{seed_split}"
    latent_path = f"{latent_rooth}/res_n-samples-{n_res}_transform-test_n-epochs-1.pt"
    probits, errors = read_probits(latent_path, order=order, subclasses=subclasses, temperature=temperature, space=space, n_dim=n_dim)
    latent_path_cal = f"{latent_rooth}/cal_n-samples-{n_cal}_transform-test_n-epochs-1.pt"
    cal_probits, cal_errors = read_probits(latent_path_cal, order=order, subclasses=subclasses, temperature=temperature, space=space, n_dim=n_dim)
    latent_path_test = f"{latent_rooth}/test_n-samples-{n_test}.pt"
    test_probits, test_errors = read_probits(latent_path_test, order=order, subclasses=subclasses, temperature=temperature, space=space, n_dim=n_dim)
    print("Number of res samples:", probits.shape)
    print("Number of cal samples:", cal_probits.shape)
    print("Number of test samples:", test_probits.shape)

    probits = probits.to(device)
    errors = errors.to(device)
    test_probits = test_probits.to(device)
    test_errors = test_errors.to(device)
    cal_probits = cal_probits.to(device)
    cal_errors = cal_errors.to(device)

    n_tr = int(ratio_res * n_res)
    tr_idx = np.arange(n_tr)
    va_idx = np.arange(n_tr, n_res)
    n_val = n_res - n_tr
    probits_res_tr = probits[tr_idx]
    errors_res_tr = errors[tr_idx]
    probits_res_va = probits[va_idx]
    errors_res_va = errors[va_idx]




    upper_str = "" if upper_source == "res_tr" else "_upper-val"
    n_dim_str = "" if n_dim is None else f"_ndim-{n_dim}"
    save_folder  = os.path.join(
        "./code/utils/clustering/decision-tree_results",
        f"{data_name}_{model_name}",
        f"n_res-{n_res}_n_cal-{n_cal}",
        "seed-split-9",
        f"results_opt-{quantizer_metric}_qunatiz-metric-{quantizer_metric}-ratio-{ratio_res}"
        f"_n-split-val-{n_split_val}_weight-std-{weight_std}_mode-{mode}{upper_str}_seed-{seed}",
        f"{n_dim_str}_K-{n_clusters}_bound-{bound}_spliiter-{splitter}_maxdepth-{max_depth}_classweight-{class_weight}"
    )
  
       
    # save_folder  = f"./code/utils/clustering/toy_example_with_logdet_campled_min-{var_min}_cov/seed_{seed}/"
    os.makedirs(save_folder, exist_ok=True)
    print("save filer:", save_folder)

   

    quantizer = TreeQuantizer(
                seed=seed,
                verbose=0, 
                n_clusters=n_clusters,
                splitter=splitter,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                res_errors=errors_res_tr,
                cal_errors=cal_errors,
                cal_probits=cal_probits,
                test_errors=test_errors,
                test_probits=test_probits,
                val_probits=probits_res_va,
                val_errors=errors_res_va,
                bound=bound,
                tol=None,
                device=device,
            
            )
    clusters_res_tr = quantizer.fit_predict(probits_res_tr, errors_res_tr).unsqueeze(0)
    #print("shape clusters res tr:", clusters_res_tr.shape)
    # print("Iterations:", model.n_iter)
    # print("FPR test:", model.results.list_results_test.iloc[-1]["fpr"])
    # print("Thr test:", model.results.list_results_test.iloc[-1]["thr"])

    ################################################################################

 
    n_clusters = quantizer.tree.get_n_leaves()
    k = torch.tensor(n_clusters, device=device)
    clusters_res_val = quantizer.predict(probits_res_va).unsqueeze(0)
    clusters_cal = quantizer.predict(cal_probits).unsqueeze(0)
    clusters_test = quantizer.predict(test_probits).unsqueeze(0)
    clusters_dic = {"res_tr": clusters_res_tr, "res_val": clusters_res_val, "cal": clusters_cal, "test": clusters_test}
    errors_dic = {"res_tr": errors_res_tr, "res_val": errors_res_va, "cal": cal_errors, "test": test_errors}
    print("shape clusters res val:", clusters_res_val.shape)
    print("shape clusters cal:", clusters_cal.shape)
    print("shape clusters test:", clusters_test.shape)
    
    
    
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
          
            scores =  upper.to(device).gather(1, clusters)
        
           
            # print("score shape:", scores.shape)
            results = compute_all_metrics(
            conf=scores.cpu(),
            detector_labels=errors.cpu(),
        )
            # print("fpr shape:", np.shape(fpr))
            results = pd.DataFrame(results)
          
    
            
            classif_results[split][upper_type] = results
    

    
    best_init = np.argmin(classif_results["res_val"]["res_tr"][quantizer_metric])
                   

                                         
    # Plot res_va vs test results
    print("MEan Test FPR:", np.mean(classif_results["test"]["cal"]["fpr"]))
    print("Min Test FPR:", np.min(classif_results["test"]["cal"]["fpr"]))
    print("Best init FPR test:", classif_results["test"]["cal"].iloc[best_init]["fpr"])