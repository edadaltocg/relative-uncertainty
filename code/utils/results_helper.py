
import os
import re
import glob
import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import yaml
from matplotlib import rcParams
import seaborn as sns
from scipy import stats

# Set publication-quality matplotlib parameters
def setup_publication_style():
    """Set up matplotlib for publication-quality plots"""
    rcParams.update({
        'figure.figsize': (8, 5),
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'text.usetex': False,  # Set to True if you have LaTeX installed
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.axisbelow': True,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'errorbar.capsize': 3,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True,
        'legend.framealpha': 0.9
    })

def read_table(
    data_name=None,
    model_name=None,
    method_name=None,
    # r=None,
    # seed_split=None,
    # n_epochs=None,
    # transform=None,
    hyperparam=True,
    expe_folder=None,
    # n_folds=None,
    # space=None,
    # distance=None,
    # clustering_name=None,
    # normalize_gini=None,
    num=None,
):
    """
    Load and concatenate all matching result CSVs from `expe_folder`.

    Matching patterns:
      - hyperparam=True  -> 'hyperparams_results*.csv'
      - hyperparam=False -> 'results_opt_fpr*.csv'

    Adds:
      - file_num (int): 0 for base file, i for '_i' files
      - source_file (str): basename of the CSV

    If `num` is provided, only that specific file is read (and annotated).
    """

    # ----- Build experiment folder if not provided -----
    if expe_folder is None:
        expe_folder = RESULTS_FILES.get(data_name, {}).get(model_name, {}).get(method_name, None)
        if expe_folder is None:
            raise ValueError(f"Experiment folder not found for data '{data_name}', model '{model_name}', method '{method_name}'. Please provide expe_folder.")


        # if any(v is None for v in [data_name, model_name, method_name, r, seed_split]):
        #     raise ValueError("When expe_folder is None, provide data_name, model_name, method_name, r, seed_split.")
        # root = f"../../results/{data_name}_{model_name}_r-{r}_seed-split-{seed_split}"
        # if method_name in ["clustering", "metric_learning", "random_forest"]:
        #     if any(v is None for v in [transform, n_epochs, n_folds, space]):
        #         raise ValueError("transform, n_epochs, n_folds, space required for these methods when expe_folder is None.")
        #     root += f"/transform-{transform}_n-epoch{n_epochs}_n-folds{n_folds}_{space}"
        #     if method_name == "clustering":
        #         if distance is not None:
        #             root += f"_distance-{distance}"
        #         if clustering_name != "soft-kmeans":
        #             root += f"_{clustering_name}"
        # elif method_name == "gini":
        #     if any(v is None for v in [transform, normalize_gini]):
        #         raise ValueError("transform and normalize_gini required for method 'gini' when expe_folder is None.")
        #     root += f"/transform-{transform}_normalize-{normalize_gini}"
        # else:
        #     if transform is None:
        #         raise ValueError("transform is required when expe_folder is None.")
        #     root += f"/transform-{transform}"
        # expe_folder = root + f"_{method_name}"

    # base_name = "hyperparams_results" if hyperparam else "results_opt_fpr"
    # rx = re.compile(rf"{re.escape(base_name)}(?:_(\d+))?\.csv$")
    if hyperparam:
        candidate_basenames = ["hyperparams_results"]
    else:
        candidate_basenames = ["results_opt_fpr", "results_opt_fpr_", "all_results"]  # <- accept both

    # Build (basename -> regex) for suffix parsing
    regex_by_base = {
        b: re.compile(rf"{re.escape(b)}(?:_(\d+))?\.csv$", re.IGNORECASE)
        for b in candidate_basenames
    }

    # ---------- Specific file requested ----------
    if num is not None:
        # Try each basename until one exists
        tried = []
        for base_name in candidate_basenames:
            fname = f"{base_name}.csv" if str(num) in ("", "0") else f"{base_name}_{num}.csv"
            target = os.path.join(expe_folder, fname)
            tried.append(target)
            if os.path.isfile(target):
                df = pd.read_csv(target)
                m = regex_by_base[base_name].search(os.path.basename(target))
                file_num = int(m.group(1)) if (m and m.group(1) is not None) else 0
                df = df.copy()
                df["file_num"] = file_num
                df["source_file"] = os.path.basename(target)
                return df
        raise FileNotFoundError(f"No such file among candidates:\n  " + "\n  ".join(tried))

    # ---------- Otherwise, gather *all* matching files ----------
    files = []
    for base_name in candidate_basenames:
        pattern = os.path.join(expe_folder, f"{base_name}*.csv")
        files.extend(
            f for f in glob.glob(pattern)
            if regex_by_base[base_name].search(os.path.basename(f))
        )

    if not files:
        pats = [os.path.join(expe_folder, f"{b}*.csv") for b in candidate_basenames]
        raise FileNotFoundError("No matching files found. Tried:\n  " + "\n  ".join(pats))

    # Sort: base without suffix first, then _NN in numeric order; keep basenames grouped
    def sort_key(p):
        bn = os.path.basename(p)
        # identify which regex matched
        for base_name, rx in regex_by_base.items():
            m = rx.search(bn)
            if m:
                idx = m.group(1)
                return (candidate_basenames.index(base_name), 0 if idx is None else 1, int(idx) if idx else -1)
        return (len(candidate_basenames), 1, 10**9)

    files.sort(key=sort_key)

    dfs = []
    for fpath in files:
        df_i = pd.read_csv(fpath)
        # parse suffix using the matching regex
        file_num = 0
        matched = False
        for base_name, rx in regex_by_base.items():
            m = rx.search(os.path.basename(fpath))
            if m:
                matched = True
                if m.group(1) is not None:
                    file_num = int(m.group(1))
                break
        if not matched:
            file_num = 0
        df_i = df_i.copy()
        df_i["file_num"] = file_num
        df_i["source_file"] = os.path.basename(fpath)
        dfs.append(df_i)

    return pd.concat(dfs, axis=0, ignore_index=True)


def pretty_name(name: str) -> str:
    """
    Map raw keys (models or metrics) to nice display names.
    - Keeps underscores so metric keys like 'fpr_val_cross' match.
    - Still strips spaces and hyphens; lowercases for robust lookup.
    """
    # key = name.strip().lower().replace(" ", "").replace("-", "")  # NOTE: no .replace("_","")

    table = {
        # --- Datasets ---
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "imagenet": "ImageNet",
        # --- Models ---
        "resnet34": "ResNet-34",
        "densenet121": "DenseNet-121",
        "timm_vit_base16": "ViT-B/16",
        "timm_vit_tiny16": "ViT-Tiny/16",
        "timm-vit-base16": "ViT-B/16",
        "timm-vit-tiny16": "ViT-Tiny/16",
        # --- Methods ---
        "soft-kmeans_torch": "Soft KMeans",
        "kmeans": "KMeans",
        # --- Metrics (val_cross variants etc.) ---
        "aupr_val_cross": "AUPR",
        "auroc_val_cross": "AUROC",
        "roc_auc_test": "AUROC Test",
        "fpr_val_cross": "FPR@95",
        "fpr_val": "FPR@95",
        "fpr_test": "FPR@95",
        "aurc_val_cross": "AURC",
        "aupr_err_val_cross": "AUPR-Err",
        "aupr_success_val_cross": "AUPR-Succ",
        # common alternates (optional)
        "roc_auc_val": "AUROC",
        "aurc_val": "AURC",
        "fpr_val": "FPR@95",
        "aupr_err_val": "AUPR-Err",
        "aupr_success_val": "AUPR-Succ",
        # Variable
        "temperature": "Temperature",
        "n_clusters": r"Number of level sets $\mathcal{X}_z$ (i.e., $|\mathcal{Z}|$)"

    }

    if name in table:
        return table[name]
    else:
        raise KeyError(f"pretty_name: no mapping for '{name}' (key='{name}')")

 

RESULTS_FILES = {
    "cifar10": {
        "resnet34": {
            "clustering": "results/cifar10_resnet34_r-2_seed-split-9/transform-test_n-epoch1_n-folds3_probits_clustering",
            "metric_learning": "results/cifar10_resnet34_r-2_seed-split-9/transform-test_n-epoch1_metric_learning",
            "gini": "results/cifar10_resnet34_r-2_seed-split-9/transform-test_gini",
            "max_proba": "results/cifar10_resnet34_r-2_seed-split-9/transform-test_max_proba",
        },
        "densenet121": {
            "clustering": "results/cifar10_densenet121_r-2_seed-split-9/transform-test_n-epoch1_n-folds3_probits_clustering",
            "metric_learning": "results/cifar10_densenet121_r-2_seed-split-9/transform-test_n-epoch1_metric_learning",
            "gini": "results/cifar10_densenet121_r-2_seed-split-9/transform-test_gini",
            "max_proba": "results/cifar10_densenet121_r-2_seed-split-9/transform-test_max_proba",
        },
    },
    "cifar100": {
        "resnet34": {
            "clustering": "results/cifar100_resnet34_r-2_seed-split-9/transform-test_n-epoch1_n-folds3_probits_clustering",
            "metric_learning": "results/cifar100_resnet34_r-2_seed-split-9/transform-test_n-epoch1_metric_learning",
            "gini": "results/cifar100_resnet34_r-2_seed-split-9/transform-test_gini_normalized-True",
            "max_proba": "results/cifar100_resnet34_r-2_seed-split-9/transform-test_max_proba",
        },
        "densenet121": {
            "clustering": "results/cifar100_densenet121_r-2_seed-split-9/transform-test_n-epoch1_n-folds3_probits_clustering",
            "metric_learning": "results/cifar100_densenet121_r-2_seed-split-9/transform-test_n-epoch1_metric_learning",
            "gini": "results/cifar100_densenet121_r-2_seed-split-9/transform-test_gini",
            "max_proba": "results/cifar100_densenet121_r-2_seed-split-9/transform-test_max_proba",

        },

    },
    "imagenet": {
        "timm_vit_base16": {
            "clustering": "results/imagenet_timm_vit_base16_r-2_seed-split-9/transform-test_n-epoch1_clustering",
            "metric_learning": "results/imagenet_timm_vit_base16_r-2_seed-split-9/transform-test_n-epoch1_probits_metric_learning",
            "gini": "results/imagenet_timm_vit_base16_r-2_seed-split-9/transform-test_gini",
            "max_proba": "results/imagenet_timm_vit_base16_r-2_seed-split-9/transform-test_max_proba",
        },
        "timm_vit_tiny16": {
            "clustering": "results/imagenet_timm_vit_tiny16_r-2_seed-split-9/transform-test_n-epoch1_probits_clustering",
            "metric_learning": "results/imagenet_timm_vit_tiny16_r-2_seed-split-9/transform-test_n-epoch1_probits_metric_learning",
            "gini": "results/imagenet_timm_vit_tiny16_r-2_seed-split-9/transform-test_normalized-False_gini",
            "max_proba": "results/imagenet_timm_vit_tiny16_r-2_seed-split-9/transform-test_max_proba",
        },
    },
    "imagenet_fair": {
        "timm_vit_base16": {
            "clustering": "fair_ce_results/imagenet_timm_vit_base16_r-0.5_seed-split-9/transform-test_n-epoch1_n-folds3_probits_soft-kmeans_torch_bernstein_n-init-10_kmeans_clustering",
        },
        "timm_vit_tiny16": {
            "clustering": "fair_ce_results/imagenet_timm_vit_tiny16_r-0.5_seed-split-9/transform-test_n-epoch1_n-folds3_probits_soft-kmeans_torch_bernstein_n-init-10_kmeans_clustering",
        },
    },
}



OFFICIAL_RESULTS_FILES = {
    "cifar10": {
        "resnet34": {
            "clustering_hyperparams": "results/cifar10_resnet34_r-2_seed-split-9/transform-test_n-epoch1_n-folds10_probits_probweights_clustering/hyperparams_results.csv",
            "clustering_opt": "results/cifar10_resnet34_r-2_seed-split-9/transform-test_n-epoch1_n-folds10_probits_probweights_clustering/all_results.csv",
            "metric_learning": "results/cifar10_resnet34_r-2_seed-split-9/transform-test_n-epoch1_metric_learning",
            "gini": "results/cifar10_resnet34_r-2_seed-split-9/transform-test_gini",
            "max_proba": "results/cifar10_resnet34_r-2_seed-split-9/transform-test_max_proba",
        },
        "densenet121": {
            "clustering_hyperparams": "results/cifar10_densenet121_r-2_seed-split-9/transform-test_n-epoch1_n-folds10_probits_predweight_clustering/hyperparams_results.csv",
            "clustering_opt": "results/cifar10_densenet121_r-2_seed-split-9/transform-test_n-epoch1_n-folds10_probits_predweight_clustering/all_results.csv",
            "metric_learning": "results/cifar10_densenet121_r-2_seed-split-9/transform-test_n-epoch1_metric_learning",
            "gini": "results/cifar10_densenet121_r-2_seed-split-9/transform-test_gini",
            "max_proba": "results/cifar10_densenet121_r-2_seed-split-9/transform-test_max_proba",
        },
    },
    "cifar100": {
        "resnet34": {
            "clustering_hyperparams": "results/cifar100_resnet34_r-2_seed-split-9/transform-test_n-epoch1_n-folds3_probits_clustering/hyperparams_results.csv",
            "clustering_opt": "results/cifar100_resnet34_r-2_seed-split-9/transform-test_n-epoch1_n-folds3_probits_clustering/all_results.csv",
            "metric_learning": "results/cifar100_resnet34_r-2_seed-split-9/transform-test_n-epoch1_metric_learning",
            "gini": "results/cifar100_resnet34_r-2_seed-split-9/transform-test_gini_normalized-True",
            "max_proba": "results/cifar100_resnet34_r-2_seed-split-9/transform-test_max_proba",
        },
        "densenet121": {
            "clustering_hyperparams": "results/cifar100_densenet121_r-2_seed-split-9/transform-test_n-epoch1_n-folds10_probits_predweight_clustering/hyperparams_results.csv",
            "clustering_opt": "results/cifar100_densenet121_r-2_seed-split-9/transform-test_n-epoch1_n-folds10_probits_predweight_clustering/all_results.csv",
            "metric_learning": "results/cifar100_densenet121_r-2_seed-split-9/transform-test_n-epoch1_metric_learning",
            "gini": "results/cifar100_densenet121_r-2_seed-split-9/transform-test_gini",
            "max_proba": "results/cifar100_densenet121_r-2_seed-split-9/transform-test_max_proba",

        },

    },
    "imagenet": {
        "timm_vit_base16": {
            "clustering_hyperparams": "fair_ce_results/imagenet_timm_vit_base16_r-0.5_seed-split-9/transform-test_n-epoch1_n-folds3_probits_soft-kmeans_torch_bernstein_n-init-10_kmeans_clustering/hyperparams_results.csv",
            "clustering_opt": "fair_ce_results/imagenet_timm_vit_base16_r-0.5_seed-split-9/transform-test_n-epoch1_n-folds3_probits_soft-kmeans_torch_bernstein_n-init-10_kmeans_clustering/results_opt_fpr.csv",
            "metric_learning": "results/imagenet_timm_vit_base16_r-2_seed-split-9/transform-test_n-epoch1_probits_metric_learning",
            "gini": "results/imagenet_timm_vit_base16_r-2_seed-split-9/transform-test_gini",
            "max_proba": "results/imagenet_timm_vit_base16_r-2_seed-split-9/transform-test_max_proba",
        },
        "timm_vit_tiny16": {
            "clustering_hyperparams": "fair_ce_results/imagenet_timm_vit_tiny16_r-0.5_seed-split-9/transform-test_n-epoch1_n-folds3_probits_soft-kmeans_torch_bernstein_n-init-10_kmeans_clustering/hyperparams_results_2.csv",
            "clustering_opt": "fair_ce_results/imagenet_timm_vit_tiny16_r-0.5_seed-split-9/transform-test_n-epoch1_n-folds3_probits_soft-kmeans_torch_bernstein_n-init-10_kmeans_clustering/results_opt_fpr_2.csv",
            "metric_learning": "results/imagenet_timm_vit_tiny16_r-2_seed-split-9/transform-test_n-epoch1_probits_metric_learning",
            "gini": "results/imagenet_timm_vit_tiny16_r-2_seed-split-9/transform-test_normalized-False_gini",
            "max_proba": "results/imagenet_timm_vit_tiny16_r-2_seed-split-9/transform-test_max_proba",
        },
    },
}


NEW_OFFICIAL_RESULTS_FILES = {
    "cifar10": {
        "resnet34": {
            "clustering_hyperparams": "results/cifar10_resnet34_r-2_seed-split-9/transform-test_n-epoch1_n-folds10_probits_probweights_clustering/hyperparams_results.csv",
            "clustering_opt": "results/cifar10_resnet34_r-2_seed-split-9/transform-test_n-epoch1_n-folds10_probits_probweights_clustering/all_results.csv",
            "metric_learning": "results/cifar10_resnet34_r-2_seed-split-9/transform-test_n-epoch1_metric_learning",
            "gini": "results/cifar10_resnet34_r-2_seed-split-9/transform-test_gini",
            "max_proba": "results/cifar10_resnet34_r-2_seed-split-9/transform-test_max_proba",
        },
        "densenet121": {
            "clustering_hyperparams": "results/cifar10_densenet121_r-2_seed-split-9/transform-test_n-epoch1_n-folds10_probits_predweight_clustering/hyperparams_results.csv",
            "clustering_opt": "results/cifar10_densenet121_r-2_seed-split-9/transform-test_n-epoch1_n-folds10_probits_predweight_clustering/all_results.csv",
            "metric_learning": "results/cifar10_densenet121_r-2_seed-split-9/transform-test_n-epoch1_metric_learning",
            "gini": "results/cifar10_densenet121_r-2_seed-split-9/transform-test_gini",
            "max_proba": "results/cifar10_densenet121_r-2_seed-split-9/transform-test_max_proba",
        },
    },
    "cifar100": {
        "resnet34": {
            "clustering_hyperparams": "results/cifar100_resnet34_r-2_seed-split-9/transform-test_n-epoch1_n-folds3_probits_clustering/hyperparams_results.csv",
            "clustering_opt": "results/cifar100_resnet34_r-2_seed-split-9/transform-test_n-epoch1_n-folds3_probits_clustering/all_results.csv",
            "metric_learning": "results/cifar100_resnet34_r-2_seed-split-9/transform-test_n-epoch1_metric_learning",
            "gini": "results/cifar100_resnet34_r-2_seed-split-9/transform-test_gini_normalized-True",
            "max_proba": "results/cifar100_resnet34_r-2_seed-split-9/transform-test_max_proba",
        },
        "densenet121": {
            "clustering_hyperparams": "results/cifar100_densenet121_r-2_seed-split-9/transform-test_n-epoch1_n-folds10_probits_predweight_clustering/hyperparams_results.csv",
            "clustering_opt": "results/cifar100_densenet121_r-2_seed-split-9/transform-test_n-epoch1_n-folds10_probits_predweight_clustering/all_results.csv",
            "metric_learning": "results/cifar100_densenet121_r-2_seed-split-9/transform-test_n-epoch1_metric_learning",
            "gini": "results/cifar100_densenet121_r-2_seed-split-9/transform-test_gini",
            "max_proba": "results/cifar100_densenet121_r-2_seed-split-9/transform-test_max_proba",

        },

    },
    "imagenet": {
        "timm_vit_base16": {
            "clustering_hyperparams": "fair3_calib-1250_ce_results/imagenet_timm_vit_base16_r-0.5_seed-split-9/transform-test_n-epoch1_n-folds10_probits_soft-kmeans_torch_bernstein_n-init-10_kmeans_clustering/hyperparams_results_without-1.15_1.1_cluster-more-250.csv",
            "clustering_opt": "fair3_calib-1250_ce_results/imagenet_timm_vit_base16_r-0.5_seed-split-9/transform-test_n-epoch1_n-folds10_probits_soft-kmeans_torch_bernstein_n-init-10_kmeans_clustering/hyperparams_results_without-1.15_1.1_cluster-more-250._opt_fpr.csv",
            "metric_learning": "results/imagenet_timm_vit_base16_r-2_seed-split-9/transform-test_n-epoch1_probits_metric_learning",
            "gini": "results/imagenet_timm_vit_base16_r-2_seed-split-9/transform-test_gini",
            "max_proba": "results/imagenet_timm_vit_base16_r-2_seed-split-9/transform-test_max_proba",
        },
        "timm_vit_tiny16": {
            "clustering_hyperparams": "fair3_calib-1250_ce_results/imagenet_timm_vit_tiny16_r-0.5_seed-split-9/transform-test_n-epoch1_n-folds10_probits_soft-kmeans_torch_bernstein_n-init-10_kmeans_clustering/hyperparams_results_without-1.02_clusters-more-300.csv",
            "clustering_opt": "fair3_calib-1250_ce_results/imagenet_timm_vit_tiny16_r-0.5_seed-split-9/transform-test_n-epoch1_n-folds10_probits_soft-kmeans_torch_bernstein_n-init-10_kmeans_clustering/hyperparams_results_without-1.02_clusters-more-300._opt_fpr.csv",
            "metric_learning": "results/imagenet_timm_vit_tiny16_r-2_seed-split-9/transform-test_n-epoch1_probits_metric_learning",
            "gini": "results/imagenet_timm_vit_tiny16_r-2_seed-split-9/transform-test_normalized-False_gini",
            "max_proba": "results/imagenet_timm_vit_tiny16_r-2_seed-split-9/transform-test_max_proba",
        },
    },
}

LIST_DATASETS = list(NEW_OFFICIAL_RESULTS_FILES.keys())
LIST_MODELS = {dataset: [] for dataset in NEW_OFFICIAL_RESULTS_FILES}
for dataset in NEW_OFFICIAL_RESULTS_FILES.keys():
    for model in NEW_OFFICIAL_RESULTS_FILES[dataset].keys():
    
        LIST_MODELS[dataset].append(model)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Print summary table of results.")
    parser.add_argument("--data_name",  default="cifar10",
                        help="Dataset namme")
    parser.add_argument("--model_name", type=str, default="resnet34",
                        help="Model name.")
    parser.add_argument("--method_name", type=str, default="clustering",
                        help="Method name.")
    parser.add_argument("--hyperparam", action="store_true", default=False)
    parser.add_argument("--metric", type=str, default="fpr_val_cross")
    args = parser.parse_args()
    df = read_table(
        data_name=args.data_name,
        model_name=args.model_name,
        method_name=args.method_name,
        hyperparam=args.hyperparam,
    )
    method_cols = [col for col in df.columns if col.startswith(f"{args.method_name}_")]
    cols = method_cols + [args.metric] + ["file_num", "source_file"]
    print(df[cols].sort_values(by=args.metric, ascending=True).to_string(index=False))