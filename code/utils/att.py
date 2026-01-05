import os 
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
from code.utils.results_helper import setup_publication_style, pretty_name, LIST_MODELS, LIST_DATASETS
import torch
from typing import Optional, Tuple, List

def _coerce_metric_column(series: pd.Series) -> pd.Series:
    """
    Coerce possibly stringified numbers like '[0.123]' or '0.123' into float.
    Non-parsable entries become NaN (then dropped).
    """
    def _to_float(x):
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        if isinstance(x, str):
            s = x.strip()
            # strip wrapping brackets if any
            if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
                s = s[1:-1].strip()
            try:
                return float(s)
            except Exception:
                return np.nan
        return np.nan

    out = series.apply(_to_float)
    return out

def aggregate_results(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in results.columns:
        raise KeyError(f"Metric '{metric}' not found in results columns: {list(results.columns)}")

    results = results.copy()
    results[metric] = _coerce_metric_column(results[metric])
    results = results.dropna(subset=[metric])

    agg = (
        results
        .groupby("n_cal", dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .fillna({"std": 0.0})  # if a group has a single sample, std becomes NaN -> set to 0
        .sort_index()
    )
    return agg

def plot_metric_curve(
    df,
    metric: str,
    fixed: str = "temperature",            # "temperature" or "n_clusters"
    save_file= None,
    fixed_value: Optional[float] = None,   # if None: auto-select the best for THIS df
    min_value: Optional[float] = None,     # filter free axis (>=)
    max_value: Optional[float] = None,     # filter free axis (<=)
    metric_col_suffix: str = "_val_cross",
    std_col_suffix: str = "_val_cross_std",
    temp_col: str = "clustering.temperature",
    k_col: str = "clustering.n_clusters",
    ascending: Optional[bool] = None,      # None -> infer from metric name
    title_prefix: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    fix_tol: float = 1e-10,                # tolerance for float equality on fixed temperature
    print_duplicates: bool = False,        # print rows before raising
    label: Optional[str] = None,           # legend label for this series
) -> Tuple[float, plt.Axes]:
    """Strict plotting: raises if duplicate x remain after fixing one hyperparameter.
       Column names can be dotted or underscored (auto-resolved)."""

    # import pandas as pd

    def _resolve_col(df, preferred: str, alts: Optional[List[str]] = None) -> str:
        cands = [preferred]
        if "." in preferred: cands.append(preferred.replace(".", "_"))
        if "_" in preferred: cands.append(preferred.replace("_", "."))
        if alts:
            cands += alts
            for a in list(alts):
                if "." in a: cands.append(a.replace(".", "_"))
                if "_" in a: cands.append(a.replace("_", "."))
        for c in cands:
            if c in df.columns: return c
        raise KeyError(f"None of these columns found: {cands}")

    temp_col = _resolve_col(df, temp_col, alts=["temperature","temp","clustering_temperature"])
    k_col    = _resolve_col(df,  k_col,   alts=["n_clusters","k","clustering_n_clusters"])

    mean_col = f"{metric}{metric_col_suffix}"
    std_col  = f"{metric}{std_col_suffix}"
    for c in (mean_col, std_col):
        if c not in df.columns:
            raise KeyError(f"Required metric column '{c}' not in df")

    if ascending is None:
        lower_is_better = {"fpr","aurc","err","nll","loss","brier","ece"}
        ascending = metric.lower() in lower_is_better  # True => minimize

    if fixed not in {"temperature","n_clusters"}:
        raise ValueError("`fixed` must be 'temperature' or 'n_clusters'.")

    if fixed == "temperature":
        fixed_col = temp_col; x_col = k_col
        fixed_name = "temperature"; free_name = r"Number of level sets $\mathcal{X}_z$"
    else:
        fixed_col = k_col; x_col = temp_col
        fixed_name = "number of clusters"; free_name = "Temperature"

    keep = [temp_col, k_col, mean_col, std_col]
    df_clean = df[keep].dropna()

    # choose best fixed value for THIS df, if not supplied
    if fixed_value is None:
        fixed_value = df_clean.sort_values(mean_col, ascending=ascending).iloc[0][fixed_col]

    # slice by fixed value (with tolerance for float temperatures)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    for fixed_value in [150, 170, 200, 240, 270, 300]:
        if fixed == "temperature":
            vals = df_clean[fixed_col].to_numpy()
            mask = np.isclose(vals, fixed_value, rtol=0.0, atol=fix_tol)
            df_sel = df_clean.loc[mask].copy()
        else:
            df_sel = df_clean.loc[df_clean[fixed_col] == fixed_value].copy()

        if df_sel.empty:
            raise ValueError(f"No rows after fixing {fixed_col}={fixed_value}.")

        # optional range filter on the free axis
        if min_value is not None: df_sel = df_sel[df_sel[x_col] >= min_value]
        if max_value is not None: df_sel = df_sel[df_sel[x_col] <= max_value]
        if df_sel.empty:
            raise ValueError("All rows filtered out by min_value/max_value.")

        # strict duplicate check
        vc = df_sel[x_col].value_counts().sort_index()
        dup = vc[vc > 1]
        if not dup.empty:
            if print_duplicates:
                print("Duplicate x-values:", dict(dup))
                with pd.option_context("display.width", 160):
                    print(df_sel[df_sel[x_col].isin(dup.index)]
                        .sort_values([x_col, temp_col, k_col, mean_col])
                        .to_string(index=False))
            raise ValueError(
                f"Duplicate x-values for {x_col} after fixing {fixed_col}={fixed_value}: {dict(dup)}"
            )

        # plot
        df_plot = df_sel.sort_values(x_col)
        x = df_plot[x_col].to_numpy()
        y = df_plot[mean_col].to_numpy()
        s = np.clip(df_plot[std_col].to_numpy(), 0.0, None)

        if np.any(~np.isfinite(x) | ~np.isfinite(y) | ~np.isfinite(s)):
            raise ValueError("Non-finite values in x/mean/std.")

       
        line, = ax.plot(x, y, marker="o", label=fixed_value)
        ax.fill_between(x, y - s, y + s, alpha=0.20, linewidth=0)

        # if title_prefix is None:
        #     title_prefix = f"{metric.upper()} vs {free_name}"
        # ax.set_title(title_prefix)
        ax.set_xlabel(free_name)
        ax.set_ylabel(mean_col)
        ax.set_title(f"{metric.upper()} vs {free_name} (fixed {fixed_name}={fixed_value})")
        ax.grid(True)
    # Saving
    save_file=os.path.join(ROOT, f"{file}_fixed-{fixed}-{fixed_value}_{args.metric}.png")
    if save_file is not None:
        plt.savefig(save_file)
        print("Saved plot to", save_file)




# def read_table(data_name, model_name, res):

#     result_folder = os.path.join(ROOT, f"{args.data_name}_{args.model_name}_n_cal")
#     if res != "":
#         result_folder += res
#     result_file = os.path.join(result_folder, "results.csv")
#     results = pd.read_csv(result_file)
#     return results, result_folder




def _coerce_metric_column(series: pd.Series) -> pd.Series:
    """
    Coerce possibly stringified numbers like '[0.123]' or '0.123' into float.
    Non-parsable entries become NaN (then dropped).
    """
    def _to_float(x):
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        if isinstance(x, str):
            s = x.strip()
            # strip wrapping brackets if any
            if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
                s = s[1:-1].strip()
            try:
                return float(s)
            except Exception:
                return np.nan
        return np.nan

    out = series.apply(_to_float)
    return out

def aggregate_results(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in results.columns:
        raise KeyError(f"Metric '{metric}' not found in results columns: {list(results.columns)}")

    results = results.copy()
    results[metric] = _coerce_metric_column(results[metric])
    results = results.dropna(subset=[metric])
    if args.fixed == "n_clusters":
        var = "temperature"
    else:
        var = "n_clusters"
    agg = (
        results
        .groupby(var, dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .fillna({"std": 0.0})  # if a group has a single sample, std becomes NaN -> set to 0
        .sort_index()
    )
    return agg

names_legend = {
    "temperature": "T",
    "n_clusters" : r"$|\mathcal{Z}|$"
}

def plot_ablation_single(ax, results: pd.DataFrame, metric: str, label: str = None, fixed_value=1):
    agg = aggregate_results(results, metric)
    x = agg.index.values
    y = agg["mean"].values
    s = agg["std"].values

    line, = ax.plot(x, y, marker='o', label=f"{pretty_name(label)} - {names_legend[args.fixed]} = {fixed_value}")
    ax.fill_between(x, y - s, y + s, alpha=0.2)  # uses same color automatically
    return line

def plot_ablation(data_name, model, metric, fixed, xlabel=r'$|\mathcal{D}_{\mathrm{cal}}|$'):

    results = pd.read_csv(os.path.join(ROOT, f"{data_name}_{model}/results_{fixed}.csv"))

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    plot_ablation_single(ax, results, metric)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(pretty_name(metric))
    ax.set_ylim(0.2, 0.7)  # y-axis starts at 0
    ax.grid(True)
    plt.tight_layout()
    save_path = os.path.join(ROOT, f"{file}_viz")
    plt.savefig(save_path, dpi=200)
    plt.show()
    print("Saved plot to", save_path)


def plot_ablation_both(data_name: str, model_list, metric: str, 
                       n_res: str = None,
                         fixed="temperature"):
    """
    Plot both models on a single figure with legend.
    Saves into a shared folder: ./results_ablation/{data_name}_both_n_cal{res}/ablation_{metric}.png
    """
    # Shared output folder
    shared_folder = os.path.join(ROOT, f"{data_name}_both")
    # if res:
    #     shared_folder += res
    os.makedirs(shared_folder, exist_ok=True)
    save_path = os.path.join(shared_folder, f"{data_name}_hyperparams-{fixed}.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    for model in model_list:
        if n_res is not None:
            results = pd.read_csv(os.path.join(ROOT, f"{data_name}_{model}_with-res-{n_res}/results_{fixed}.csv"))
        else:
            results = pd.read_csv(os.path.join(ROOT, f"{data_name}_{model}/results_{fixed}.csv"))
        # print("unique free values:", results[FIXED_VAR_INFO[fixed]["free"]].sort_values().unique())
        # exit()
        # Filter x values to the ones in FIXED_VAR_INFO
        x_values = FIXED_VAR_INFO[fixed][data_name]["x_values"]
        results = results[results[FIXED_VAR_INFO[fixed]["free"]].isin(x_values)]
        # print("After filtering, unique free values:", results[FIXED_VAR_INFO[
        fixed_value = results[args.fixed].mode().iloc[0]
        plot_ablation_single(ax, results, metric, label= model, fixed_value=fixed_value)

    ax.set_xlabel(pretty_name(FIXED_VAR_INFO[fixed]["free"]))
    ax.set_ylabel(pretty_name(metric))
    ax.grid(True)
    if fixed == "n_clusters":
        x_lim_low, x_lim_high = FIXED_VAR_INFO[fixed][data_name]["x_lim"]
        ax.set_xlim(x_lim_low, x_lim_high)  # y-axis starts at
    #     # ax.set_xlim(0, 60)  # y-axis starts at
    # y_lim_low, y_lim_high = FIXED_VAR_INFO[fixed][data_name]["y_lim"]
    # ax.set_ylim(y_lim_low, y_lim_high)  # y-axis starts at
    # ax.set_ylim(0.2, 0.7)  # y-axis starts at 0
    ax.legend( loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print("Saved plot to", save_path)

FIXED_VAR_INFO = {
    "temperature": {
        "fixed": "temperature",
        "free": "n_clusters",
        "cifar10": {
            "y_lim": (0.15, 0.5),
            "x_values": [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 64, 70, 74, 80, 84, 90,  94,  98,
                          105, 110, 120, 130, 140, 150, 175,
                         200, 225, 250, 275, 300]
        },
        "cifar100": {
            "y_lim": (0.4, 0.7),
            "x_values": [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 64, 70, 74, 80, 84, 90,  94,  98,
                          105, 110, 120, 130, 140, 150, 175,
                         200, 225, 250, 275, 300]
        },
        "imagenet": {
            # "y_lim": (0.2, 0.6),
            "x_values": [ 10, 30, 50, 70, 90, 110, 130, 150,  170,  190, 210,230, 250,
                          270,  290,  310,  330, 360, 400, 450, 500, 550, 600, 650, 700]
        }

    },
    "n_clusters": {
        "fixed": "n_clusters",
        "free": "temperature",
        "cifar10": {
            "y_lim": (0.15, 0.5),
            "x_lim": (0, 37),
            "x_values": [1.,  2.,  3.,  4.,  4.9, 
                         5.9, 7., 8., 9., 10., 12., 15., 17., 20.,
                         23., 25., 27., 30., 32., 35.]
        },
        "cifar100": {
            "y_lim": (0.35, 1),
            "x_lim": (0, 15.5),
            "x_values": [1.,  2.,   3.,   4.,    4.9, 
                         5.9,  7.,  8.,  9.,  10., 11,12, 13, 14, 15]
        },
        "imagenet": {
            # "y_lim": (0.2, 0.6),
            "x_lim": (0.6, 4.2),
            "x_values": [0.8, 0.9, 1.,  1.2,  1.4,  1.6, 
                         1.8, 2., 2.3, 2.6, 3., 3.3, 3.6, 4., 
                        #  4.3, 4.6, 5.,
                        #  10., 15.
                         ]
        }
    }
}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ablation results.")
    parser.add_argument("--metric", type=str, default="fpr", help="Metric to plot (e.g., accuracy, nll, ece).")
    parser.add_argument("--data_name", type=str, default="cifar10", help="Dataset name (e.g., cifar10, cifar100, imagenet).")
    parser.add_argument("--model_name", type=str, default="resnet34", help="Model name (e.g., resnet34, densenet121, timm-vit-tiny16).")
    parser.add_argument("--fixed_value", type=float, default=None, help="Fixed value for the plot.")
    parser.add_argument("--n_folds", type=int, default=10, help="Number of folds for cross-validation.")
    parser.add_argument("--n_res", default=None, help="Number of res samples.")
    parser.add_argument("--fixed", type=str, default="n_clusters", help="Number of res samples.")
    parser.add_argument("--ablation_type", type=str, default="hyperparams", help="Ablation type.")


    args = parser.parse_args()
    setup_publication_style()

    if args.ablation_type == "hyperparams":

        # ROOT = f"results_ablation_hyperparams/{args.data_name}_{args.model_name}"
        ROOT = "results_ablation_hyperparams"
        args.metric += "_test"

        if args.model_name == "both":
            model_list = LIST_MODELS[args.data_name]
        
            if not isinstance(model_list, (list, tuple)) or len(model_list) < 2:
                raise ValueError(f"LIST_MODELS[{args.data_name}] must contain at least two models to use 'both'. Got: {model_list}")

            plot_ablation_both(
                args.data_name, 
                model_list, 
                args.metric, 
                n_res=args.n_res, 
                fixed=args.fixed)
        else:
            # results, result_folder = read_table(args.data_name, args.model_name, res=args.res)
            # print("Result folder:", result_folder)
            # out_path = os.path.join(result_folder, f"ablation_{args.metric}.png")

            # plot_ablation(results, args.metric, out_path)

            plot_ablation(
                args.data_name, 
                args.model_name, 
                args.metric, 
                fixed=args.fixed, 
                xlabel=pretty_name(FIXED_VAR_INFO[args.fixed]["free"])
                )
    else:
        ROOT = f"./fair3_calib-{args.n_res}_ce_results/imagenet_{args.model_name}_r-0.5_seed-split-9/transform-test_n-epoch1_n-folds{args.n_folds}_probits_soft-kmeans_torch_bernstein_n-init-10_kmeans_clustering"
        file = "hyperparams_results_3"
    

        # results, result_folder = read_table(args.data_name, args.model_name, res=args.res)
        results = pd.read_csv(os.path.join(ROOT, f"{file}.csv"))
        # cols = ["clustering.n_clusters", "clustering.temperature", f"{args.metric}_val_cross", f"{args.metric}_val_cross_std"]
        # print(results[cols].sort_values(f"{args.metric}_val_cross").head(10))

        plot_metric_curve(results, fixed=args.fixed, metric="fpr", fixed_value=args.fixed_value)


