import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import pandas as pd
from .results_helper import read_table

# def read_table(
#     data_name=None,
#     model_name=None,
#     method_name=None,
#     r=None,
#     seed_split=None,
#     n_epochs=None,
#     transform=None,
#     hyperparam=True,
#     expe_folder=None,
#     n_folds=None,
#     space=None,
#     distance=None,
#     clustering_name=None,
#     normalize_gini=None,
#     num=None
#     ):

#     if expe_folder is None:

#         root = f"results/{data_name}_{model_name}_r-{r}_seed-split-{seed_split}"
#         if method_name in ["clustering", "metric_learning", "random_forest"]:
#             root += f"/transform-{transform}_n-epoch{n_epochs}_n-folds{n_folds}_{space}"
#             if method_name == "clustering":
#                 if distance is not None:
#                     root += f"_distance-{distance}"
#                 if clustering_name != "soft-kmeans":
#                     root += f"_{clustering_name}"
#                 # root += f"_probweights"
#         elif method_name == "gini":
#             root += f"/transform-{transform}_normalize-{normalize_gini}"
#         else:
#             root += f"/transform-{transform}"
#         expe_folder = root + f"_{method_name}"

    
#     if hyperparam:
#         expe_file = os.path.join(expe_folder, "hyperparams_results.csv")
#     #     df = pd.read_csv(os.path.join(expe_folder, "hyperparams_results.csv"))
#     else:
#         expe_file = os.path.join(expe_folder, "results_opt_fpr.csv")
#         # df = pd.read_csv(os.path.join(expe_folder, "all_results.csv"))
#     if num is not None:
#         expe_file = expe_file.replace(".csv", f"_{num}.csv")
#     df = pd.read_csv(expe_file)
#     return df
import os
import re
import glob
import pandas as pd



# def smooth_curve(x, y, window_size=3):
#     """Apply smoothing to reduce noise in curves"""
#     if len(y) < window_size:
#         return x, y
    
#     # Simple moving average
#     smoothed_y = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
#     smoothed_x = x[window_size-1:]
    
#     return smoothed_x, smoothed_y

# def plot_with_confidence_bands(x, y, std, label, color, alpha_fill=0.15, smooth=True):
#     """Plot curves with confidence bands and optional smoothing"""
    
#     if smooth and len(x) > 3:
#         # Smooth the main curve
#         x_smooth, y_smooth = smooth_curve(x, y)
#         # Also smooth the confidence bands
#         _, upper_smooth = smooth_curve(x, y + std)
#         _, lower_smooth = smooth_curve(x, y - std)
        
#         # Plot smoothed curve
#         plt.plot(x_smooth, y_smooth, label=label, color=color, linewidth=2.5, 
#                 marker='o', markersize=6, markerfacecolor='white', markeredgecolor=color, 
#                 markeredgewidth=2)
        
#         # Add confidence band
#         plt.fill_between(x_smooth, lower_smooth, upper_smooth, 
#                         color=color, alpha=alpha_fill, interpolate=True)
        
#         # Add error bars at original points for clarity
#         plt.errorbar(x, y, yerr=std, fmt='none', color=color, alpha=0.7, 
#                     capsize=3, capthick=1.5, elinewidth=1.5)
#     else:
#         # Plot original curve with error bars
#         plt.errorbar(x, y, yerr=std, label=label, color=color, 
#                     linewidth=2.5, marker='o', markersize=6, capsize=3,
#                     markerfacecolor='white', markeredgecolor=color, markeredgewidth=2)
        
#         # Add confidence band
#         plt.fill_between(x, y - std, y + std, color=color, alpha=alpha_fill)

# def main():
#     parser = argparse.ArgumentParser(description='Create publication-quality unlearning evaluation plots')
#     parser.add_argument('--cfg_file', default='configs/larger_bs_no_weight_decay_2_layers_spectral_norm_waterbird.yaml')
#     parser.add_argument('--smooth', action='store_true', help='Apply smoothing to curves')
#     parser.add_argument('--style', choices=['default', 'seaborn', 'bmh'], default='default',
#                        help='Plot style theme')
#     args = parser.parse_args()

#     # Set up publication style
#     setup_publication_style()
    
#     # Apply additional style if requested
#     if args.style == 'seaborn':
#         sns.set_style("whitegrid")
#         sns.set_palette("husl")
#     elif args.style == 'bmh':
#         plt.style.use('bmh')

#     with open(args.cfg_file, 'r') as f:
#         config = yaml.safe_load(f)
    
#     # Create figure with better proportions
#     fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
#     n_points = [1000,2000,4000, 4795]
#     checkpoints = [1, 5, 10, 15,20]
#     same_fraction = False
#     lr = config['training']['lr']
#     l = config['training']['n_layers']
#     spectral_norm = config['training']['spectral_norm']
#     plot_estimates = False
#     layer_norm = config['training']['layer_norm']
#     add = config['set_up']['suffix']
    
#     if same_fraction:
#         n_points = [10000, 15000, 20000]
    
#     data_path = os.path.join('results', add)
#     print(f"Data path: {data_path}")
    
#     # Define a professional color palette
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
#     if plot_estimates:
#         # Estimates plotting section (improved)
#         for i, n in enumerate(n_points):
#             try:
#                 estimates_path = os.path.join(data_path, f'{n}public_points/test_spectral_norm{spectral_norm}layer_norm{layer_norm}estimates_checkpoint_0_lr{lr}len{l}.npy')
#                 std_path = os.path.join(data_path, f'{n}public_points/test_spectral_norm{spectral_norm}layer_norm{layer_norm}estimates_std_checkpoint_0_lr{lr}len{l}.npy')
                
#                 estimates = np.load(estimates_path)
#                 std = np.load(std_path)
#                 x_vals = np.arange(len(estimates))
#                 n_private_points = 22000 - n
#                 plot_with_confidence_bands(x_vals, estimates, std, 
#                                          f'{n//1000}K public points, {n_private_points//1000}K private points', 
#                                          colors[i % len(colors)], 
#                                          smooth=args.smooth)
                
#             except FileNotFoundError as e:
#                 print(f"Warning: Could not load data for {n} points: {e}")
#                 continue
        
#         plt.xlabel('Iteration', fontweight='bold')
#         plt.ylabel('Renyi Divergence Estimate', fontweight='bold')
#         plt.title('Renyi Divergence Estimation During Unlearning', fontweight='bold', pad=20)
        
#     else:
#         # Main results plotting (improved)
#         max_renyi = -np.inf
#         min_renyi = np.inf
        
#         for i, n in enumerate(n_points):
#             print(f'Processing {n} public points...')
#             renyi_vals = []
#             stds = []
#             base_path = os.path.join(data_path, f'{n}_public_points/')
#             actual_checkpoints = []
            
#             if os.path.exists(base_path):
#                 for checkpoint in checkpoints:
#                     renyi_path = os.path.join(base_path, f'test_spectral_norm_{spectral_norm}layer_norm{layer_norm}renyi_result_checkpoint{checkpoint}lr{lr}len{l}.npy')
#                     last_estimate_path = os.path.join(base_path,f'test_spectral_norm_{spectral_norm}layer_norm{layer_norm}all_estimates_checkpoint{checkpoint}lr{lr}len{l}.npy')
#                     if os.path.exists(renyi_path):
#                         try:
#                             actual_checkpoints.append(checkpoint)
#                             renyi_val = np.load(renyi_path, allow_pickle=True).item()
#                             renyi_valu = np.load(last_estimate_path, allow_pickle=True)[-1].mean().item()
#                             #renyi_vals.append(renyi_val['renyi_val'])
#                             renyi_vals.append(renyi_valu)
#                             stds.append(renyi_val['std'])
#                             #stds.append(0)  # Placeholder if std not available
#                             #print(f'  Checkpoint {checkpoint}: Renyi = {renyi_val["renyi_val"]:.4f} ± {renyi_val["std"]:.4f}')
#                             print(f'  Checkpoint {checkpoint}: Renyi = {renyi_valu:.4f} ± {renyi_val["std"]:.4f}')
#                         except Exception as e:
#                             print(f"  Warning: Could not load checkpoint {checkpoint}: {e}")
#                             continue
                
#                 if len(renyi_vals) > 0:
#                     renyi_vals = np.array(renyi_vals)
#                     stds = np.array(stds)
#                     actual_checkpoints = np.array(actual_checkpoints)
                    
#                     # Track min/max for y-axis scaling
#                     max_renyi = max(max_renyi, np.max(renyi_vals + stds))
#                     min_renyi = min(min_renyi, np.min(renyi_vals - stds))
#                     n_private_points = 22000 - n
#                     # Plot with improved styling
#                     plot_with_confidence_bands(actual_checkpoints, renyi_vals, stds,
#                                              f'{n//1000}K public points, {n_private_points//1000}K private points',
#                                              colors[i % len(colors)],
#                                              smooth=args.smooth)
#                 else:
#                     print(f'  No valid data found for {n} public points')
#             else:
#                 print(f'  Path does not exist: {base_path}')
        
#         # Customize axes and labels
#         plt.xlabel('Unlearning Iteration (K)', fontweight='bold')
#         plt.ylabel(r'$\widehat{D_{\alpha}(\pi_R^{T+K}\|\pi_U^{K})}$', fontweight='bold')

#         # Improve y-axis scaling
#         if max_renyi > min_renyi:
#             y_margin = (max_renyi - min_renyi) * 0.1
#             plt.ylim(min_renyi - y_margin, max_renyi + y_margin)
    
#     # Improve legend
#     legend = plt.legend(loc='best', fancybox=True, shadow=True, ncol=1)
#     legend.get_frame().set_alpha(0.9)
    
#     # Add subtle grid
#     plt.grid(True, alpha=0.3, linestyle='--')
    
#     # Tight layout with padding
#     plt.tight_layout(pad=2.0)
    
#     # Save with high DPI for publication
#     output_path = os.path.join(data_path, 'renyi_estimation_publication.pdf')
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', 
#                facecolor='white', edgecolor='none')
#     plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight',
#                facecolor='white', edgecolor='none')
    
#     print(f"Publication-quality plot saved to: {output_path}")
#     plt.show()




def plot_metric_curve(
    df,
    metric: str,
    fixed: str = "temperature",            # "temperature" or "n_clusters"
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
        g = df_clean.groupby(fixed_col, as_index=False)[mean_col].mean()
        if g.empty:
            raise ValueError("No data to choose a best fixed value.")
        fixed_value = g.sort_values(mean_col, ascending=ascending).iloc[0][fixed_col]

    # slice by fixed value (with tolerance for float temperatures)
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

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    line, = ax.plot(x, y, marker="o", label=label)
    ax.fill_between(x, y - s, y + s, alpha=0.20, linewidth=0)

    # if title_prefix is None:
    #     title_prefix = f"{metric.upper()} vs {free_name}"
    # ax.set_title(title_prefix)
    ax.set_xlabel(free_name)
    ax.set_ylabel(pretty_name(mean_col))
    ax.grid(True)

    return (float(fixed_value) if isinstance(fixed_value, (int,float,np.floating)) else fixed_value), ax
import re

def pretty_name(name: str) -> str:
    """
    Map raw keys (models or metrics) to nice display names.
    - Keeps underscores so metric keys like 'fpr_val_cross' match.
    - Still strips spaces and hyphens; lowercases for robust lookup.
    """
    key = name.strip().lower().replace(" ", "").replace("-", "")  # NOTE: no .replace("_","")

    table = {
        # --- Models ---
        "resnet34": "ResNet-34",
        "densenet121": "DenseNet-121",
        "densenet": "DenseNet",
        "timm_vit_base16": "ViT-B/16",
        "timm_vit_tiny16": "ViT-Tiny/16",
        "vit_tiny16": "ViT-Tiny/16",
        "vitbase16": "ViT-B/16",
        "vit_b16": "ViT-B/16",
        "vittiny": "ViT-Tiny/16",           # fallback if patch size omitted
        "vitbase": "ViT-B/16",
        "mlpsynthdim10classes7": "MLP (Synth d=10, C=7)",

        # --- Metrics (val_cross variants etc.) ---
        "aupr_val_cross": "AUPR",
        "auroc_val_cross": "AUROC",
        "fpr_val_cross": "FPR@95",
        "aurc_val_cross": "AURC",
        "aupr_err_val_cross": "AUPR-Err",
        "aupr_success_val_cross": "AUPR-Succ",
        # common alternates (optional)
        "roc_auc_val": "AUROC",
        "aurc_val": "AURC",
        "fpr_val": "FPR@95",
        "aupr_err_val": "AUPR-Err",
        "aupr_success_val": "AUPR-Succ",
    }

    if key in table:
        return table[key]

    # Generic ViT pattern: vit[-_]?SIZE[-_]?PATCH -> ViT-{B/L/H/...}/PATCH
    m = re.match(r"^vit[-_]?(\w+?)(?:[-_]?)(\d+)?$", key)
    if m:
        size, patch = m.groups()
        size_map = {"tiny":"Tiny","tsmall":"Tiny","small":"Small","base":"B","b":"B",
                    "large":"L","l":"L","huge":"H","h":"H"}
        tag = size_map.get(size, size.capitalize())
        return f"ViT-{tag}/{patch or '16'}"

    # Fallback: title-case, fix common names
    return (name.replace("_", " ").replace("-", " ")
                .title()
                .replace("Vit", "ViT")
                .replace("Resnet", "ResNet")
                .replace("Densenet", "DenseNet"))

# Backward-compatible alias if you already call pretty_model_name(...)



if __name__ == "__main__":
    import argparse
    import os
    # import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Plot metric vs free hyperparameter.")
    # --- experiment args (unchanged) ---
    parser.add_argument("--data_name", default="cifar10")
    parser.add_argument("--models", nargs="+", default=["resnet34", "densenet121"])
    parser.add_argument("--method_name", default="clustering")
    parser.add_argument("--r", type=float, default=2)
    parser.add_argument("--seed_split", type=int, default=9)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--transform", default="test")
    parser.add_argument("--space", default="probits")
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--clustering_name", default="soft-kmeans")
    parser.add_argument("--distance", default=None)
    parser.add_argument("--normalize_gini", default=None)
    parser.add_argument("--expe_folder", default=None)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--hyperparam", action="store_true", default=True)
    parser.add_argument("--no-hyperparam", dest="hyperparam", action="store_false")

    # --- plotting/selection args (unchanged) ---
    parser.add_argument("--metric", default="fpr_val_cross")
    parser.add_argument("--fixed", choices=["temperature", "n_clusters"], default="temperature")
    parser.add_argument("--fixed_value", type=float, default=None)
    parser.add_argument("--min_value", type=float, default=None)
    parser.add_argument("--max_value", type=float, default=None)
    parser.add_argument("--ascending", type=str, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--legend_loc", default="best")
    parser.add_argument("--fix_tol", type=float, default=1e-10)
    parser.add_argument("--print_duplicates", action="store_true", default=False)
    # NEW: extra args needed by the ImageNet path template
    parser.add_argument("--bound", default="bernstein")        # e.g., ce / other tag you use
    parser.add_argument("--n_inits", type=int, default=1)
    parser.add_argument("--init", default="kmeans")   # e.g., kmeanspp / random

    # args = parser.parse_args()


    # --- NEW: output controls ---
    parser.add_argument("--out_dir", default="images",
                        help="Directory to save the figure (default: images).")
    parser.add_argument("--save", default=None,
                        help="Explicit output path (overrides --out_dir and auto-naming).")
    parser.add_argument("--dpi", type=int, default=150)

    args = parser.parse_args()

    setup_publication_style()  
    # normalize --ascending if provided
    asc = None
    if args.ascending is not None:
        s = args.ascending.strip().lower()
        if s in {"true", "t", "1", "yes", "y"}:
            asc = True
        elif s in {"false", "f", "0", "no", "n"}:
            asc = False
        else:
            raise ValueError("--ascending must be true/false")

    # --- load all requested models ---
    dfs = {}
    for m in args.models:
        
        # Default: let read_table build its own folder (non-ImageNet)
        expe_folder = args.expe_folder

        # SPECIAL CASE: ImageNet → build expe_folder using your template
        if args.data_name.lower().startswith(("imagenet", "imagenet1k", "imagenet-1k")):
            if m == "timm_vit_tiny16":
                args.init = "k-means++"
            if m == "timm_vit_base16":
                args.init = "kmeans" 
            root = f"test_ce_results/{args.data_name}_{m}_r-{args.r}_seed-split-{args.seed_split}"
            expe_folder = (
                root
                + f"/transform-{args.transform}"
                f"_n-epoch{args.n_epochs}"
                f"_n-folds{args.n_folds}"
                f"_{args.space}"
                f"_{args.clustering_name}"
                f"_{args.bound}"
                f"_n-init-{args.n_inits}"
                f"_{args.init}"
                f"_clustering"
            )

        df_m = read_table(
            data_name=args.data_name,
            model_name=m,
            method_name=args.method_name,
            expe_folder=expe_folder,   # <- pass the computed folder (for ImageNet)
            num=args.num,
            hyperparam=args.hyperparam,
        )
        dfs[m] = df_m
        print(f"Loaded {len(df_m)} rows for model '{m}'.")
    # --- plot on one axes ---
    fig, ax = plt.subplots(figsize=(8, 6))
    chosen_fixed_values = {}

    for m, df_m in dfs.items():
        label = pretty_name(m)
        best_val, _ = plot_metric_curve(
            df=df_m,
            metric=args.metric,
            fixed=args.fixed,
            fixed_value=args.fixed_value,     # None => choose best per model
            min_value=args.min_value,
            max_value=args.max_value,
            ascending=asc,
            ax=ax,
            # label=m,
            label=label,
            fix_tol=args.fix_tol,
            print_duplicates=args.print_duplicates,
        )
        chosen_fixed_values[m] = best_val

    # Title & legend
    # if args.title:
    #     ax.set_title(args.title)
    # else:
    #     free_name = "Number of clusters" if args.fixed == "temperature" else "Temperature"
    #     ax.set_title(f"{args.metric.upper()} vs {free_name}")
    ax.legend(loc=args.legend_loc)

    # --- NEW: auto output path if --save not provided ---
    if args.save is None:
        os.makedirs(args.out_dir, exist_ok=True)
        models_slug = "-".join(args.models)
        fixed_slug = args.fixed if args.fixed_value is None else f"{args.fixed}-{args.fixed_value:g}"
        auto_name = f"{args.data_name}_{models_slug}_{args.metric}_fixed-{fixed_slug}.png"
        out_path = os.path.join(args.out_dir, auto_name)
    else:
        out_path = args.save
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure to {out_path}")
