# import os 
# import matplotlib.pyplot as plt
# import numpy as np
# import argparse
# import pandas as pd
# from code.utils.results_helper import setup_publication_style, pretty_name, LIST_MODELS, LIST_DATASETS


# ROOT = "./results_ablation"
# def plot_ablation(results, metric, save_path):
#     results[metric] = results[metric].apply(lambda x: float(x.strip("[]")))
#     # print(results[metric])
#     agg = (
#         results
#         .groupby("n_cal", dropna=False)[metric]
#         .agg(["mean", "std", "count"])
#         .round(4)
#     )

#     # Example: sort by smallest mean overfitting
    
#     plt.figure(figsize=(8, 6))
#     plt.plot(agg.index, agg["mean"], marker='o', label='Mean')
#     plt.fill_between(agg.index, agg["mean"] - agg["std"], agg["mean"] + agg["std"], color='b', alpha=0.2, label='Std Dev')
#     plt.xlabel(r'$|\mathcal{D}_{cal}|$')
#     plt.ylabel(pretty_name(metric))
#     plt.grid(True)
#     plt.savefig(save_path)
#     plt.show()
#     print("Saved plot to", save_path)
#     return

# def read_table(data_name, model_name, res):

#     result_folder = os.path.join(ROOT, f"{args.data_name}_{args.model_name}_n_cal")
#     if res != "":
#         result_folder += res
#     result_file = os.path.join(result_folder, "results.csv")
#     results = pd.read_csv(result_file)
#     return results, result_folder


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Plot ablation results.")
#     parser.add_argument("--metric", type=str, default="fpr_test", help="Metric to plot (e.g., accuracy, nll, ece).")
#     parser.add_argument("--data_name", type=str, default="cifar10", help="Dataset name (e.g., cifar10, cifar100, imagenet).")
#     parser.add_argument("--model_name", type=str, default="resnet34", help="Model name (e.g., resnet34, densenet121, timm-vit-tiny16).")
#     parser.add_argument("--res", type=str, default="", help="Whether to include res samples in the plot.")
#     args = parser.parse_args()

#     setup_publication_style()  
#     if args.model_name == "both":
#         for model in LIST_MODELS[args.data_name]:
#     results, result_folder = read_table(args.data_name, args.model_name, res=args.res)
#     print("Result folder:", result_folder)
#     plot_ablation(results, args.metric, os.path.join(result_folder, f"ablation_{args.metric}.png"))
    


import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
from code.utils.results_helper import setup_publication_style, pretty_name, LIST_MODELS, LIST_DATASETS

# Try to get a pretty model-name mapper if available; otherwise fall back.
try:
    from code.utils.results_helper import pretty_model_name as _pretty_model_name
except Exception:
    def _pretty_model_name(x: str) -> str:
        return x

ROOT = "./results_ablation"

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

def read_table(data_name: str, model_name: str, res: str, method: str="clustering"):
    # if method == "clustering":
        
    result_folder = os.path.join(ROOT, method, f"{data_name}_{model_name}_n_cal")
    # else:
    #     result_folder = os.path.join(ROOT, method, f"{data_name}_{model_name}_n_cal")
    if res:
        result_folder += res
    result_file = os.path.join(result_folder, "results.csv")
    results = pd.read_csv(result_file)
    return results, result_folder

def plot_ablation_single(ax, results: pd.DataFrame, metric: str, label: str = None):
    agg = aggregate_results(results, metric)
    x = agg.index.values
    y = agg["mean"].values
    s = agg["std"].values

    line, = ax.plot(x, y, marker='o', label=pretty_name(label) if label else 'Mean')
    ax.fill_between(x, y - s, y + s, alpha=0.2)  # uses same color automatically
    return line

def plot_ablation(results, metric, save_path, xlabel=r'$|\mathcal{D}_{\mathrm{cal}}|$'):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    plot_ablation_single(ax, results, metric)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(pretty_name(metric))
    ax.set_ylim(0.2, 0.7)  # y-axis starts at 0
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print("Saved plot to", save_path)

def plot_ablation_both(data_name: str, model_list, metric: str, res: str, method: str="clustering"):
    """
    Plot both models on a single figure with legend.
    Saves into a shared folder: ./results_ablation/{data_name}_both_n_cal{res}/ablation_{metric}.png
    """
    # Shared output folder
    # if method != "clustering":
    #     shared_folder = os.path.join(ROOT, method, f"{data_name}_both_n_cal")
    # else:
    shared_folder = os.path.join(ROOT, method, f"{data_name}_both_n_cal")
    if res:
        shared_folder += res
    os.makedirs(shared_folder, exist_ok=True)
    save_path = os.path.join(shared_folder, f"{args.data_name}_calibration_size.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    for model in model_list:
        results, _ = read_table(data_name, model, res=res, method=method)
        plot_ablation_single(ax, results, metric, label=_pretty_model_name(model))

    ax.set_xlabel(r'$|\mathcal{D}_{\mathrm{cal}}|$')
    ax.set_ylabel(pretty_name(metric))
    ax.grid(True)
    y_lim_low, y_lim_high = INFO[data_name]["y_lim"]
    ax.set_ylim(y_lim_low, y_lim_high)  # y-axis starts at
    # ax.set_ylim(0.2, 0.7)  # y-axis starts at 0
    ax.legend(title="Model", loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print("Saved plot to", save_path)

INFO = {
    "imagenet": {
        "y_lim": (0.43, 0.52),
    },
    "cifar100": {
        "y_lim": (0.4, 0.6),
    },
    "cifar10": {
        "y_lim": (0.2, 0.55),
    }
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ablation results.")
    parser.add_argument("--metric", type=str, default="fpr_test",
                        help="Metric to plot (e.g., fpr_test, accuracy, nll, ece).")
    parser.add_argument("--data_name", type=str, default="cifar10",
                        help="Dataset name (e.g., cifar10, cifar100, imagenet).")
    parser.add_argument("--model_name", type=str, default="resnet34",
                        help='Model name (e.g., resnet34, densenet121, timm-vit-tiny16), or "both".')
    parser.add_argument("--res", type=str, default="",
                        help="Optional suffix for result folder (e.g., '_with_res').")
    parser.add_argument("--method", type=str, default="clustering",
                        help="Method to use (e.g., 'clustering', 'relu').")
    args = parser.parse_args()

    # Sanity on dataset / models
    if args.data_name not in LIST_MODELS:
        raise ValueError(f"Unknown data_name '{args.data_name}'. Known: {list(LIST_MODELS.keys())}")

    setup_publication_style()

    if args.model_name == "both":
        model_list = LIST_MODELS[args.data_name]
        if not isinstance(model_list, (list, tuple)) or len(model_list) < 2:
            raise ValueError(f"LIST_MODELS[{args.data_name}] must contain at least two models to use 'both'. Got: {model_list}")
        plot_ablation_both(args.data_name, model_list, args.metric, res=args.res, method=args.method)
    else:
        results, result_folder = read_table(args.data_name, args.model_name, res=args.res, method=args.method)
        print("Result folder:", result_folder)
        out_path = os.path.join(result_folder, f"ablation_{args.metric}.png")
        plot_ablation(results, args.metric, out_path)
