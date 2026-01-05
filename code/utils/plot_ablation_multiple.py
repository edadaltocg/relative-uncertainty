import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from code.utils.results_helper import (
    setup_publication_style,
    pretty_name,
    LIST_MODELS,
)

try:
    from code.utils.results_helper import pretty_model_name as _pretty_model_name
except Exception:
    def _pretty_model_name(x: str) -> str:
        return x

ROOT = "./results_ablation"

def _coerce_metric_column(series: pd.Series) -> pd.Series:
    def _to_float(x):
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        if isinstance(x, str):
            s = x.strip()
            if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
                s = s[1:-1].strip()
            try:
                return float(s)
            except Exception:
                return np.nan
        return np.nan
    return series.apply(_to_float)

def aggregate_results(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not in columns: {list(df.columns)}")
    df = df.copy()
    df[metric] = _coerce_metric_column(df[metric])
    df = df.dropna(subset=[metric])
    agg = (
        df.groupby("n_cal", dropna=False)[metric]
          .agg(["mean", "std", "count"])
          .fillna({"std": 0.0})
          .sort_index()
    )
    return agg

def read_table(data_name: str, model_name: str, method: str, res_suffix: str = ""):
    if method == "clustering":
        folder = os.path.join(ROOT, f"{data_name}_{model_name}_n_cal")
    else:
        folder = os.path.join(ROOT, method, f"{data_name}_{model_name}_n_cal")
    if res_suffix:
        folder += res_suffix
    csv_path = os.path.join(folder, "results.csv")
    results = pd.read_csv(csv_path)
    return results, folder

def plot_methods_and_models(
    data_name: str,
    model_list,
    metric: str = "fpr_test",
    res_suffix: str = "",
    xlabel: str = r"$|\mathcal{D}_{\mathrm{cal}}|$",
    ylim: tuple | None = None,
    save_name: str | None = None,
):
    methods = ["clustering", "relu"]
    linestyle_map = {"clustering": "-", "relu": "--"}
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]  # per-model markers

    # Load once
    loaded = {}  # (model, method) -> agg df
    any_folder = None
    for model in model_list:
        for method in methods:
            df, folder = read_table(data_name, model, method, res_suffix)
            loaded[(model, method)] = aggregate_results(df, metric)
            any_folder = any_folder or folder

    # Output
    out_dir = os.path.join(ROOT, f"{data_name}_multi_models_methods")
    if res_suffix:
        out_dir += res_suffix
    os.makedirs(out_dir, exist_ok=True)
    if save_name is None:
        save_name = f"{data_name}_{metric}_models_{len(model_list)}_clustering_vs_relu.png"
    save_path = os.path.join(out_dir, save_name)

    setup_publication_style()
    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    # Assign one color per model (and reuse it for both methods)
    # We draw a dummy line per model to consume a color from the cycler, record it, then remove it.
    model_color = {}
    tmp_lines = []
    for _ in model_list:
        tmp_lines.append(ax.plot([], [])[0])
    for model, line in zip(model_list, tmp_lines):
        model_color[model] = line.get_color()
        line.remove()

    # Plot: color by model, linestyle by method, marker by model
    for mi, model in enumerate(model_list):
        color = model_color[model]
        marker = markers[mi % len(markers)]
        for method in methods:
            agg = loaded[(model, method)]
            x = agg.index.values
            y = agg["mean"].values
            s = agg["std"].values

            label = f"{_pretty_model_name(model)} · {method}"
            line, = ax.plot(
                x, y,
                marker=marker,
                linestyle=linestyle_map[method],
                label=label,
                color=color,
            )
            ax.fill_between(x, y - s, y + s, alpha=0.18, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(pretty_name(metric))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend(title="Model · Method", loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=240)
    plt.show()
    print(f"Saved plot to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot both methods (clustering, relu) for multiple models on one plot.")
    parser.add_argument("--data_name", type=str, default="cifar10",
                        help=f"Dataset key (one of {list(LIST_MODELS.keys())}).")
    parser.add_argument("--model_name", type=str, default="both",
                        help='Either a specific model name, or "both" to use LIST_MODELS[data_name].')
    parser.add_argument("--metric", type=str, default="fpr_test",
                        help="Metric column to plot (e.g., fpr_test, aurc, ece).")
    parser.add_argument("--res", type=str, default="",
                        help="Optional suffix for result folder names (e.g., '_with_res').")
    parser.add_argument("--ylim", type=float, nargs=2, default=None,
                        help="Optional y-limits, e.g. --ylim 0.2 0.7")
    parser.add_argument("--save_name", type=str, default=None,
                        help="Optional custom output filename.")
    args = parser.parse_args()

    if args.data_name not in LIST_MODELS:
        raise ValueError(f"Unknown data_name '{args.data_name}'. Known: {list(LIST_MODELS.keys())}")

    if args.model_name == "both":
        model_list = LIST_MODELS[args.data_name]
        if not isinstance(model_list, (list, tuple)) or len(model_list) < 2:
            raise ValueError(f"LIST_MODELS[{args.data_name}] must contain at least two models. Got: {model_list}")
    else:
        model_list = [args.model_name]

    plot_methods_and_models(
        data_name=args.data_name,
        model_list=model_list,
        metric=args.metric,
        res_suffix=args.res,
        ylim=tuple(args.ylim) if args.ylim is not None else None,
        save_name=args.save_name,
    )

if __name__ == "__main__":
    main()
