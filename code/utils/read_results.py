from .results_helper import read_table, RESULTS_FILES

METRICS = ("fpr_val", "roc_auc_val", "aurc_val", "aupr_err_val", "aupr_success_val")

def _first_col(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None

def _get_metrics(df):
    # Accept dotted or underscored variants
    col_map = {
        "fpr_val":          ("fpr_val",),
        "roc_auc_val":      ("roc_auc_val", "auroc_val"),
        "aurc_val":         ("aurc_val",),
        "aupr_err_val":     ("aupr_err_val", "aupr_error_val"),
        "aupr_success_val": ("aupr_success_val", "aupr_ok_val", "aupr_correct_val"),
    }
    out = {k: None for k in col_map}
    if df is None or len(df) == 0:
        return out
    row = df.iloc[0]
    for k, cands in col_map.items():
        c = next((nm for nm in cands if nm in df.columns), None)
        out[k] = float(row[c]) if c is not None else None
    return out

def _fmt(x, metric, decimals=1):
    if x is None:
        return "–"
    # if metric == "fpr_val":
    return f"{x*100:.{decimals}f}"
    # return f"{x:.{decimals}f}"

def build_table(results_files, bold=False, decimals=(1,1,1,1,1)):
    """
    Returns a nested dict: {(dataset, model): {method: 'FPR%/AUROC/AURC/AUPR-Err/AUPR-Succ'}}
    """
    import pandas as pd

    rows = []
    for dataset, models in results_files.items():
        for model, methods in models.items():
            for method, expe_folder in methods.items():
                try:
                    df = read_table(expe_folder=expe_folder, hyperparam=False)
                except Exception as e:
                    print(f"[WARN] skip {dataset}-{model}-{method}: {e}")
                    df = None
                metrics_vals = _get_metrics(df)
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "method": method,
                    **metrics_vals
                })

    if not rows:
        raise RuntimeError("No results found.")

    df = pd.DataFrame(rows)

    # Build cell text, optionally bold best per metric within (dataset, model)
    disp = {}
    for (ds, md), g in df.groupby(["dataset", "model"], sort=False):
        # compute best per metric if requested
        best_min = {"fpr_val", "aurc_val"}
        best_max = {"roc_auc_val", "aupr_err_val", "aupr_success_val"}
        best = {}
        if bold:
            for m in METRICS:
                series = g[m].dropna()
                if series.empty:
                    best[m] = None
                elif m in best_min:
                    best[m] = series.min()
                else:
                    best[m] = series.max()

        row = {}
        for method, g_m in g.groupby("method", sort=False):
            r = g_m.iloc[0]
            parts = []
            for m, d in zip(METRICS, decimals):
                s = _fmt(r.get(m, None), m, d)
                if bold and best.get(m, None) is not None and r.get(m, None) == best[m]:
                    s = f"**{s}**"
                parts.append(s)
            row[method] = "/".join(parts)
        disp[(ds, md)] = row
    return disp

# add these two helpers to your script:

def print_table_compact(df_out):
    """Pretty print compact view (one cell per method) with tabulate if available."""
    try:
        from tabulate import tabulate
        print(tabulate(df_out, headers="keys", tablefmt="github", showindex=False))
    except Exception:
        # Plain text fallback
        print(df_out.to_string(index=False))

def print_table_wide(nested):
    """
    Wide view: split metrics out so they align in separate columns.
    `nested` is the dict returned by build_table(...).
    """
    import pandas as pd
    # recover all methods
    methods = []
    for row in nested.values():
        for m in row.keys():
            if m not in methods:
                methods.append(m)

    # The compact cell is "FPR%/AUROC/AURC/AUPR-Err/AUPR-Succ".
    metric_labels = ["FPR%", "AUROC", "AURC", "AUPR-Err", "AUPR-Succ"]

    records = []
    for (ds, md), cols in nested.items():
        rec = {"Dataset": ds, "Model": md}
        for method in methods:
            cell = cols.get(method, "–")
            parts = cell.split("/")
            # pad or truncate to 5 parts
            parts = (parts + ["–"]*5)[:5]
            for label, val in zip(metric_labels, parts):
                rec[f"{method}::{label}"] = val
        records.append(rec)
    wide_df = pd.DataFrame(records)

    # try tabulate for aligned output
    try:
        from tabulate import tabulate
        print(tabulate(wide_df, headers="keys", tablefmt="github", showindex=False))
    except Exception:
        print(wide_df.to_string(index=False))


def print_markdown_table(nested):
    import pandas as pd
    from packaging.version import Version

    # Collect consistent method columns
    methods = []
    for row in nested.values():
        for m in row.keys():
            if m not in methods:
                methods.append(m)

    records = []
    for (ds, md), cols in nested.items():
        rec = {"Dataset": ds, "Model": md}
        rec.update({m: cols.get(m, "–") for m in methods})
        records.append(rec)
    df_out = pd.DataFrame(records)

    # Try markdown via pandas (requires tabulate>=0.9.0)
    try:
        import tabulate  # noqa: F401
        # If you want to be strict about the version:
        try:
            from importlib.metadata import version
            if Version(version("tabulate")) < Version("0.9.0"):
                raise ImportError("tabulate<0.9.0")
        except Exception:
            # if version check fails, just try to_markdown and let it raise if incompatible
            pass

        print(df_out.to_markdown(index=False))  # pretty Markdown table
        return
    except Exception:
        # Fallback: decent plain-text table (no extra deps)
        # You can also switch to CSV by using df_out.to_csv(index=False)
        print(df_out.to_string(index=False))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Print summary table of results.")
    parser.add_argument("--bold", action="store_true", default=False,
                        help="Bold best per metric within each (dataset, model).")
    parser.add_argument("--dec", type=int, nargs=5, default=[1,1,1,1,1],
                        help="Decimals for (FPR%%, AUROC, AURC, AUPR-Err, AUPR-Succ).")
    parser.add_argument("--view", choices=["compact","wide"], default="wide",
                        help="compact = one cell/method; wide = separate metric columns.")
    args = parser.parse_args()

    table = build_table(RESULTS_FILES, bold=args.bold, decimals=tuple(args.dec))
    print_markdown_table(table)
    # if args.view == "compact":
    #     # turn the nested dict into a compact DataFrame
    #     import pandas as pd
    #     methods = []
    #     for row in table.values():
    #         for m in row.keys():
    #             if m not in methods:
    #                 methods.append(m)
    #     rows = []
    #     for (ds, md), cols in table.items():
    #         r = {"Dataset": ds, "Model": md}
    #         r.update({m: cols.get(m, "–") for m in methods})
    #         rows.append(r)
    #     df_out = pd.DataFrame(rows)
    #     print_table_compact(df_out)
    # else:
    #     # wide, easier to read aligned numbers
    #     print_table_wide(table)
