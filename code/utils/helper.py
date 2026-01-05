from itertools import product
from code.utils import set_nested
from copy import deepcopy
from time import strftime, localtime
import os
import pandas as pd

import re
import glob

import torch
import random
import numpy as np

import copy 
def make_grid(cfg_detection, key ="ablation_args"):
    base = deepcopy(cfg_detection["postprocessor_args"])
    keys = cfg_detection[key].keys()
    for vals in product(*cfg_detection[key].values()):
        cfg = deepcopy(base)
        cfg.update(zip(keys, vals))
        yield cfg

def metric_direction(metric: str) -> str:

    dic = {
        "fpr": "min",
        "roc_auc": "max",
        "aurc": "min",
        "aupr_err": "max",
        "aupr_success": "max",
    }
    if metric not in dic:
        raise ValueError(f"Unknown metric '{metric}'")
    return dic[metric]


def setup_seeds(seed: int, seed_split: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): The seed to use.
    """
    random.seed(seed_split)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_table(
    data_name=None,
    model_name=None,
    method_name=None,
    r=None,
    seed_split=None,
    n_epochs=None,
    transform=None,
    hyperparam=True,
    expe_folder=None,
    n_folds=None,
    space=None,
    distance=None,
    clustering_name=None,
    normalize_gini=None,
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
        if any(v is None for v in [data_name, model_name, method_name, r, seed_split]):
            raise ValueError("When expe_folder is None, provide data_name, model_name, method_name, r, seed_split.")
        root = f"../../results/{data_name}_{model_name}_r-{r}_seed-split-{seed_split}"
        if method_name in ["clustering", "metric_learning", "random_forest"]:
            if any(v is None for v in [transform, n_epochs, n_folds, space]):
                raise ValueError("transform, n_epochs, n_folds, space required for these methods when expe_folder is None.")
            root += f"/transform-{transform}_n-epoch{n_epochs}_n-folds{n_folds}_{space}"
            if method_name == "clustering":
                if distance is not None:
                    root += f"_distance-{distance}"
                if clustering_name != "soft-kmeans":
                    root += f"_{clustering_name}"
        elif method_name == "gini":
            if any(v is None for v in [transform, normalize_gini]):
                raise ValueError("transform and normalize_gini required for method 'gini' when expe_folder is None.")
            root += f"/transform-{transform}_normalize-{normalize_gini}"
        else:
            if transform is None:
                raise ValueError("transform is required when expe_folder is None.")
            root += f"/transform-{transform}"
        expe_folder = root + f"_{method_name}"

    base_name = "hyperparams_results" if hyperparam else "results_opt_fpr"
    rx = re.compile(rf"{re.escape(base_name)}(?:_(\d+))?\.csv$")

    # If `num` is specified, read that specific file only (annotated).
    if num is not None:
        # Allow num==0 or '' to mean base file (no suffix)
        fname = f"{base_name}.csv" if str(num) in ("", "0") else f"{base_name}_{num}.csv"
        target = os.path.join(expe_folder, fname)
        if not os.path.isfile(target):
            raise FileNotFoundError(f"No such file: {target}")
        df = pd.read_csv(target)
        # annotate
        m = rx.search(os.path.basename(target))
        file_num = int(m.group(1)) if (m and m.group(1) is not None) else 0
        df = df.copy()
        df["file_num"] = file_num
        df["source_file"] = os.path.basename(target)
        return df

    # Otherwise, gather *all* matching files.
    pattern = os.path.join(expe_folder, f"{base_name}*.csv")
    files = [f for f in glob.glob(pattern) if rx.search(os.path.basename(f))]

    if not files:
        raise FileNotFoundError(f"No matching files found with pattern: {pattern}")

    def sort_key(p):
        m = rx.search(os.path.basename(p))
        idx = m.group(1)
        return (0 if idx is None else 1, int(idx) if idx is not None else -1)

    files.sort(key=sort_key)

    dfs = []
    for fpath in files:
        try:
            df_i = pd.read_csv(fpath)
        except Exception as e:
            raise RuntimeError(f"Failed to read '{fpath}': {e}") from e

        m = rx.search(os.path.basename(fpath))
        file_num = int(m.group(1)) if (m and m.group(1) is not None) else 0

        df_i = df_i.copy()
        df_i["file_num"] = file_num
        df_i["source_file"] = os.path.basename(fpath)
        dfs.append(df_i)

    return pd.concat(dfs, axis=0, ignore_index=True)

def append_results_to_file(config, train_results, val_results, result_file):

    config = _prepare_config_for_results(config)
    config = pd.json_normalize(config, sep="_")
    results = pd.concat([config, train_results, val_results], axis=1)
    # print(results)
    print(f"Saving results to {result_file}")
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    # df_pa_table = pa.Table.from_pandas(results)
    if not os.path.isfile(result_file):
        results.to_csv(result_file, header=True, index=False)
    else:  # it exists, so append without writing the header
        results.to_csv(result_file, mode="a", header=False, index=False)


def make_config_list(base_config: dict, parameter_space: dict | None) -> list[dict]:
    """
    Expand a dict of parameter lists into a list of full configs.
    If parameter_space is empty/None, return a single config (base_config).
    """
    if not parameter_space:                     # covers {}, None
        return [deepcopy(base_config)]

    keys, values = zip(*parameter_space.items()) 
    grid = [dict(zip(keys, combo)) for combo in product(*values)]
    list_configs = []
    for params in grid:
        config = deepcopy(base_config)
        for path, val in params.items():
            set_nested(config, path, val) 
        list_configs.append(config)
    return list_configs



def _prepare_config_for_results(config, experiment_nb=None):

    def noneify(d):
        """
        Return a new dict with the same keys (and nested dict‐structure),
        but with every non‐dict value replaced by None.
        """
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out[k] = noneify(v)
            else:
                out[k] = None
        return out

    timestamp = strftime("%Y-%m-%d_%H-%M-%S", localtime())
    config["experiment"] = {}
    config["experiment"]["datetime"] = timestamp
    if experiment_nb is not None:
        config["experiment"]["folder"] = f"experiment_{experiment_nb}"
    else:
        config["experiment"]["folder"] = "bwe"

    list_methods = ["gini", "metric_learning", "clustering", "bayes", "max_proba","knn", "logistic", "random_forest"]
    method_name = config.get("method_name")

    if method_name not in list_methods:
        raise ValueError(f"Unknown method '{method_name}'")

    for m in list_methods:
        if m == method_name:
            continue

        subconf = config.get(m)
        if isinstance(subconf, dict):
            # reset all its keys to None
            config[m] = noneify(subconf)
        else:
            # nothing to reset (either missing or not a dict)
            # Optionally, you could initialize it:
            # config[m] = {}
            pass
    if config["clustering"]["reduction"]["name"] is None:
        config["clustering"]["reduction"]= dict.fromkeys(config["clustering"]["reduction"].keys(), None)

    return config
