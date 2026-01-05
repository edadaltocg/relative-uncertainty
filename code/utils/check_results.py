from code.utils.results_helper import OFFICIAL_RESULTS_FILES, NEW_OFFICIAL_RESULTS_FILES
OFFICIAL_RESULTS_FILES = NEW_OFFICIAL_RESULTS_FILES
import pandas as pd
import os
from code.utils.config import Config

def check_results(results_files=OFFICIAL_RESULTS_FILES):

    def get_results_params():
        values = {}
        list_datasets = list(results_files.keys())
        list_models = {dataset: [] for dataset in list_datasets}
        for dataset in results_files.keys():
            for model in results_files[dataset].keys():
         
                list_models[dataset].append(model)

                file = results_files[dataset][model]["clustering_opt"]
                try:
                    df = pd.read_csv(file)
                except Exception as e:
                    print(f"Could not read {file}: {e}")
                    continue
                values[(dataset, model)] = {
                    "n_clusters": df["clustering_n_clusters"].values[0],
                    "alpha": df["clustering_alpha"].values[0],
                    "method": df["clustering_name"].values[0],
                    "clustering_seed": df["clustering_seed"].values[0],
                    "init_scheme": df["clustering_init_scheme"].values[0],
                    "n_init": df["clustering_n_init"].values[0],
                    "space": df["clustering_space"].values[0],
                    "temperature": df["clustering_temperature"].values[0],
                    "cov_type": df["clustering_cov_type"].values[0],
                    "reorder_embs": df["clustering_reorder_embs"].values[0],
                    "bound": "hoeffding" if dataset != "imagenet" else df["clustering_bound"].values[0],
                    "pred_weights": 0 if ("clustering_pred_weight" not in df.columns or df["clustering_pred_weight"].isnull().values[0]) else df["clustering_pred_weight"].values[0],
                }
        return values, list_datasets, list_models

    def check_config_params(values, list_datasets, list_models):
        root = "./configs/postprocessors/clustering"
        for dataset in list_datasets:
            for model in list_models[dataset]:
                file_path = os.path.join(root, f"clustering_{dataset}_{model}_cross.yml")
                cfg = Config(file_path)["postprocessor_args"]
                del cfg["n_classes"]
                # print(cfg)
                # print(values[(dataset, model)])
                if cfg == values[(dataset, model)]:
                    print(f"[OK] {dataset}-{model} config matches results")
                else:
                    print(f"[MISMATCH] {dataset}-{model} config does not match results")
        

    results_params, list_datasets, list_models = get_results_params()

    check_config_params(results_params, list_datasets, list_models)

if __name__ == "__main__":
    check_results()

    
