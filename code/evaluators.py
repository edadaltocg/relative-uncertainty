
import torch
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from copy import deepcopy

from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits



from code.utils.helper import _prepare_config_for_results, make_grid
from code.utils.eval import  MultiDetectorEvaluator, AblationDetector
from code.utils.metrics import compute_all_metrics
from code.utils.postprocessors import get_postprocessor
# from code.utils.postprocessors.methods import (
#     PartitionDetector,
#     MegaPartitionDetector,
#     MetricLearningLagrange,
# )


class EvaluatorAblation:
    def __init__(
            self, 
            model=None, 
            cfg_detection=None,
            cfg_dataset=None,
            cal_loader=None,    
            res_loader=None,
            test_loader=None,
            device=None,
            metric='fpr', 
            quantizer_metric="same",
            result_folder="results/",
            latent_paths="latent/train_latent.pt",
            n_epochs={"res": 1, "cal": 1, "test": 1},
            n_cal=None,
            var_ablation=None,
            seed_split=0,
            verbose = True,
            mode="search",
            ):

        """
        Args:
            detectors (list): List of detector instances.
        """
        self.model = model
        self.cfg_detection = cfg_detection
        self.cfg_dataset = cfg_dataset
        self.device = device
        self.result_folder = result_folder
        self.n_epochs = n_epochs
        self.seed_split = seed_split
        self.n_cal = cfg_dataset["n_samples"]["cal"] if n_cal is None else n_cal
        self.fixed_var_ablation = var_ablation
        self.fit_after_cv = cfg_detection.get("experience_args", {}).get("fit_after_cv", False)
        # self.is_relu = is_relu
        self.postprocessor_name = cfg_detection["name"]
        self.verbose = verbose

        self.metric = metric
        self.quantizer_metric = quantizer_metric
        if metric in ["fpr", "aurc"]:
            self.metric_direction = "min"
        else:
            self.metric_direction = "max"

        self.cal_loader = cal_loader
        self.res_loader = res_loader
        self.val_loader = test_loader
        self.latent_paths = latent_paths

        self.evaluator_cal = AblationDetector(
            self.model, self.cal_loader, device=self.device, suffix="cal", latent_path=self.latent_paths["cal"],
            cfg_dataset=self.cfg_dataset, postprocessor_name=self.postprocessor_name
        )


        self.evaluator_test = AblationDetector(
            self.model, self.val_loader, device=self.device, suffix="test", latent_path=self.latent_paths["test"],
            cfg_dataset=self.cfg_dataset, postprocessor_name=self.postprocessor_name,
            result_folder=self.result_folder
        )

        self.detector = None
        self.values = {"res": None, "cal": None}
        self.mode = mode
        self.n_folds = self.cfg_detection.get("experience_args", {}).get("n_folds", 5)
        self.fit_after_cv = self.cfg_detection.get("experience_args", {}).get("fit_after_cv", False)
        self.ratio_res_split = self.cfg_detection.get("experience_args", {}).get("ratio_res_split", None)
        self.n_split_val = self.cfg_detection.get("experience_args", {}).get("n_split_val", 1)
        self.weight_std = self.cfg_detection.get("experience_args", {}).get("weight_std", 0.0)
        self.calibrate = self.cfg_detection.get("postprocessor_args", {}).get("calibrate", False)
        self.hyperparam_combination = list(make_grid(self.cfg_detection, key ="postprocessor_grid")) 
        # self.run()

   

    def get_values(self, dataloader, name="cal"):

        # all_model_preds = []
        if dataloader is None:
            return
        latent_path = self.latent_paths[f"{name}"]

        if os.path.exists(latent_path):
            pkg = torch.load(latent_path, map_location="cpu")
            all_logits = pkg["logits"].to(torch.float32)        # (N, C)
            all_labels = pkg["labels"]              # (N,)
            all_model_preds  = pkg["model_preds"]# (N,)
            all_detector_labels = (all_model_preds != all_labels).float()
            # print("logits [0]", all_logits[:1])
            # print("labels [0]", all_labels[:1])

        
        else:
                        
            self.model.to(self.device)
            self.model.eval()

            all_model_preds = []
            all_labels = []
            all_logits = []
     
            # os.makedirs("debug_aug", exist_ok=True)
            for epoch in range(self.n_epochs[f"{name}"]):
                with torch.no_grad():
                    for batch, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Getting Training Logits", disable=False):
                        # print("inputs [0]", inputs[0, :3, :3, :3])
                        # print("targets [0]", targets[:3])
                        # exit()
                        inputs = inputs.to(self.device)
                        # targets = targets.to(self.device)
                    
                        logits = self.model(inputs).cpu()  # logits: [batch_size, num_classes]
                        model_preds = torch.argmax(logits, dim=1)

                        # detector_labels = (model_preds != targets).float()
                        # # all_model_preds.append(model_preds)
                        # all_detector_labels.append(detector_labels)
                        all_logits.append(logits)
                        all_labels.append(targets.cpu())
                        all_model_preds.append(model_preds)
                 

            
            
            # all_model_preds = torch.cat(all_model_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_model_preds = torch.cat(all_model_preds, dim=0)
            all_detector_labels = (all_model_preds != all_labels).float()
            all_logits = torch.cat(all_logits, dim=0)
      
            # print("all logits 0:", all_logits[2, :10])
            # print("all labels 0:", all_labels[:10])
            # print("all model preds 0:", all_model_preds[:10])
                    
            # AFTER (robust)
            parent = os.path.dirname(latent_path)
            os.makedirs(parent, exist_ok=True)

            tmp = latent_path + ".tmp"
            torch.save(
                {
                    "logits": all_logits.cpu(),     # compact on disk
                    "labels": all_labels.cpu().to(torch.int64),
                    "model_preds": all_model_preds.cpu().to(torch.int64),
                },
                tmp,
            )
            os.replace(tmp, latent_path)  # atomic rename

        self.values[f"{name}"] = {"logits": all_logits, "detector_labels": all_detector_labels}
        if self.postprocessor_name in ["conformal", "partition"]:
            self.values[f"{name}"] = {"logits": all_logits, "detector_labels": all_detector_labels, "targets": all_labels}


    # def get_detector(self):

    #     if self.postprocessor_name == "clustering":
    #         if self.cfg_detection["postprocessor_args"]["method"] not in ["soft-kmeans_torch"]:

    #             self.detector = PartitionDetector(
    #                 model=None, 
    #                 n_clusters=self.cfg_detection["postprocessor_args"]["n_clusters"], 
    #                 alpha=self.cfg_detection["postprocessor_args"]["alpha"], 
    #                 name=self.cfg_detection["postprocessor_args"]["method"],
    #                 n_classes=self.cfg_detection["postprocessor_args"]["n_classes"], 
    #                 seed=self.cfg_detection["postprocessor_args"]["clustering_seed"], 
    #                 init_scheme=self.cfg_detection["postprocessor_args"]["init_scheme"], # "random" or "k-means++", 
    #                 n_init=self.cfg_detection["postprocessor_args"]["n_init"], # Number of initializations for k-means
    #                 space=self.cfg_detection["postprocessor_args"]["space"], 
    #                 temperature=self.cfg_detection["postprocessor_args"]["temperature"], 
    #                 cov_type = self.cfg_detection["postprocessor_args"]["cov_type"],
    #                 reorder_embs=self.cfg_detection["postprocessor_args"]["reorder_embs"], # Whether to reorder the embeddings based on the clustering
    #                 experiment_folder=self.result_folder,
    #                 bound=self.cfg_detection["postprocessor_args"]["bound"],
    #                 pred_weight=self.cfg_detection["postprocessor_args"]["pred_weights"],
    #                 batch_size=2048,
    #                 device=self.device
    #                 )
    #         else:
    #             self.detector = MegaPartitionDetector(
    #                 model=None, 
    #                 list_n_cluster=[self.cfg_detection["postprocessor_args"]["n_clusters"]],
    #                 alpha=self.cfg_detection["postprocessor_args"]["alpha"], 
    #                 name=self.cfg_detection["postprocessor_args"]["method"],
    #                 n_classes=self.cfg_detection["postprocessor_args"]["n_classes"], 
    #                 seed=self.cfg_detection["postprocessor_args"]["clustering_seed"], 
    #                 init_scheme=self.cfg_detection["postprocessor_args"]["init_scheme"],
    #                 n_init=self.cfg_detection["postprocessor_args"]["n_init"],
    #                 space=self.cfg_detection["postprocessor_args"]["space"],
    #                 temperature=self.cfg_detection["postprocessor_args"]["temperature"],                     # single temp here
    #                 cov_type=self.cfg_detection["postprocessor_args"]["cov_type"],
    #                 reorder_embs=self.cfg_detection["postprocessor_args"]["reorder_embs"],
    #                 bound=self.cfg_detection["postprocessor_args"]["bound"],
    #                 pred_weight=self.cfg_detection["postprocessor_args"].get("pred_weights", None),
    #                 batch_size=2048,
    #                 device=self.device,
    #                 experiment_folder=self.result_folder,
    #             )
    #     elif self.postprocessor_name == "relu":
    #         self.detector = MetricLearningLagrange(
    #             model=None, 
    #             lbd=self.cfg_detection["postprocessor_args"]["lambda"], 
    #             temperature=self.cfg_detection["postprocessor_args"]["temperature"],
    #             device=self.device
    #             )
    #     # else:
    #     #     self.detector = get_postprocessor(
    #     #         name=self.method, 
    #     #         model=self.model, 
    #     #         cfg=self.cfg_detection["postprocessor_args"], 
    #     #         device=self.device
    #     #         )


    def save_results(self, result_file, results, mode="append"):
        """
        Save `results` (a pandas DataFrame) to CSV.

        Parameters
        ----------
        result_file : str | Path
            Target CSV path, e.g. ".../results.csv".
        results : pd.DataFrame
            Data to write.
        mode : {"append", "increment", "overwrite"}, default "append"
            - "append": append to `result_file` (create if missing). Writes header only if file doesn't exist.
            - "increment": do not touch existing files; write to the next available incremented filename:
                results.csv, results_1.csv, results_2.csv, ...
            - "overwrite": write to `result_file`, replacing any existing file.

        Returns
        -------
        Path
            The path actually written to.
        """
        from pathlib import Path
        import re
        import pandas as pd

        # result_file = os.path.join(self.result_folder, f"{result_file}_")
        p = Path(result_file)
        p.parent.mkdir(parents=True, exist_ok=True)

        results["n_cal"] = self.n_cal
        results["seed_split"] = self.seed_split

        if mode not in {"append", "increment", "overwrite"}:
            raise ValueError(f"mode must be 'append', 'increment', or 'overwrite'; got {mode}")

        def _align_columns_for_append(target_path: Path, df_new: pd.DataFrame) -> pd.DataFrame:
            """
            If appending to an existing CSV with different columns, align to the union of columns.
            Missing columns are filled with NaN. Column order: existing columns then new-only columns.
            """
            import os
            if not target_path.exists():
                return df_new
            try:
                # Read only header to get existing columns without loading whole file.
                with target_path.open("r", encoding="utf-8") as f:
                    header_line = f.readline().rstrip("\n")
                existing_cols = header_line.split(",")
                if len(existing_cols) == 1 and existing_cols[0] == "":  # empty header edge case
                    return df_new
            except Exception:
                # If anything goes wrong, just return new df (pandas will error if incompatible)
                return df_new

            new_cols = list(df_new.columns)
            union = existing_cols + [c for c in new_cols if c not in existing_cols]
            # Reindex to the union, preserving order
            return df_new.reindex(columns=union)

        if mode == "increment":
            suffix = p.suffix  # ".csv"
            base_stem = re.sub(r"_(\d+)$", "", p.stem)
            pat = re.compile(rf"^{re.escape(base_stem)}(?:_(\d+))?$")

            max_i = -1
            for q in p.parent.glob(f"{base_stem}*{suffix}"):
                m = pat.fullmatch(q.stem)
                if m:
                    if m.group(1) is None:
                        max_i = max(max_i, 0)
                    else:
                        max_i = max(max_i, int(m.group(1)))

            target = p if max_i < 0 else p.with_name(f"{base_stem}_{max_i + 1}{suffix}")
            results.to_csv(target, header=True, index=False)
            if self.verbose:
                print(f"[save_results] wrote (increment) -> {target}")
            return target

        if mode == "overwrite":
            results.to_csv(p, header=True, index=False)
            if self.verbose:
                print(f"[save_results] wrote (overwrite) -> {p}")
            return p

        # mode == "append"
        aligned = _align_columns_for_append(p, results)
        write_header = not p.exists()
        aligned.to_csv(p, mode="a", header=write_header, index=False)
        if self.verbose:
            print(f"[save_results] wrote (append, header={write_header}) -> {p}")
        return p

    

    def fit_clustering(self):
        if self.res_loader is None:
            print("Fitting best detector on full training data")
            t0 = time.time()
            self.detector.fit(
                logits=self.values["cal"]["logits"].to(self.detector.device),
                detector_labels=self.values["cal"]["detector_labels"].to(self.detector.device),
                dataloader=self.cal_loader,
                fit_clustering=True
            )
            t1 = time.time()
            print(f"Total time: {t1 - t0:.2f} seconds")
        else:
            if self.fit_after_cv:
                print("Fitting resolution function on full resolution data")
                t0 = time.time()
                self.detector.fit(
                    logits=self.values["res"]["logits"].to(self.detector.device),
                    detector_labels=self.values["res"]["detector_labels"].to(self.detector.device),
                    dataloader=self.res_loader,
                    fit_clustering=True
                )
                t1 = time.time()
                print(f"Total time: {t1 - t0:.2f} seconds")
                if self.quantizer_metric == "likelihood":
                # for likelihood, we want to maximize it
                    best_init = torch.argmax(self.detector.clustering_algo.results.lower_bound).item()
                else:
                    conf = self.detector(logits=self.values["res"]["logits"].to(self.detector.device))
                    metrics = compute_all_metrics(
                    conf=conf.cpu().numpy(),
                    detector_labels=self.values["res"]["detector_labels"].cpu().numpy(),
                )

                    best_init = int(np.argmax(metrics[self.quantizer_metric]) if self.metric_direction == "max" else np.argmin(metrics[self.quantizer_metric])) 
                if hasattr(self.detector, "clustering_algo"):
                    
                    self.detector.clustering_algo.best_init = best_init
            print("Fitting confidence intervals on calibration data")
            t0 = time.time()
            self.detector.fit(
                logits=self.values["cal"]["logits"].to(self.detector.device),
                detector_labels=self.values["cal"]["detector_labels"].to(self.detector.device),
                dataloader=self.cal_loader,
                fit_clustering=False
            )
            t1 = time.time()
            print(f"Total time: {t1 - t0:.2f} seconds")

    def run(self):
        """
        Fit all detectors on the training data.
        
        Args:
            train_dataloader (DataLoader): DataLoader for the training data.
        """

        if self.verbose:
            print("Collecting values on res/cal data")
        t0 = time.time()
        self.get_values(self.res_loader, name="res")
        self.get_values(self.cal_loader)
        # self.get_values(self.calib_loader, calib=True)
        t1 = time.time()
        if self.verbose:
            print(f"Total time: {t1 - t0:.2f} seconds")

       
        self.detectors = [get_postprocessor(
                postprocessor_name=self.postprocessor_name, 
                model=self.model, 
                cfg=cfg, 
                result_folder=self.result_folder,
                device=self.device
                ) for cfg in self.hyperparam_combination]
       
        
     

        if self.postprocessor_name in ["partition", "clustering"]:
            for dec in self.detectors:
                dec.fit(
                    logits=self.values["cal"]["logits"].to(dec.device),
                    detector_labels=self.values["cal"]["detector_labels"].to(dec.device),
                    dataloader=self.cal_loader,
                    fit_clustering=True
                )
            
                # self.fit_clustering()
        elif self.postprocessor_name in ["relu", "random_forest", "scikit", "mlp" ]:
            self.detector.fit(
                logits=self.values["cal"]["logits"].to(self.detector.device),
                detector_labels=self.values["cal"]["detector_labels"].to(self.detector.device),
            )
        elif self.postprocessor_name == "conformal":
            
            self.detector.fit(
                logits=self.values["cal"]["logits"].to(self.detector.device),
                targets=self.values["cal"]["targets"].to(self.detector.device),
            )

        else:
            if self.verbose:
                print('No fitting required for this method')

        
        self.cal_results = self.evaluator_cal.evaluate(self.hyperparam_combination, self.detectors)
        self.cal_results = pd.concat(self.cal_results, axis=0)

        if self.verbose:
            print("Evaluating best detector on validation data")
        t0 = time.time()
        self.test_results = self.evaluator_test.evaluate(self.hyperparam_combination, self.detectors)
        self.test_results = pd.concat(self.test_results, axis=0)
        t1 = time.time()
        if self.verbose:
            print(f"Total time: {t1 - t0:.2f} seconds")


        result_file = f"results_opt-{self.metric}_qunatiz-metric-{self.quantizer_metric}-ratio-{self.ratio_res_split}_n-split-val-{self.n_split_val}_weight-std-{self.weight_std}_mode-{self.mode}.csv"
        self.save_results(
            result_file=os.path.join(self.result_folder, result_file),
            results=pd.merge(
                self.cal_results, 
                self.test_results,
                how="outer")               
            )


class HyperparamsSearch(EvaluatorAblation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    

    def search_no_fit(self):

        self.hyperparam_combination = list(make_grid(self.cfg_detection, key ="postprocessor_grid")) 

        self.detectors = [get_postprocessor(
                postprocessor_name=self.postprocessor_name, 
                model=self.model, 
                cfg=cfg, 
                result_folder=self.result_folder,
                device=self.device
                ) for cfg in self.hyperparam_combination]

        list_results = self.evaluator_cal.evaluate(self.hyperparam_combination, self.detectors)
        hyperparam_results = pd.concat(list_results, axis=0)
                   

        self.best_idx = np.argmin([np.mean(res[f"{self.metric}_cal"].values) for res in list_results])
        self.config = self.hyperparam_combination[self.best_idx]
        self.best_result = list_results[self.best_idx]
        self.detector = self.detectors[self.best_idx]

        print(f"Best Configs: {self.config}")
        print(f"Best result ({self.metric}): {self.best_result[f'{self.metric}_cal'].values}")
        self.cal_results = self.best_result

        self.save_results(
            result_file=os.path.join(self.result_folder, "hyperparams_results.csv"),
            results=hyperparam_results
            )
    def search_cross_validation(self):

        self.hyperparam_combination = list(make_grid(self.cfg_detection, key ="postprocessor_grid")) 

        self.detectors = [get_postprocessor(
                postprocessor_name=self.postprocessor_name, 
                model=self.model, 
                cfg=cfg, 
                result_folder=self.result_folder,
                device=self.device
                ) for cfg in self.hyperparam_combination]


        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=False)

        
        list_results = []

        for dec_idx, dec in tqdm(enumerate(self.detectors),total=len(self.detectors), desc="Cross validation", disable=False):

            tr_metrics = {metric: [] for metric in ["fpr", "tpr", "thr", "roc_auc", "model_acc", "aurc", "aupr_err", "aupr_success"]}
            val_metrics = {metric: [] for metric in ["fpr", "tpr", "thr", "roc_auc", "model_acc", "aurc", "aupr_err", "aupr_success"]}
            for _, (tr_idx, va_idx) in enumerate(skf.split(
                np.zeros_like(self.values["cal"]["detector_labels"]), 
                self.values["cal"]["detector_labels"]), 1):

                dec.fit(logits=self.values["cal"]["logits"][tr_idx], detector_labels=self.values["cal"]["detector_labels"][tr_idx])
                # Evaluate on validation set

                train_conf = dec(logits=self.values["cal"]["logits"][tr_idx])
                val_conf = dec(logits=self.values["cal"]["logits"][va_idx])


                for split in ["tr_cross", "val_cross"]:
                    if split == "tr_cross":
                        conf = train_conf
                        detector_labels = self.values["cal"]["detector_labels"][tr_idx]
                        metrics = tr_metrics
                    else:
                        conf = val_conf
                        detector_labels = self.values["cal"]["detector_labels"][va_idx]
                        metrics = val_metrics
                    fpr, tpr, thr, auroc, accuracy, aurc_value, aupr_in, aupr_out = compute_all_metrics(
                        conf=conf.cpu().numpy(),
                        detector_labels=detector_labels.cpu().numpy(),
                    )
                    metrics["fpr"].append(fpr)
                    metrics["tpr"].append(tpr)
                    metrics["thr"].append(thr)
                    metrics["roc_auc"].append(auroc)
                    metrics["model_acc"].append(accuracy)
                    metrics["aurc"].append(aurc_value)
                    metrics["aupr_err"].append(aupr_in)
                    metrics["aupr_success"].append(aupr_out)
            
            results = {}
            [results.update({f"{metric}_tr_cross": np.mean(tr_metrics[metric]), f"{metric}_tr_cross_std": np.std(tr_metrics[metric])}) for metric in tr_metrics.keys()]
            [results.update({f"{metric}_val_cross": np.mean(val_metrics[metric]), f"{metric}_val_cross_std": np.std(val_metrics[metric])}) for metric in val_metrics.keys()]

    
            results = pd.concat([
                pd.DataFrame([self.hyperparam_combination[dec_idx]]), 
                pd.DataFrame([results])],
                axis=1)
            list_results.append(results)

        hyperparam_results = pd.concat(list_results, axis=0)
                   

        self.best_idx = np.argmin([np.mean(res[f"{self.metric}_val_cross"].values) for res in list_results])
        self.config = self.hyperparam_combination[self.best_idx]
        self.best_result = list_results[self.best_idx]
        self.detector = self.detectors[self.best_idx]

        print(f"Best Configs: {self.config}")
        print(f"Best result ({self.metric}): {self.best_result[f'{self.metric}_val_cross'].values}")
        # self.cal_results = self.best_result

        self.save_results(
            result_file=os.path.join(self.result_folder, "hyperparams_results.csv"),
            results=hyperparam_results
            )

    def search_partition_on_val(self):

        self.hyperparam_combination = list(make_grid(self.cfg_detection, key ="postprocessor_grid")) 

        self.detectors = [get_postprocessor(
                postprocessor_name=self.postprocessor_name, 
                model=self.model, 
                cfg=cfg, 
                result_folder=self.result_folder,
                device=self.device
                ) for cfg in self.hyperparam_combination]


        
        list_results = []
        # tr_idx = first 75 %, va_idx = last 25 %
        n_tr = int(self.ratio_res_split * len(self.values["res"]["detector_labels"]))
        n_res = len(self.values["res"]["detector_labels"])
        tr_idx = np.arange(n_tr)
        va_idx = np.arange(n_tr, n_res)
        results = []

        for dec_idx, dec in tqdm(enumerate(self.detectors),total=len(self.detectors), desc="Cross validation", disable=False):

            tr_metrics = {metric: None for metric in ["fpr", "tpr", "thr", "roc_auc", "model_acc", "aurc", "aupr_err", "aupr_success"]}
            val_metrics = {metric: None for metric in ["fpr", "tpr", "thr", "roc_auc", "model_acc", "aurc", "aupr_err", "aupr_success"]}
           

            dec.fit(logits=self.values["res"]["logits"][tr_idx], detector_labels=self.values["res"]["detector_labels"][tr_idx])
            # Evaluate on validation set

            # train_conf = dec(logits=self.values["res"]["logits"][tr_idx])
            val_conf = dec(logits=self.values["res"]["logits"][va_idx])


            conf = val_conf

            val_metrics = compute_all_metrics(
                conf=conf.cpu().numpy(),
                detector_labels=self.values["res"]["detector_labels"][va_idx].cpu().numpy(),
                n_split=self.n_split_val,
                weight_std=self.weight_std,
            )


            best_init = int(np.argmax(val_metrics[self.quantizer_metric]) if self.metric_direction == "max" else np.argmin(val_metrics[self.quantizer_metric])) 
            dec.clustering_algo.best_init = best_init
            val_metrics = pd.DataFrame(val_metrics)
           
            val_metrics.columns = [f"{col}_val_res" for col in val_metrics.columns]
            print(f"Best {self.quantizer_metric}:", val_metrics.loc[best_init, f"{self.quantizer_metric}_val_res"])
            # print(val_metrics.head())
            results = pd.concat([
                pd.DataFrame([self.hyperparam_combination[dec_idx]]).reset_index(drop=True), 
                val_metrics.iloc[[best_init]].reset_index(drop=True)
                ],
                axis=1)
            list_results.append(results)
            

        hyperparam_results = pd.concat(list_results, axis=0)
                   

        # self.best_idx = np.argmin([np.mean(res[f"{self.metric}_val_res"].values) for res in list_results])
        self.best_idx = int(np.argmin(hyperparam_results[f"{self.metric}_val_res"].values) if self.metric_direction == "min" else np.argmax(hyperparam_results[f"{self.metric}_val_res"].values))
        self.config = self.hyperparam_combination[self.best_idx]
        self.best_result = list_results[self.best_idx]
        self.detector = self.detectors[self.best_idx]

        print(f"Best Configs: {self.config}")
        print(f"Best result ({self.metric}): {self.best_result[f'{self.metric}_val_res'].values}")
        # self.cal_results = self.best_result

        self.save_results(
            result_file=os.path.join(self.result_folder, f"hyperparams_results_opt-{self.metric}_qunatiz-metric-{self.quantizer_metric}-ratio-{self.ratio_res_split}_n-split-val-{self.n_split_val}_weight-std-{self.weight_std}_mode-{self.mode}.csv"),
            results=hyperparam_results
            )
        
    def search_partition_on_res(self):

        self.hyperparam_combination = list(make_grid(self.cfg_detection, key ="postprocessor_grid")) 

        

        self.detectors = [get_postprocessor(
                postprocessor_name=self.postprocessor_name, 
                model=self.model, 
                cfg=cfg, 
                result_folder=self.result_folder,
                device=self.device
                ) for cfg in self.hyperparam_combination]

        if self.calibrate:
            self.detectors[0].calibrate(logits=self.values["res"]["logits"], targets=self.values["res"]["targets"])
            T_calibrate = self.detectors[0].temperature
            print(f"Calibrated temperature: {T_calibrate}")
            for dec in self.detectors[1:]:
                dec.temperature = T_calibrate

        

        
        

        list_results = []

        for dec_idx, dec in tqdm(enumerate(self.detectors),total=len(self.detectors), desc="Model Selection on Res", disable=False):

            tr_metrics = {metric: None for metric in ["fpr", "tpr", "thr", "roc_auc", "model_acc", "aurc", "aupr_err", "aupr_success"]}
            val_metrics = {metric: None for metric in ["fpr", "tpr", "thr", "roc_auc", "model_acc", "aurc", "aupr_err", "aupr_success"]}
           

            dec.fit(logits=self.values["res"]["logits"].to(self.device), detector_labels=self.values["res"]["detector_labels"].to(self.device))
            # Evaluate on validation set

            # train_conf = dec(logits=self.values["res"]["logits"][tr_idx])
            val_conf = dec(logits=self.values["res"]["logits"])

            conf = val_conf

            metrics = compute_all_metrics(
                conf=conf.cpu().numpy(),
                detector_labels=self.values["res"]["detector_labels"].cpu().numpy(),
                n_split=self.n_split_val,
                weight_std=self.weight_std,
            )

            if self.quantizer_metric == "likelihood":
                # for likelihood, we want to maximize it
                best_init = torch.argmax(dec.clustering_algo.results.lower_bound).item()
            else:

                best_init = int(np.argmax(metrics[self.quantizer_metric]) if self.metric_direction == "max" else np.argmin(metrics[self.quantizer_metric])) 
            dec.clustering_algo.best_init = best_init
            val_metrics = pd.DataFrame(metrics)
        
           
            val_metrics.columns = [f"{col}_res" for col in val_metrics.columns]
            print(f"Best {self.quantizer_metric}:", val_metrics.loc[best_init, f"{self.quantizer_metric}_res"])
            # print(val_metrics.head())
            results = pd.concat([
                pd.DataFrame([self.hyperparam_combination[dec_idx]]).reset_index(drop=True), 
                val_metrics.iloc[[best_init]].reset_index(drop=True)
                ],
                axis=1)
            list_results.append(results)
            

        hyperparam_results = pd.concat(list_results, axis=0)
                   

        # self.best_idx = np.argmin([np.mean(res[f"{self.metric}_val_res"].values) for res in list_results])
        self.best_idx = int(np.argmin(hyperparam_results[f"{self.metric}_res"].values) if self.metric_direction == "min" else np.argmax(hyperparam_results[f"{self.metric}_res"].values))
        self.config = self.hyperparam_combination[self.best_idx]
        self.best_result = list_results[self.best_idx]
        self.detector = self.detectors[self.best_idx]

        print(f"Best Configs: {self.config}")
        print(f"Best result ({self.metric}): {self.best_result[f'{self.metric}_res'].values}")
        # self.cal_results = self.best_result

        self.save_results(
            result_file=os.path.join(self.result_folder, f"hyperparams_results_opt-{self.metric}_qunatiz-metric-{self.quantizer_metric}-ratio-{self.ratio_res_split}_n-split-val-{self.n_split_val}_weight-std-{self.weight_std}_mode-{self.mode}.csv"),
            results=hyperparam_results
            )

    def search_partition_cv(self):

        self.hyperparam_combination = list(make_grid(self.cfg_detection, key ="postprocessor_grid")) 

        self.detectors = [get_postprocessor(
                postprocessor_name=self.postprocessor_name, 
                model=self.model, 
                cfg=cfg, 
                result_folder=self.result_folder,
                device=self.device
                ) for cfg in self.hyperparam_combination]


        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=False)

        
        list_results = []

        for dec_idx, dec in tqdm(enumerate(self.detectors),total=len(self.detectors), desc="Cross validation", disable=False):

            tr_metrics = {metric: [] for metric in ["fpr", "tpr", "thr", "roc_auc", "accuracy", "aurc", "aupr_in", "aupr_out"]}
            val_metrics = {metric: [] for metric in ["fpr", "tpr", "thr", "roc_auc", "accuracy", "aurc", "aupr_in", "aupr_out"]}
            dec_results = []
            for _, (tr_idx, va_idx) in enumerate(skf.split(
                np.zeros_like(self.values["res"]["detector_labels"]), 
                self.values["res"]["detector_labels"]), 1):

                dec.fit(logits=self.values["res"]["logits"][tr_idx], detector_labels=self.values["res"]["detector_labels"][tr_idx])
                # Evaluate on validation set

                train_conf = dec(logits=self.values["res"]["logits"][tr_idx])
                val_conf = dec(logits=self.values["res"]["logits"][va_idx])


                for split in ["tr_cross", "val_cross"]:
                    if split == "tr_cross":
                        conf = train_conf
                        detector_labels = self.values["res"]["detector_labels"][tr_idx]
                        metrics = tr_metrics
                    else:
                        conf = val_conf
                        detector_labels = self.values["res"]["detector_labels"][va_idx]
                        metrics = val_metrics
                    results = compute_all_metrics(
                        conf=conf.cpu().numpy(),
                        detector_labels=detector_labels.cpu().numpy(),
                    )
                    results = pd.DataFrame(results)
                    
                    if self.quantizer_metric == "likelihood":
                # for likelihood, we want to maximize it
                        best_init = torch.argmax(dec.clustering_algo.results.lower_bound).item()
                    else:

                        best_init = int(np.argmax(results[self.metric]) if self.metric_direction == "max" else np.argmin(results[self.metric])) 
                    # dec.clustering_algo.best_init = best_init
                    results = results.iloc[best_init].to_dict()
                    [metrics[key].append(results[key]) for key in results.keys()]
                  
            
            results = {}
            [results.update({f"{metric}_tr_cross": np.mean(tr_metrics[metric]), f"{metric}_tr_cross_std": np.std(tr_metrics[metric])}) for metric in tr_metrics.keys()]
            [results.update({f"{metric}_val_cross": np.mean(val_metrics[metric]), f"{metric}_val_cross_std": np.std(val_metrics[metric])}) for metric in val_metrics.keys()]

    
            results = pd.concat([
                pd.DataFrame([self.hyperparam_combination[dec_idx]]), 
                pd.DataFrame([results])],
                axis=1)
            list_results.append(results)

        hyperparam_results = pd.concat(list_results, axis=0)
                   

        self.best_idx = np.argmin([np.mean(res[f"{self.metric}_val_cross"].values) for res in list_results])
        self.config = self.hyperparam_combination[self.best_idx]
        self.best_result = list_results[self.best_idx]
        self.detector = self.detectors[self.best_idx]

        print(f"Best Configs: {self.config}")
        print(f"Best result ({self.metric}): {self.best_result[f'{self.metric}_val_cross'].values}")
        # self.cal_results = self.best_result

        self.save_results(
            result_file=os.path.join(self.result_folder, f"hyperparams_results_opt-{self.metric}_qunatiz-metric-{self.quantizer_metric}-ratio-{self.ratio_res_split}_n-split-val-{self.n_split_val}_weight-std-{self.weight_std}_mode-{self.mode}_n-folds-{self.n_folds}.csv"),
            results=hyperparam_results
            )
    


    def run(self):
        """
        Fit all detectors on the training data.
        
        Args:
            train_dataloader (DataLoader): DataLoader for the training data.
        """
        if self.verbose:
            print("Collecting values on res/cal data")
        t0 = time.time()
        self.get_values(self.res_loader, name="res")
        self.get_values(self.cal_loader)
        # self.get_values(self.calib_loader, calib=True)
        t1 = time.time()
        if self.verbose:
            print(f"Total time: {t1 - t0:.2f} seconds")

        if self.mode == "evaluation":

            # if (self.postprocessor_name == "clustering") & (self.cfg_detection["clustering"]["name"] in ["soft-kmeans_torch", "k-means_torch"]):
            #     # self.get_optimal_detector()
            self.detector = get_postprocessor(
                postprocessor_name=self.postprocessor_name, 
                model=self.model, 
                cfg=self.cfg_detection["postprocessor_args"], 
                result_folder=self.result_folder,
                # class_subset=None,
                device=self.device
                )
            self.config = self.cfg_detection["postprocessor_args"]
            # else:
            #     print("No hyperparameter search, using the first detector")
            #     self.best_idx = 0
            #     self.best_config = self.list_configs[self.best_idx]
          
        else:
            # if (self.postprocessor_name == "clustering") & (self.cfg_detection["clustering"]["name"] in ["soft-kmeans_torch", "k-means_torch"]):
            #     print("Performing Partition cross-validation")
            #     self.partition_cross_validation()
            
            

            # elif (self.postprocessor_name in ["clustering", "random_forest"]) & (self.n_splits >= 2):
            
            #     print("Performing cross-validation")
            #     self.cross_validation()

            # elif self.postprocessor_name == "metric_learning":
            #     print("Performing cross-validation with magnitude search")
            #     self.prepare_configs_group()
            #     self.cross_validation_magnitude()

            if self.postprocessor_name in ["odin", "doctor"]:
                if self.verbose:
                    print("Performing hyperparameter search without fitting")
                t0 = time.time()
                self.search_no_fit()
                t1 = time.time()
                if self.verbose:
                    print(f"Total time: {t1 - t0:.2f} seconds")
            elif self.postprocessor_name in ["random_forest", "scikit", "mlp"]:
                if self.verbose:
                    print("Performing hyperparameter search without fitting")
                t0 = time.time()
                self.search_cross_validation()
                t1 = time.time()
                if self.verbose:
                    print(f"Total time: {t1 - t0:.2f} seconds")
            elif self.postprocessor_name == "partition":
                if self.verbose:
                    print("Performing hyperparameter search without fitting")
                t0 = time.time()
                if self.mode == "search_res":
                    self.search_partition_on_res()
                elif self.mode == "search_val":
                    self.search_partition_on_val()
                elif self.mode == "search_cv":
                    self.search_partition_cv()
                    # self.search_cross_validation()
                else:
                    raise ValueError("Unknown mode for partition search")
                t1 = time.time()
                if self.verbose:
                    print(f"Total time: {t1 - t0:.2f} seconds")

            else:
                raise ValueError("Unknown method")



        # if self.best_detector is None:
        #     self.best_detector = self.detectors[self.best_idx]

        if self.postprocessor_name in ["partition", "clustering"]:
            self.fit_clustering()
        elif self.postprocessor_name in ["relu", "random_forest", "scikit", "mlp" ]:
            self.detector.fit(
                logits=self.values["cal"]["logits"].to(self.detector.device),
                detector_labels=self.values["cal"]["detector_labels"].to(self.detector.device),
            )
        elif self.postprocessor_name == "conformal":
            
            self.detector.fit(
                logits=self.values["cal"]["logits"].to(self.detector.device),
                targets=self.values["cal"]["targets"].to(self.detector.device),
            )

        else:
            if self.verbose:
                print('No fitting required for this method')

        
        self.cal_results = self.evaluator_cal.evaluate([self.config], [self.detector])[0]


        if self.verbose:
            print("Evaluating best detector on validation data")
        t0 = time.time()
        self.test_results = self.evaluator_test.evaluate([self.config], [self.detector])[0]
        t1 = time.time()
        if self.verbose:
            print(f"Total time: {t1 - t0:.2f} seconds")

        print(f"Test result ({self.metric}): {self.test_results[f'{self.metric}_test'].values}")
        
        # if self.hyperparam_file is not None:
        #     result_file = self.hyperparam_file[:-3] + f"_opt_{self.metric}.csv"
        # else:
        result_file = f"results_opt-{self.metric}_qunatiz-metric-{self.quantizer_metric}-ratio-{self.ratio_res_split}_n-split-val-{self.n_split_val}_weight-std-{self.weight_std}_mode-{self.mode}.csv"
        self.save_results(
            result_file=os.path.join(self.result_folder, result_file),
            results=pd.merge(
                self.cal_results, 
                self.test_results,
                how="outer")
                # self.val_results.loc[:, self.val_results.columns.difference(cfg_cols)]
                
            )
        # self.save_results(
        #     result_file=os.path.join(self.result_folder, f"results_opt_{self.metric}.csv"),
        #     results=self.train_results
        #         # self.val_results.loc[:, self.val_results.columns.difference(cfg_cols)]
                
        #     )

# class HyperparameterSearch:
#     def __init__(
#             self, 
#             detectors=None, 
#             model=None, 
#             train_loader=None,
#             calib_loader=None,
#             val_loader=None,
#             device=None,
#             base_config=None, 
#             list_configs=None, 
#             list_partition_hyperparams=None,
#             metric='fpr', 
#             result_folder="results/",
#             mode = "search",
#             class_subset = None,
#             hyperparam_file = None,
#             n_cal=500
#             ):

#         """
#         Args:
#             detectors (list): List of detector instances.
#         """
#         self.detectors = detectors
#         self.model = model
#         self.device = device
#         self.base_config = base_config
#         self.method_name = base_config.get("method_name")
#         self.list_configs = list_configs
#         self.n_splits = base_config["data"]["n_splits"]
#         self.result_folder = result_folder
#         self.n_epochs = base_config["data"]["n_epochs"]
#         self.list_partition_hyperparams = list_partition_hyperparams
#         self.root = f"storage_latent/{base_config['data']['name']}_{base_config['model']['name']}_{base_config['model']['preprocessor']}_r-{base_config['data']['r']}_seed-split-{base_config['data']['seed_split']}/"
#         #self.root = f"storage_latent/______{base_config['data']['name']}_{base_config['model']['name']}_{base_config['model']['preprocessor']}_r-{base_config['data']['ratio_calib']}_seed-split-{base_config['data']['seed_split']}/"
#         self.class_subset = class_subset
#         self.hyperparam_file = hyperparam_file
#         self.n_cal = n_cal
    

#         if  (base_config['method_name'] == "clustering") & (base_config['clustering']['space'] == "classifier"):
#             self.latent_path = self.root + f"{base_config['clustering']['space']}_train_n-epochs{self.n_epochs}_transform-{base_config['data']['transform']}.pt"
            
#         else:
#             self.latent_path = self.root + f"logits_train_n-epochs{self.n_epochs}_transform-{base_config['data']['transform']}.pt"
#             self.latent_path_calib = self.root + f"{base_config['clustering']['space']}_calib_n-epochs{self.n_epochs}_transform-{base_config['data']['transform']}.pt"
#             # self.latent_path_calib = self.latent_path
#         #self.latent_path_calib = f"latent/ablation/cifar10_resnet34_n_cal/seed-split-{base_config['data']['seed_split']}/cal_n-samples-{self.n_cal}_transform-test_n-epochs-1.pt"
#         print("Latent latent_path_calib:", self.latent_path_calib)
#         self.mode = mode
#         if self.hyperparam_file is not None:
#             self.mode = "evaluation"
#         self.metric = metric
#         if metric in ["fpr", "aurc"]:
#             self.metric_direction = "min"
#         else:
#             self.metric_direction = "max"
#         self.train_loader = train_loader
#         self.calib_loader = calib_loader
#         self.val_loader = val_loader

#         self.evaluator_train = MultiDetectorEvaluator(
#             self.model, self.calib_loader, device=self.device, suffix="train", base_config=self.base_config,
          
#             )
#         self.evaluator_test = MultiDetectorEvaluator(
#             self.model, self.val_loader, device=self.device, suffix="val", base_config=self.base_config,
#         )
#         self.evaluator_cross = MultiDetectorEvaluator(
#             self.model, self.val_loader, device=self.device, suffix="cross", base_config=self.base_config,
#         )
        
#         self.best_detector = None
#         self.run()

   



#     def get_values(self, train_dataloader, calib=False):

        

#         # all_model_preds = []
#         if calib:
#             latent_path = self.latent_path_calib
#         else:
#             latent_path = self.latent_path
#         if os.path.exists(latent_path):
#             pkg = torch.load(latent_path, map_location="cpu")
#             all_logits = pkg["logits"].to(torch.float32)        # (N, C)
#             all_labels = pkg["labels"]              # (N,)
#             all_model_preds  = pkg["model_preds"]# (N,)
#             all_detector_labels = (all_model_preds != all_labels).float()
    

        
#         else:
                        
#             self.model.to(self.device)
#             self.model.eval()

#             all_model_preds = []
#             all_labels = []
#             all_logits = []
#             all_inputs = []
#             # os.makedirs("debug_aug", exist_ok=True)
#             for epoch in range(self.n_epochs):
#                 with torch.no_grad():
#                     for batch, (inputs, targets) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Getting Training Logits", disable=False):
                            
#                         inputs = inputs.to(self.device)
#                         # print("inputs [0]", inputs[0, :3, :3, :3])
#                         # print("targets [0]", targets[:3])
#                         # exit()
#                         # targets = targets.to(self.device)
                    
#                         logits = self.model(inputs).cpu()  # logits: [batch_size, num_classes]
#                         model_preds = torch.argmax(logits, dim=1)

#                         # detector_labels = (model_preds != targets).float()
#                         # # all_model_preds.append(model_preds)
#                         # all_detector_labels.append(detector_labels)
#                         all_logits.append(logits)
#                         all_labels.append(targets.cpu())
#                         all_model_preds.append(model_preds)
#                         all_inputs.append(inputs.cpu())

            
            
#             # all_model_preds = torch.cat(all_model_preds, dim=0)
#             all_labels = torch.cat(all_labels, dim=0)
#             all_model_preds = torch.cat(all_model_preds, dim=0)
#             all_detector_labels = (all_model_preds != all_labels).float()
#             all_logits = torch.cat(all_logits, dim=0)
#             all_inputs = torch.cat(all_inputs, dim=0)

#             # AFTER (robust)
#             parent = os.path.dirname(latent_path)
#             os.makedirs(parent, exist_ok=True)



#             # print("logit", all_logits.mean(), all_logits.std())
#             # print("dtype:", all_logits.dtype)
#             # print("inputs means std:", all_inputs.mean(), all_inputs.std())
#             # print("inputs dtype:", all_inputs.dtype)
#             # torch.save(all_logits, "debug_aug/logits.pt")
          
#             tmp = latent_path + ".tmp"
           
#             torch.save(
#                 {
#                     "logits": all_logits.cpu(),     # compact on disk
#                     "labels": all_labels.cpu().to(torch.int64),
#                     "model_preds": all_model_preds.cpu().to(torch.int64),
#                 },
#                 tmp,
#             )
#             os.replace(tmp, latent_path)  # atomic rename
            
#         if calib:
#             self.values_calib = {"logits": all_logits, "detector_labels": all_detector_labels}
#         else:
#             self.values = {"logits": all_logits, "detector_labels": all_detector_labels}


#     def prepare_configs_group(self):
#         groups = {}
#         order = []
        
#         for i, cfg in enumerate(self.list_configs):
#             key = tuple((k, v) for k, v in cfg[self.method_name].items() if k != "magnitude")
#             if key not in groups:
#                 groups[key] = []
#                 order.append(key)
#             groups[key].append(i)
#         self.config_groups = [groups[k] for k in order]

            

#     def aggregate_cv_over_folds(self, per_fold_results: list[list[pd.DataFrame]]) -> list[pd.DataFrame]:
#         """
#         Args
#         ----
#         per_fold_results : list over folds [
#             # fold k (1..K)
#             [ df_det0_k, df_det1_k, ..., df_det{D-1}_k ]   # each df is 1 row with config + metrics_{cross_fold-k}
#         ]

#         Returns
#         -------
#         per_detector_agg : list of length D
#             Each element is a 1-row DataFrame with config columns first,
#             then {metric}_cross_mean and {metric}_cross_std.
#         """
#         import re

#         METRICS = ["fpr","tpr","thr","roc_auc","model_acc","aurc","aupr_err","aupr_success"]

#         n_folds = len(per_fold_results)
#         D = len(per_fold_results[0]) if n_folds > 0 else 0
        

#         per_detector_agg = []

#         for det_idx in range(D):
#             # Stack the 1-row DataFrames for this detector across folds (rows become n_folds)
#             df_all = pd.concat([per_fold_results[k][det_idx] for k in range(n_folds)],
#                             axis=0, ignore_index=True)

#             # Identify config columns (everything that is NOT suffixed with _val_cross_fold-<num>)
#             fold_suffix_re = re.compile(r"_val_cross_fold-\d+$")
#             cfg_cols = [c for c in df_all.columns if not fold_suffix_re.search(c)]

#             # Take config from the first row (identical across folds by construction)
#             out = df_all[cfg_cols].iloc[[0]].copy()   # keep as 1-row DataFrame

#             # For each metric, collect all fold-specific columns and aggregate
#             for m in METRICS:
#                 pat = re.compile(rf"^{re.escape(m)}_val_cross_fold-\d+$")
#                 mcols = [c for c in df_all.columns if pat.match(c)]
#                 if not mcols:
#                     out[f"{m}_val_cross_mean"] = np.nan
#                     out[f"{m}_val_cross_std"]  = np.nan
#                     continue

              
#                 stacked = df_all[mcols].stack(future_stack=True)  # new implementation, no dropna
#                 stacked = stacked.dropna()  
#                 vals = pd.to_numeric(stacked, errors="coerce").to_numpy()
#                 cnt = np.isfinite(vals).sum()
#                 mean = float(np.nanmean(vals)) if cnt else np.nan
#                 std  = float(np.nanstd(vals, ddof=1)) if cnt > 1 else 0.0

#                 out[f"{m}_val_cross_mean"] = mean
#                 out[f"{m}_val_cross_std"]  = std

#             per_detector_agg.append(out)

#         return per_detector_agg

#     def cross_validation_magnitude(self):
        

#         list_results = []
#         list_magnitudes = [config[self.method_name]["magnitude"] for config in self.list_configs]
#         skf = StratifiedKFold(n_splits=self.n_splits, shuffle=False)
        

         

#         # for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros_like(self.values["detector_labels"]), self.values["detector_labels"]), 1):
#         for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros_like(self.values["detector_labels"]), self.values["detector_labels"]), 1):
            
            
#             val_scores = {i : np.zeros(len(va_idx)) for i in range(len(self.list_configs))}
#             val_loader = torch.utils.data.DataLoader(
#                 torch.utils.data.Subset(self.train_loader.dataset, va_idx),
#                 batch_size=self.train_loader.batch_size, shuffle=False,
#                 num_workers=self.train_loader.num_workers, pin_memory=True
#             )

#             logits_train = self.values["logits"][tr_idx].to(self.device)
#             detector_labels_train = self.values["detector_labels"][tr_idx].to(self.device)
#             detector_labels_val = self.values["detector_labels"][va_idx].cpu().numpy()

#             for group in tqdm(self.config_groups, total=len(self.config_groups), desc="Group Cross validation", disable=False):

#                 list_magnitudes = [self.list_configs[cfg_idx][self.method_name]["magnitude"] for cfg_idx in group]
#                 proto_dec = self.detectors[group[0]]
#                 proto_dec.fit(logits=logits_train, detector_labels=detector_labels_train)

#                 write = 0
#                 for inputs, _ in val_loader:
                  
     
#                     bs = inputs.size(0)
#                     inputs = inputs.to(self.device).detach().requires_grad_(True)
#                     logits = self.model(inputs)
#                     score = proto_dec(logits=logits)
#                     loss = torch.log(score + 1e-12).sum()
#                     grad_inputs, = torch.autograd.grad(loss, inputs, retain_graph=False, create_graph=False)
#                     grad_sign = grad_inputs.sign()
#                     with torch.no_grad():
#                         list_adv_inputs = [inputs + magnitude * grad_sign for magnitude in list_magnitudes]
#                     with torch.inference_mode():
#                         list_logits_adv = [self.model(adv) for adv in list_adv_inputs]
#                         scores_adv = [proto_dec(logits=logits_adv) for logits_adv in list_logits_adv]

#                     for cfg_idx, scores in zip(group, scores_adv):
#                         val_scores[cfg_idx][write:write+bs] = scores.cpu().numpy()
#                     write += bs
#             # print("val_scores[cfg_idx]", val_scores[0][:10])
            
#             list_results.append(self.evaluator_cross.evaluate(
#                 list_configs=self.list_configs,
#                 all_scores= [val_scores[i] for i in range(len(self.list_configs))],
#                 detector_labels=detector_labels_val,
#                 suffix=f"val_cross_fold-{fold}"))
        
#         list_results = self.aggregate_cv_over_folds(list_results)

#         cross_val_results = pd.concat(list_results, axis=0)
#         self.crossval_results = cross_val_results


#         self.best_idx = np.argmin([np.mean(res[f"{self.metric}_val_cross_mean"].values) for res in list_results])
#         self.best_config = self.list_configs[self.best_idx]
#         self.best_result = list_results[self.best_idx]
#         print(f"Best results: {self.best_result[[col for col in self.best_result.columns if col.startswith(self.method_name)]]}")
#         print(f"Best result ({self.metric}): {self.best_result[f'{self.metric}_val_cross_mean'].values}")


#         self.save_results(
#             result_file=os.path.join(self.result_folder, "hyperparams_results.csv"),
#             results=cross_val_results
#             )
        


#     def get_optimal_detector(self):
#         if self.hyperparam_file is not None:
#             # from code.utils.helper import read_table
#             # cross_val_results = read_table(
#             #     expe_folder=self.result_folder,
#             #     hyperparam=True
#             # )
#             cross_val_results = pd.read_csv(os.path.join(self.result_folder, self.hyperparam_file))
#             vals = cross_val_results[f"{self.metric}_val_cross"].values
#             best_idx = int(np.argmax(vals) if self.metric_direction == "max" else np.argmin(vals))
#             self.best_idx = best_idx
#             self.best_result = cross_val_results.iloc[[best_idx]]
#             self.best_config = deepcopy(self.base_config)
#             self.best_config["clustering"]["n_clusters"] = self.best_result["clustering.n_clusters"].item()
#             self.best_config["clustering"]["temperature"] = self.best_result["clustering.temperature"].item()
        
#             print(f"Best results: {self.best_result[[col for col in self.best_result.columns if col.startswith(self.method_name)]]}")
#             print(f"Best result ({self.metric}): {self.best_result[f'{self.metric}_val_cross'].values}")


#             # self.save_results(
#             #     result_file=os.path.join(self.result_folder, "hyperparams_results.csv"),
#             #     results=cross_val_results
#             #     )
#             self.best_detector = MegaPartitionDetector(
#                     model=None, 
#                     list_n_cluster=[self.best_config["clustering"]["n_clusters"]],
#                     alpha=self.base_config["clustering"]["alpha"], 
#                     name=self.base_config["clustering"]["name"],
#                     n_classes=self.base_config["data"]["n_classes"], 
#                     seed=self.base_config["clustering"]["seed"], 
#                     init_scheme=self.base_config["clustering"]["init_scheme"],
#                     n_init=self.base_config["clustering"]["n_init"],
#                     space=self.base_config["clustering"]["space"],
#                     temperature=self.best_config["clustering"]["temperature"],                     # single temp here
#                     cov_type=self.base_config["clustering"]["cov_type"],
#                     reduction_name=self.base_config["clustering"]["reduction"]["name"],
#                     reduction_dim=self.base_config["clustering"]["reduction"]["dim"],
#                     reduction_n_neighbors=self.base_config["clustering"]["reduction"]["n_neighbors"],
#                     reduction_seed=self.base_config["clustering"]["reduction"]["seed"],
#                     normalize_gini=self.base_config["clustering"]["normalize_gini"],
#                     distance=self.base_config["clustering"]["distance"],
#                     reorder_embs=self.base_config["clustering"]["reorder_embs"],
#                     experiment_folder=None,
#                     class_subset=self.base_config["clustering"].get("pred_weight", None),
#                     params_path=None,
#                     pred_weight=self.base_config["clustering"].get("pred_weight", None),
#                     batch_size=2048,
#                     device=self.device,
#                 )
#         else:
#             self.best_detector = MegaPartitionDetector(
#             model=None, 
#             list_n_cluster=[self.base_config["clustering"]["n_clusters"]],
#             alpha=self.base_config["clustering"]["alpha"], 
#             name=self.base_config["clustering"]["name"],
#             n_classes=self.base_config["data"]["n_classes"], 
#             seed=self.base_config["clustering"]["seed"], 
#             init_scheme=self.base_config["clustering"]["init_scheme"],
#             n_init=self.base_config["clustering"]["n_init"],
#             space=self.base_config["clustering"]["space"],
#             temperature=self.base_config["clustering"]["temperature"],                     # single temp here
#             cov_type=self.base_config["clustering"]["cov_type"],
#             reduction_name=self.base_config["clustering"]["reduction"]["name"],
#             reduction_dim=self.base_config["clustering"]["reduction"]["dim"],
#             reduction_n_neighbors=self.base_config["clustering"]["reduction"]["n_neighbors"],
#             reduction_seed=self.base_config["clustering"]["reduction"]["seed"],
#             normalize_gini=self.base_config["clustering"]["normalize_gini"],
#             distance=self.base_config["clustering"]["distance"],
#             reorder_embs=self.base_config["clustering"]["reorder_embs"],
#             experiment_folder=None,
#             class_subset=self.class_subset,
#             params_path=None,
#             pred_weight=self.base_config["clustering"].get("pred_weight", None),
#             batch_size=2048,
#             device=self.device,
#         )
#         self.best_config = deepcopy(self.base_config)

    
#     def partition_cross_validation(self):
        
#         skf = StratifiedKFold(n_splits=self.n_splits, shuffle=False)

#         # Pre-build the full hyperparam grid (temperature × n_clusters).
#         temps = list(self.list_partition_hyperparams["temperatures"])
#         nclist = list(self.list_partition_hyperparams["n_clusters"])

#         # If your detector batches only across n_clusters (not temps), we loop temps outside.
#         all_results = []
#         all_configs = []

#         for temperature in tqdm(temps, desc="Temperature", disable=False):
#             # One batched detector across k; fit once per fold.
#             mega_detector = MegaPartitionDetector(
#                 model=None, 
#                 list_n_cluster=nclist,
#                 alpha=self.base_config["clustering"]["alpha"], 
#                 name=self.base_config["clustering"]["name"],
#                 n_classes=self.base_config["data"]["n_classes"], 
#                 seed=self.base_config["clustering"]["seed"], 
#                 init_scheme=self.base_config["clustering"]["init_scheme"],
#                 n_init=self.base_config["clustering"]["n_init"],
#                 space=self.base_config["clustering"]["space"],
#                 temperature=temperature,                     # single temp here
#                 cov_type=self.base_config["clustering"]["cov_type"],
#                 reduction_name=self.base_config["clustering"]["reduction"]["name"],
#                 reduction_dim=self.base_config["clustering"]["reduction"]["dim"],
#                 reduction_n_neighbors=self.base_config["clustering"]["reduction"]["n_neighbors"],
#                 reduction_seed=self.base_config["clustering"]["reduction"]["seed"],
#                 normalize_gini=self.base_config["clustering"]["normalize_gini"],
#                 distance=self.base_config["clustering"]["distance"],
#                 reorder_embs=self.base_config["clustering"]["reorder_embs"],
#                 experiment_folder=None,
#                 class_subset=self.class_subset,
#                 params_path=None,
#                 pred_weight=self.base_config["clustering"].get("pred_weight", None),
#                 batch_size=2048,
#                 device=self.device,
#             )

#             # Accumulators per H config across folds
#             H = len(nclist)
#             tr_accum = {k: [] for k in ["fpr", "tpr", "thr", "roc_auc", "model_acc", "aurc", "aupr_err", "aupr_success"]}
#             va_accum = {k: [] for k in ["fpr", "tpr", "thr", "roc_auc", "model_acc", "aurc", "aupr_err", "aupr_success"]}

#             # Optional: if you want to keep curves, use separate accumulators for arrays

#             for fold_idx, (tr_idx, va_idx) in enumerate(
#                 skf.split(np.zeros_like(self.values["detector_labels"]), self.values["detector_labels"]), 1
#             ):
#                 logits_tr = self.values["logits"][tr_idx].to(self.device)
#                 labels_tr = self.values["detector_labels"][tr_idx].to(self.device)
#                 logits_va = self.values["logits"][va_idx].to(self.device)
#                 labels_va = self.values["detector_labels"][va_idx].to(self.device)

#                 # Fit once per fold (batched over H n_clusters)
#                 mega_detector.fit(logits=logits_tr, detector_labels=labels_tr)
#                 # print("lower bounds", mega_detector.lower_bound)
        
#                 # Scores per H config
#                 conf_tr = mega_detector(logits=logits_tr)  # (H, N_tr)
#                 conf_va = mega_detector(logits=logits_va)  # (H, N_va)

#                 # print("shape conf_tr", conf_tr.shape)
#                 # print("shape conf_va", conf_va.shape)   
#                 # Compute metrics in batch (assumed to return H-shaped tensors/arrays)
#                 tr = compute_all_metrics(conf=conf_tr.detach().cpu().numpy(),
#                                         detector_labels=labels_tr.detach().cpu().numpy())
#                 va = compute_all_metrics(conf=conf_va.detach().cpu().numpy(),
#                                         detector_labels=labels_va.detach().cpu().numpy())
#                 # print("shape tr", tr)
#                 # Accumulate scalar metrics; convert to np.array with shape (H,)
#                 for idx, k in enumerate(tr_accum.keys()):
#                     tr_accum[k].append(np.asarray(tr[idx]))   # each append is (H,)
#                     va_accum[k].append(np.asarray(va[idx]))

#             # Aggregate across folds: mean & std along axis=0 (fold axis)
#             summary_rows = []
#             for h, k_val in enumerate(nclist):
#                 row = {
#                     "clustering.n_clusters": int(k_val),
#                     "clustering.temperature": float(temperature),
#                 }
#                 # For each metric, stack folds (F, H) then take column h
#                 for name, acc in tr_accum.items():
#                     A = np.stack(acc, axis=0)[:, h]  # (F,)
#                     row[f"{name}_tr_cross"] = A.mean()
#                     row[f"{name}_tr_cross_std"] = A.std(ddof=0)
#                 for name, acc in va_accum.items():
#                     A = np.stack(acc, axis=0)[:, h]  # (F,)
#                     row[f"{name}_val_cross"] = A.mean()
#                     row[f"{name}_val_cross_std"] = A.std(ddof=0)
#                 summary_rows.append(row)

#             df_temp = pd.DataFrame(summary_rows)
#             all_results.append(df_temp)

#             # cache configs aligned with rows
#             for k_val in nclist:
#                 cfg = deepcopy(self.base_config)
#                 cfg["clustering"]["n_clusters"] = int(k_val)
#                 cfg["clustering"]["temperature"] = float(temperature)
#                 all_configs.append(cfg)

#         cross_val_results = pd.concat(all_results, axis=0, ignore_index=True)
#         self.crossval_results = cross_val_results
#         self.list_configs = all_configs

#         # Pick best by your chosen metric/direction
#         metric = self.metric  # e.g., "roc_auc" or "aurc"
#         if metric in ["fpr", "aurc"]:
#             direction = "min"
#         else:
#             direction = "max"
    

#         vals = cross_val_results[f"{metric}_val_cross"].values
#         best_idx = int(np.argmax(vals) if direction == "max" else np.argmin(vals))
#         self.best_idx = best_idx
#         self.best_result = cross_val_results.iloc[[best_idx]]
#         self.best_config = self.list_configs[best_idx]

#         print(f"Best results: {self.best_result[[col for col in self.best_result.columns if col.startswith(self.method_name)]]}")
#         print(f"Best result ({self.metric}): {self.best_result[f'{self.metric}_val_cross'].values}")


#         self.save_results(
#             result_file=os.path.join(self.result_folder, "hyperparams_results.csv"),
#             results=cross_val_results
#             )
#         self.best_detector = MegaPartitionDetector(
#                 model=None, 
#                 list_n_cluster=[self.best_config["clustering"]["n_clusters"]],
#                 alpha=self.base_config["clustering"]["alpha"], 
#                 name=self.base_config["clustering"]["name"],
#                 n_classes=self.base_config["data"]["n_classes"], 
#                 seed=self.base_config["clustering"]["seed"], 
#                 init_scheme=self.base_config["clustering"]["init_scheme"],
#                 n_init=self.base_config["clustering"]["n_init"],
#                 space=self.base_config["clustering"]["space"],
#                 temperature=self.best_config["clustering"]["temperature"],                     # single temp here
#                 cov_type=self.base_config["clustering"]["cov_type"],
#                 reduction_name=self.base_config["clustering"]["reduction"]["name"],
#                 reduction_dim=self.base_config["clustering"]["reduction"]["dim"],
#                 reduction_n_neighbors=self.base_config["clustering"]["reduction"]["n_neighbors"],
#                 reduction_seed=self.base_config["clustering"]["reduction"]["seed"],
#                 normalize_gini=self.base_config["clustering"]["normalize_gini"],
#                 distance=self.base_config["clustering"]["distance"],
#                 reorder_embs=self.base_config["clustering"]["reorder_embs"],
#                 experiment_folder=None,
#                 class_subset=self.class_subset,
#                 params_path=None,
#                 pred_weight=self.base_config["clustering"].get("pred_weight", None),
#                 batch_size=2048,
#                 device=self.device,
#             )

    
#     def cross_validation(self):

#         skf = StratifiedKFold(n_splits=self.n_splits, shuffle=False)

        

#         # Optional: precompute/cached features/logits here to speed up, if your detectors support it.
#         list_results = []

#         # if self.method_name == "clustering":
            
#         #     train_results = []
#         #     for dec_idx, dec in tqdm(enumerate(self.detectors),total=len(self.detectors), desc="Cross validation", disable=False):
#         #         for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros_like(self.values["detector_labels"]), self.values["detector_labels"]), 1):

#         #             dec.fit(logits=self.values["logits"][tr_idx].to(dec.device), detector_labels=self.values["detector_labels"][tr_idx].to(dec.device))
#         #             scores = dec(logits=self.values["logits"][tr_idx].to(dec.device))
#         #             self.evaluator_cross.scores = {
#         #                 "scores" : scores,
#         #                 "detector_labels" : self.values["detector_labels"][tr_idx].to(dec.device)
#         #             }
#         #             train_results.append(self.evaluator_cross.evaluate([dec], [self.list_configs[dec_idx]])[0])


        

#         for dec_idx, dec in tqdm(enumerate(self.detectors),total=len(self.detectors), desc="Cross validation", disable=False):

#             tr_metrics = {metric: [] for metric in ["fpr", "tpr", "thr", "roc_auc", "model_acc", "aurc", "aupr_err", "aupr_success"]}
#             val_metrics = {metric: [] for metric in ["fpr", "tpr", "thr", "roc_auc", "model_acc", "aurc", "aupr_err", "aupr_success"]}
#             for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros_like(self.values["detector_labels"]), self.values["detector_labels"]), 1):

#                 dec.fit(logits=self.values["logits"][tr_idx], detector_labels=self.values["detector_labels"][tr_idx])
#                 # Evaluate on validation set

#                 train_conf = dec(logits=self.values["logits"][tr_idx])
#                 val_conf = dec(logits=self.values["logits"][va_idx])

#                 if self.method_name == "metric_learning":

#                     self.evaluator_cross

#                 for split in ["tr_cross", "val_cross"]:
#                     if split == "tr_cross":
#                         conf = train_conf
#                         detector_labels = self.values["detector_labels"][tr_idx]
#                         metrics = tr_metrics
#                     else:
#                         conf = val_conf
#                         detector_labels = self.values["detector_labels"][va_idx]
#                         metrics = val_metrics
#                     fpr, tpr, thr, auroc, accuracy, aurc_value, aupr_in, aupr_out = compute_all_metrics(
#                         conf=conf.cpu().numpy(),
#                         detector_labels=detector_labels.cpu().numpy(),
#                     )
#                     metrics["fpr"].append(fpr)
#                     metrics["tpr"].append(tpr)
#                     metrics["thr"].append(thr)
#                     metrics["roc_auc"].append(auroc)
#                     metrics["model_acc"].append(accuracy)
#                     metrics["aurc"].append(aurc_value)
#                     metrics["aupr_err"].append(aupr_in)
#                     metrics["aupr_success"].append(aupr_out)

#             results = pd.DataFrame([{
#                 "fpr_tr_cross": np.mean(tr_metrics["fpr"]),
#                 "fpr_tr_cross_std": np.std(tr_metrics["fpr"]),
#                 "tpr_tr_cross": np.mean(tr_metrics["tpr"]),
#                 "tpr_tr_cross_std": np.std(tr_metrics["tpr"]),
#                 "thr_tr_cross": np.mean(tr_metrics["thr"]),
#                 "thr_tr_cross_std": np.std(tr_metrics["thr"]),
#                 "roc_auc_tr_cross": np.mean(tr_metrics["roc_auc"]),
#                 "roc_auc_tr_cross_std": np.std(tr_metrics["roc_auc"]),
#                 "model_acc_tr_cross": np.mean(tr_metrics["model_acc"]),
#                 "model_acc_tr_cross_std": np.std(tr_metrics["model_acc"]),
#                 "aurc_tr_cross": np.mean(tr_metrics["aurc"]),
#                 "aurc_tr_cross_std": np.std(tr_metrics["aurc"]),
#                 "aupr_err_tr_cross": np.mean(tr_metrics["aupr_err"]),
#                 "aupr_err_tr_cross_std": np.std(tr_metrics["aupr_err"]),
#                 "aupr_success_tr_cross": np.mean(tr_metrics["aupr_success"]),
#                 "aupr_success_tr_cross_std": np.std(tr_metrics["aupr_success"]),
#                 "fpr_val_cross": np.mean(val_metrics["fpr"]),
#                 "fpr_val_cross_std": np.std(val_metrics["fpr"]),
#                 "tpr_val_cross": np.mean(val_metrics["tpr"]),
#                 "tpr_val_cross_std": np.std(val_metrics["tpr"]),
#                 "thr_val_cross": np.mean(val_metrics["thr"]),
#                 "thr_val_cross_std": np.std(val_metrics["thr"]),
#                 "roc_auc_val_cross": np.mean(val_metrics["roc_auc"]),
#                 "roc_auc_val_cross_std": np.std(val_metrics["roc_auc"]),
#                 "model_acc_val_cross": np.mean(val_metrics["model_acc"]),
#                 "model_acc_val_cross_std": np.std(val_metrics["model_acc"]),
#                 "aurc_val_cross": np.mean(val_metrics["aurc"]),
#                 "aurc_val_cross_std": np.std(val_metrics["aurc"]),
#                 "aupr_err_val_cross": np.mean(val_metrics["aupr_err"]),
#                 "aupr_err_val_cross_std": np.std(val_metrics["aupr_err"]),
#                 "aupr_success_val_cross": np.mean(val_metrics["aupr_success"]),
#                 "aupr_success_val_cross_std": np.std(val_metrics["aupr_success"]),
#             }])
            
#             config = _prepare_config_for_results(self.list_configs[dec_idx])
#             config = pd.json_normalize(config, sep="_")
#             results = pd.concat([config, results], axis=1)
#             list_results.append(results)
        
#         cross_val_results = pd.concat(list_results, axis=0)
#         self.crossval_results = cross_val_results


#         self.best_idx = np.argmin([np.mean(res[f"{self.metric}_val_cross"].values) for res in list_results])
#         self.best_config = self.list_configs[self.best_idx]
#         self.best_result = list_results[self.best_idx]
#         print(f"Best results: {self.best_result[[col for col in self.best_result.columns if col.startswith(self.method_name)]]}")
#         print(f"Best result ({self.metric}): {self.best_result[f'{self.metric}_val_cross'].values}")


#         self.save_results(
#             result_file=os.path.join(self.result_folder, "hyperparams_results.csv"),
#             results=cross_val_results
#             )


#     # def save_results(self, result_file, results):

#     #     print(f"Saving results to {result_file}")
#     #     os.makedirs(os.path.dirname(result_file), exist_ok=True)

#     #     if not os.path.isfile(result_file):
#     #         results.to_csv(result_file, header=True, index=False)
#     #     else:
#     #         print(f"Results already exist at {result_file}")
#     #         result_file = result_file.replace(".csv", "_append.csv")
#     #         results.to_csv(result_file, header=True, index=False)
#     def save_results(self, result_file, results):
#         """
#         Save `results` to CSV. If `result_file` already exists (or any numbered variants exist),
#         write to the next available incremented filename:
#             base.csv, base_1.csv, base_2.csv, ...
#         Returns the path it wrote to.
#         """
#         from pathlib import Path
#         import re

#         p = Path(result_file)
#         p.parent.mkdir(parents=True, exist_ok=True)

#         # Base name and extension
#         suffix = p.suffix  # e.g. ".csv"
#         base_stem = re.sub(r"_(\d+)$", "", p.stem)  # strip trailing _<num> if present

#         # Regex to match either "base" or "base_<num>" (same suffix)
#         pat = re.compile(rf"^{re.escape(base_stem)}(?:_(\d+))?$")

#         max_i = -1
#         # Scan directory for existing matching files to find the highest index
#         for q in p.parent.glob(f"{base_stem}*{suffix}"):
#             m = pat.fullmatch(q.stem)
#             if m:
#                 if m.group(1) is None:
#                     max_i = max(max_i, 0)     # the unnumbered base file counts as index 0
#                 else:
#                     max_i = max(max_i, int(m.group(1)))

#         if max_i < 0:
#             # No collisions at all -> use the base path
#             target = p
#         else:
#             # Bump to the next available index after the max we saw
#             next_i = max_i + 1
#             target = p.with_name(f"{base_stem}_{next_i}{suffix}")

#         print(f"Saving results to {target}")
#         results.to_csv(target, header=True, index=False)



#     def search_no_fit(self):

#         list_results = self.evaluator_train.evaluate(self.list_configs, self.detectors)
#         hyperparam_results = pd.concat(list_results, axis=0)
                   

#         self.best_idx = np.argmin([np.mean(res[f"{self.metric}_train"].values) for res in list_results])
#         self.best_config = self.list_configs[self.best_idx]
#         self.best_result = list_results[self.best_idx]

#         print(f"Best Configs: {self.best_result[[col for col in self.best_result.columns if col.startswith(self.method_name)]]}")
#         print(f"Best result ({self.metric}): {self.best_result[f'{self.metric}_train'].values}")
#         self.train_results = self.best_result

#         self.save_results(
#             result_file=os.path.join(self.result_folder, "hyperparams_results.csv"),
#             results=hyperparam_results
#             )


#     def run(self):
#         """
#         Fit all detectors on the training data.
        
#         Args:
#             train_dataloader (DataLoader): DataLoader for the training data.
#         """

#         import time
#         print("Collecting values on training data")
#         t0 = time.time()
#         if self.train_loader is not None:
#             self.get_values(self.train_loader)
#         self.get_values(self.calib_loader, calib=True)
#         t1 = time.time()
#         print(f"Total time: {t1 - t0:.2f} seconds")

#         if self.mode == "evaluation":

#             if (self.method_name == "clustering") & (self.base_config["clustering"]["name"] in ["soft-kmeans_torch", "k-means_torch"]):
#                 self.get_optimal_detector()
#             else:
#                 print("No hyperparameter search, using the first detector")
#                 self.best_idx = 0
#                 self.best_config = self.list_configs[self.best_idx]
          
#         else:
#             if (self.method_name == "clustering") & (self.base_config["clustering"]["name"] in ["soft-kmeans_torch", "k-means_torch"]):
#                 print("Performing Partition cross-validation")
#                 self.partition_cross_validation()
            
            

#             elif (self.method_name in ["clustering", "random_forest", "mlp"]) & (self.n_splits >= 2):
            
#                 print("Performing cross-validation")
#                 self.cross_validation()

#             elif self.method_name == "metric_learning":
#                 print("Performing cross-validation with magnitude search")
#                 self.prepare_configs_group()
#                 self.cross_validation_magnitude()

#             elif self.method_name in ["max_proba", "gini"]:
#                 print("Performing hyperparameter search without fitting")
#                 t0 = time.time()
#                 self.search_no_fit()
#                 t1 = time.time()
#                 print(f"Total time: {t1 - t0:.2f} seconds")
#             else:
#                 raise ValueError("Unknwon method")
    


#         if self.best_detector is None:
#             self.best_detector = self.detectors[self.best_idx]

#         if hasattr(self.best_detector, 'fit'):
#             print("Fitting best detector on full training data")
#             t0 = time.time()
#             # self.best_detector.fit(
#             #     logits=self.values_calib["logits"].to(self.best_detector.device), 
#             #     detector_labels=self.values_calib["detector_labels"].to(self.best_detector.device),
#             #     dataloader=self.train_loader,
#             #     fit_clustering=True 
#             #     )
#             self.best_detector.fit(
#                 logits=self.values["logits"].to(self.best_detector.device), 
#                 detector_labels=self.values["detector_labels"].to(self.best_detector.device),
#                 dataloader=self.train_loader,
#                 fit_clustering=True 
#                 )
#             self.best_detector.fit(
#                 logits=self.values_calib["logits"].to(self.best_detector.device), 
#                 detector_labels=self.values_calib["detector_labels"].to(self.best_detector.device),
#                 dataloader=self.calib_loader,
#                 fit_clustering=False 
#                 )
#             t1 = time.time()
#             print(f"Total time: {t1 - t0:.2f} seconds")
#             print("Evaluating best detector on training data")
#             self.train_results = self.evaluator_train.evaluate([self.best_config], [self.best_detector])[0]
#             print(f"Train result ({self.metric}): {self.train_results[f'{self.metric}_train'].values}")
        
#         print("Evaluating best detector on validation data")
#         t0 = time.time()
#         self.val_results = self.evaluator_test.evaluate([self.best_config], [self.best_detector])[0]
#         t1 = time.time()
#         print(f"Val result ({self.metric}): {self.val_results[f'{self.metric}_val'].values}")
#         print(f"Total time: {t1 - t0:.2f} seconds")
#         self.val_results["experiment_datetime"] = self.train_results["experiment_datetime"]

#         if self.hyperparam_file is not None:
#             result_file = self.hyperparam_file[:-3] + f"_opt_{self.metric}.csv"
#         else:
#             result_file = f"results_opt_{self.metric}.csv"
#         self.save_results(
#             result_file=os.path.join(self.result_folder, result_file),
#             results=pd.merge(
#                 self.train_results, 
#                 self.val_results,
#                 how="outer")
#                 # self.val_results.loc[:, self.val_results.columns.difference(cfg_cols)]
                
#             )
#         # self.save_results(
#         #     result_file=os.path.join(self.result_folder, f"results_opt_{self.metric}.csv"),
#         #     results=self.train_results
#         #         # self.val_results.loc[:, self.val_results.columns.difference(cfg_cols)]
                
#         #     )


# class MultiDetectors:
#     def __init__(self, detectors, model, device, base_config, seed_split):

#         """
#         Args:
#             detectors (list): List of detector instances.
#         """
#         self.detectors = detectors
#         self.model = model
#         self.device = device
#         self.base_config = base_config
#         self.seed_split = seed_split
#         self.latent_path = f"storage_latent/{base_config['data']['name']}_{base_config['model']['name']}_r-{base_config['data']['r']}_seed-split-{seed_split}/logits_train.pt"
        

#     def fit(self, train_dataloader):
#         """
#         Fit all detectors on the training data.
        
#         Args:
#             train_dataloader (DataLoader): DataLoader for the training data.
#         """
#         self.model.eval()

#         # all_model_preds = []
#         if os.path.exists(self.latent_path):
#             pkg = torch.load(self.latent_path, map_location="cpu")
#             all_logits = pkg["logits"].to(torch.float32)        # (N, C)
#             labels = pkg["labels"]              # (N,)
#             model_preds  = pkg["model_preds"]# (N,)
#             all_detector_labels = (model_preds != labels).float()
        
#         else:
            
#             all_model_preds = []
#             all_labels = []
#             all_logits = []
#             for epoch in range(self.n_epochs):
#                 with torch.no_grad():
#                     for inputs, targets in tqdm(train_dataloader, total=len(train_dataloader), desc="Getting Training Logits", disable=False):
#                         inputs = inputs.to(self.device)
#                         # targets = targets.to(self.device)
                    
#                         logits = self.model(inputs).cpu()  # logits: [batch_size, num_classes]
#                         model_preds = torch.argmax(logits, dim=1)

#                         # detector_labels = (model_preds != targets).float()
#                         # # all_model_preds.append(model_preds)
#                         # all_detector_labels.append(detector_labels)
#                         all_logits.append(logits)
#                         all_labels.append(targets)
#                         all_model_preds.append(model_preds)

            
            
#             # all_model_preds = torch.cat(all_model_preds, dim=0)
#             all_labels = torch.cat(all_labels, dim=0)
#             all_model_preds = torch.cat(all_model_preds, dim=0)
#             all_detector_labels = (all_model_preds != all_labels).float()
#             all_logits = torch.cat(all_logits, dim=0)

#             # AFTER (robust)
#             parent = os.path.dirname(self.latent_path)
#             os.makedirs(parent, exist_ok=True)

#             tmp = self.latent_path + ".tmp"
#             torch.save(
#                 {
#                     "logits": all_logits.cpu(),     # compact on disk
#                     "labels": all_labels.cpu().to(torch.int64),
#                     "model_preds": all_model_preds.cpu().to(torch.int64),
#                 },
#                 tmp,
#             )
#             os.replace(tmp, self.latent_path)  # atomic rename
#             del all_model_preds, all_labels

#         # Saving 

#         for dec in tqdm(self.detectors,total=len(self.detectors), desc="Fitting Detectors", disable=False):
#             dec.fit(logits=all_logits.to(dec.device), detector_labels=all_detector_labels.to(dec.device))
