
from sklearn import metrics
import numpy as np
import torch
import os

LIST_METRICS = ['fpr', 'tpr', 'thr', 'roc_auc', 'accuracy', 'aurc', 'aupr_in', 'aupr_out']

def fpr_at_fixed_tpr(fprs, tprs, thresholds, tpr_level: float = 0.95):
    
    idxs = [i for i, x in enumerate(tprs) if x >= tpr_level]
    if len(idxs) > 0:
        idx = min(idxs)
    else:
        idx = 0
    return fprs[idx], tprs[idx], thresholds[idx]

def auc_and_fpr_recall(conf, label, tpr_level: float = 0.95, result_folder=None):
    # following convention in ML we treat OOD as positive


    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values

    fprs, tprs, thresholds = metrics.roc_curve(label, conf)
    if result_folder is not None:
        # save ROC curve data
        torch.save(
            {
                "fpr": torch.from_numpy(fprs).float(),
                "tpr": torch.from_numpy(tprs).float(),
                "thresholds": torch.from_numpy(thresholds).float()
            },
            os.path.join(result_folder, "roc_curve_data.pt")   
            )
        
    
    fpr, tpr, thr = fpr_at_fixed_tpr(fprs, tprs, thresholds, tpr_level)

    auroc = metrics.auc(fprs, tprs)
    aupr_err = metrics.average_precision_score(label, conf)
    aupr_success = metrics.average_precision_score(1 - label, 1 - conf)

    return auroc, aupr_err, aupr_success, fpr, tpr, thr


# def compute_all_metrics(conf, detector_labels):

#     tpr_level = 0.95
#     auroc, aupr_in, aupr_out, fpr, tpr, thr = auc_and_fpr_recall(conf, detector_labels, tpr_level)

#     accuracy = np.mean(detector_labels)
#     aurc_value = aurc(detector_labels, conf)

#     return fpr, tpr, thr, auroc, accuracy, aurc_value, aupr_in, aupr_out
def compute_all_metrics(conf, detector_labels,  n_split = 0, weight_std=0, tpr_level: float = 0.95, result_folder=None):
    n = conf.shape[-1]
    if n_split > 1:
        n_samples_per_split = n // n_split

        results_splits = []  # will store one dict per split

        # Average scores over multiple validation splits
        for m in range(n_split):
            start_idx = m * n_samples_per_split
            end_idx = (m + 1) * n_samples_per_split if m < n_split - 1 else n

            # Slice both scores and errors on the same indices
            scores_split = conf[:, start_idx:end_idx]      # (n_methods, n_split_samples)
            errors_split = detector_labels[start_idx:end_idx]         # (n_split_samples,)

            res_m = _compute_all_metrics(
                conf=scores_split,
                detector_labels=errors_split,
                result_folder=result_folder
            )
            results_splits.append(res_m)

        # --- Aggregate metrics across splits ---
        results = {}
        for key in results_splits[0].keys():
            # Each res[key] is a list: [metric_for_run_1, ..., metric_for_run_R]
            vals = np.stack([np.asarray(res[key]) for res in results_splits], axis=0)
            # vals.shape = (n_split_val, n_runs_or_methods)
            # results[key] = vals.mean(axis=0).tolist()
            mean_vals = vals.mean(axis=0)
            std_vals = vals.std(axis=0, ddof=1)  # sample std (or ddof=0 if you prefer)

        
            results[key] = (mean_vals + weight_std * std_vals).tolist()

            # If you also want std across splits, you could store it separately:
            # std_results[key] = vals.std(axis=0).tolist()
    else:
        
        # print("score shape:", scores.shape)
        results = _compute_all_metrics(
        conf=conf,
        detector_labels=detector_labels,
        result_folder=result_folder
    )
        # print("fpr shape:", np.shape(fpr))
    return results


def _compute_all_metrics(conf, detector_labels, tpr_level: float = 0.95, result_folder=None):
    """
    conf: np.ndarray of shape (N,) or (H, N)
    detector_labels: np.ndarray of shape (N,)
    returns 8-tuple; for batched conf, each item is (H,)
    """
    # print('result_folder in compute_all_metrics:', result_folder)
    conf = np.asarray(conf)
    y = np.asarray(detector_labels).astype(int)

    # Scalar case: keep behavior identical
    if conf.ndim == 1:
        auroc, aupr_in, aupr_out, fpr, tpr, thr = auc_and_fpr_recall(conf, y, tpr_level, result_folder=result_folder)
        accuracy = float(y.mean())
        aurc_value = aurc(y, conf)
        results = {"fpr": fpr, "tpr": tpr, "thr": thr, "roc_auc": auroc,
                   "accuracy": accuracy, "aurc": aurc_value,
                   "aupr_in": aupr_in, "aupr_out": aupr_out}
        return results

    # Batched case: loop over rows (H)
    H, N = conf.shape
    # fpr = np.empty(H, dtype=float)
    # tpr = np.empty(H, dtype=float)
    # thr = np.empty(H, dtype=float)
    # auroc = np.empty(H, dtype=float)
    # accuracy = np.full(H, y.mean(), dtype=float)  # same for all rows
    # aurc_value = np.empty(H, dtype=float)
    # aupr_in = np.empty(H, dtype=float)
    # aupr_out = np.empty(H, dtype=float)
    # fpr = []
    # tpr = []
    # thr = []
    # auroc = []
    # accuracy = [y.mean()] * H  # same for all rows
    # aurc_value = []
    # aupr_in = []
    # aupr_out = []
    results = {metric: [] for metric in LIST_METRICS}
    results["accuracy"] = [y.mean()] * H  # same for all rows

    for h in range(H):
        a, ain, aout, f, t, th = auc_and_fpr_recall(conf[h], y, tpr_level, result_folder=result_folder)
        results["fpr"].append(f)
        results["tpr"].append(t)
        results["thr"].append(th)
        results["roc_auc"].append(a)
        results["aurc"].append(aurc(y, conf[h]))
        results["aupr_in"].append(ain)
        results["aupr_out"].append(aout)
            # auroc.append(a)
            # aupr_in.append(ain)
            # aupr_out.append(aout)
            # fpr.append(f)
            # tpr.append(t)
            # thr.append(th)
            # aurc_value.append(aurc(y, conf[h]))
    
    return results

def rc_curve_stats(errors, conf) -> tuple[list[float], list[float], list[float]]:
        """
        Risk–Coverage curve computation.

        Adapted from:
        https://github.com/IML-DKFZ/fd-shifts
        (file: fd_shifts/analysis/metrics.py, function: rc_curve_stats)

        """
    
        coverages = []
        risks = []

        n_errors = len(errors)
        idx_sorted = np.argsort(conf)

        coverage = n_errors
        error_sum = sum(errors[idx_sorted])

        coverages.append(coverage / n_errors)
        risks.append(error_sum / n_errors)

        weights = []

        tmp_weight = 0
        for i in range(0, len(idx_sorted) - 1):
            coverage = coverage - 1
            error_sum = error_sum - errors[idx_sorted[i]]
            selective_risk = error_sum / (n_errors - 1 - i)
            tmp_weight += 1
            if i == 0 or conf[idx_sorted[i]] != conf[idx_sorted[i - 1]]:
                coverages.append(coverage / n_errors)
                risks.append(selective_risk)
                weights.append(tmp_weight / n_errors)
                tmp_weight = 0

        # add a well-defined final point to the RC-curve.
        if tmp_weight > 0:
            coverages.append(0)
            risks.append(risks[-1])
            weights.append(tmp_weight / n_errors)

        return coverages, risks, weights


def aurc(errors, conf) -> float:
    """AURC metric function
    Adapted from:
    https://github.com/IML-DKFZ/fd-shifts
    (file: fd_shifts/analysis/metrics.py, function: aurc)
    Args:
        errors: binary array indicating whether a prediction is incorrect (1) or correct (0)
        conf: confidence scores (higher means more confident)

    Returns:
        metric value
    """
    _, risks, weights = rc_curve_stats(errors, conf)
    aurc =  (
        sum(
            [
                (risks[i] + risks[i + 1]) * 0.5 * weights[i]
                for i in range(len(weights))
            ]
        )
    )

    return aurc