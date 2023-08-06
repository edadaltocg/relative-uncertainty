from sklearn.metrics import auc, roc_curve
import numpy as np


def fpr_at_fixed_tpr(fprs, tprs, thresholds, tpr_level: float = 0.95):
    if all(tprs < tpr_level):
        raise ValueError(f"No threshold allows for TPR at least {tpr_level}.")
    idxs = [i for i, x in enumerate(tprs) if x >= tpr_level]
    if len(idxs) > 0:
        idx = min(idxs)
    else:
        idx = 0
    return fprs[idx], tprs[idx], thresholds[idx]


def hard_coverage(scores, thr: float):
    return (scores <= thr).mean()


def selective_net_risk(scores, pred, targets, thr: float):
    covered_idx = scores <= thr
    return np.sum(pred[covered_idx] != targets[covered_idx]) / np.sum(covered_idx)


def risks_coverages_selective_net(scores, pred, targets, sort=True):
    """
    Returns:

        risks, coverages, thrs
    """
    # this function is slow
    risks = []
    coverages = []
    thrs = []
    for thr in np.unique(scores):
        risks.append(selective_net_risk(scores, pred, targets, thr))
        coverages.append(hard_coverage(scores, thr))
        thrs.append(thr)
    risks = np.array(risks)
    coverages = np.array(coverages)
    thrs = np.array(thrs)

    # sort by coverages
    if sort:
        sorted_idx = np.argsort(coverages)
        risks = risks[sorted_idx]
        coverages = coverages[sorted_idx]
        thrs = thrs[sorted_idx]
    return risks, coverages, thrs


def evaluate(preds, targets, scores):
    scores = scores.view(-1).detach().cpu().numpy()
    targets = targets.view(-1).detach().cpu().numpy()
    preds = preds.view(-1).detach().cpu().numpy()

    model_acc = (preds == targets).mean()
    train_labels = preds != targets
    fprs, tprs, thrs = roc_curve(train_labels, scores)
    roc_auc = auc(fprs, tprs)

    fpr, _, _ = fpr_at_fixed_tpr(fprs, tprs, thrs, 0.95)
    risks, coverages, _ = risks_coverages_selective_net(scores, preds, targets)
    aurc = auc(coverages, risks)
    return model_acc, roc_auc, fpr, aurc
