from sklearn.metrics import auc, roc_curve


def fpr_at_fixed_tpr(fprs, tprs, thresholds, tpr_level: float = 0.95):
    if all(tprs < tpr_level):
        raise ValueError(f"No threshold allows for TPR at least {tpr_level}.")
    idxs = [i for i, x in enumerate(tprs) if x >= tpr_level]
    idx = min(idxs)
    return fprs[idx], tprs[idx], thresholds[idx]


def evaluate(preds, targets, scores):
    scores = scores.view(-1).detach().cpu().numpy()
    targets = targets.view(-1).detach().cpu().numpy()
    preds = preds.view(-1).detach().cpu().numpy()

    acc = (preds == targets).mean()
    train_labels = preds != targets
    fprs, tprs, thrs = roc_curve(train_labels, scores)
    roc_auc = auc(fprs, tprs)
    fpr, _, _ = fpr_at_fixed_tpr(fprs, tprs, thrs, 0.95)

    return acc, roc_auc, fpr
