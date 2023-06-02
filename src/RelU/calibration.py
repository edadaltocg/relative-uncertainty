import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from src.RelU.main import evaluate
from src.RelU.methods import doctor

from src.utils.helpers import append_results_to_file

CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", "checkpoints/")
CHECKPOINTS_DIR = os.path.join(CHECKPOINTS_DIR, "ce/")


def expected_calibration_error(logits, true_labels, n_bins=10):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    probs = torch.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    correct = predictions.eq(true_labels)

    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = correct[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def main():
    """Calibrate a model with temperature scaling of the logits."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    lbd = args.lbd

    try:
        test_logits = torch.load(
            os.path.join(CHECKPOINTS_DIR, args.model_name, str(1), "test_logits.pt"), map_location="cpu"
        )
        test_targets = torch.load(
            os.path.join(CHECKPOINTS_DIR, args.model_name, str(1), "test_targets.pt"), map_location="cpu"
        )
    except:
        test_logits = torch.load(os.path.join(CHECKPOINTS_DIR, args.model_name, "test_logits.pt"), map_location="cpu")
        test_targets = torch.load(os.path.join(CHECKPOINTS_DIR, args.model_name, "test_targets.pt"), map_location="cpu")

        # randomly permutate the test set
    perm = torch.randperm(len(test_logits))
    test_logits = test_logits[perm]
    test_targets = test_targets[perm]

    n = len(test_logits)
    num_train_samples = n // args.r

    train_logits = test_logits[:num_train_samples]
    train_targets = test_targets[:num_train_samples]
    train_pred = torch.argmax(train_logits, dim=1)
    train_labels = (train_pred != train_targets).int()

    test_logits = test_logits[num_train_samples:]
    test_targets = test_targets[num_train_samples:]
    test_pred = torch.argmax(test_logits, dim=1)
    test_labels = (test_pred != test_targets).int()
    test_probs = torch.softmax(test_logits, dim=1)

    logits = train_logits
    labels = train_targets
    # beginning misclassification
    print("Misclassification before calibration: {:.3f}".format(torch.mean(test_labels.float())))

    # train calibration
    nll_criterion = torch.nn.CrossEntropyLoss()
    before_cal_nll = nll_criterion(logits, labels).item()
    before_cal_ece = expected_calibration_error(test_logits, test_targets).item()
    print("Before calibration - NLL: %.3f, ECE: %.3f" % (before_cal_nll, before_cal_ece))

    # doctor results
    print("Misclassification after calibration: {:.3f}".format(torch.mean(labels.float())))
    doctor_scores = doctor(test_logits)
    acc, doctor_auc, doctor_fpr = evaluate(test_pred, test_targets, doctor_scores)
    columns = [
        "model",
        "calibration",
        "params",
        "before_cal_nll",
        "before_cal_ece",
        "after_cal_nll",
        "after_cal_ece",
        "r",
        "seed",
        "acc",
        "method",
        "fpr",
        "auc",
        "lbd",
    ]
    row = [
        args.model_name,
        args.calibration,
        None,
        before_cal_nll,
        before_cal_ece,
        None,
        None,
        args.r,
        args.seed,
        acc,
        "uncal_doctor",
        doctor_fpr,
        doctor_auc,
        lbd,
    ]
    results = dict(zip(columns, row))
    append_results_to_file(results, "results/calibration.csv")

    train_probs = torch.softmax(train_logits, dim=1)
    train_probs_pos = train_probs[train_labels == 0]
    train_probs_neg = train_probs[train_labels == 1]
    d_pos = torch.einsum("ij,ik->ijk", train_probs_pos, train_probs_pos).mean(dim=0)
    d_neg = torch.einsum("ij,ik->ijk", train_probs_neg, train_probs_neg).mean(dim=0)
    d = (1 - lbd) * d_pos - lbd * d_neg
    d = torch.tril(d, diagonal=-1)
    d = d + d.T
    score = torch.diag(test_probs @ d @ test_probs.T)
    acc, outer_auc, outer_fpr = evaluate(test_pred, test_targets, score)
    row = [
        args.model_name,
        args.calibration,
        None,
        before_cal_nll,
        before_cal_ece,
        None,
        None,
        args.r,
        args.seed,
        acc,
        "uncal_outer",
        outer_fpr,
        outer_auc,
        lbd,
    ]
    results = dict(zip(columns, row))
    append_results_to_file(results, "results/calibration.csv")

    def temperature_scale(logits, labels):
        temperature = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=1000)
        temperature = temperature.to(logits.device)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        return [temperature.cpu()]

    def temperature_op(logits, params):
        return logits / params[0]

    def platt_scaling(logits, labels):
        alpha = torch.nn.Parameter(torch.randn(logits.shape[1], dtype=logits.dtype), requires_grad=True)
        beta = torch.nn.Parameter(torch.randn(logits.shape[1], dtype=logits.dtype), requires_grad=True)
        optimizer = optim.LBFGS([alpha, beta], lr=0.01, max_iter=1000)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(torch.softmax(logits / alpha + beta, dim=1), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        return [alpha, beta]

    def platt_op(logits, params):
        return logits / params[0] + params[1]

    calibration = None
    forward_op = None
    if args.calibration == "temperature":
        calibration = temperature_scale
        forward_op = temperature_op
    elif args.calibration == "platt":
        calibration = platt_scaling
        forward_op = platt_op
    assert calibration is not None
    assert forward_op is not None

    # params = calibration(logits.to(device), labels.to(device))
    params = calibration(logits, labels)

    # Calculate NLL and ECE after calibration
    logits = test_logits
    labels = test_targets
    cal_logits = forward_op(logits, params)
    after_cal_nll = nll_criterion(cal_logits, labels).item()
    after_cal_ece = expected_calibration_error(cal_logits, labels).item()
    print("After calibration - NLL: %.3f, ECE: %.3f" % (after_cal_nll, after_cal_ece))
    print("params", params)

    # doctor results
    probs = torch.softmax(cal_logits, dim=1)
    pred = torch.argmax(cal_logits, dim=1)
    labels = (pred != test_targets).int()
    print("Misclassification after calibration: {:.3f}".format(torch.mean(labels.float())))
    doctor_scores = doctor(cal_logits)
    acc, doctor_auc, doctor_fpr = evaluate(pred, test_targets, doctor_scores)

    # outer product results
    train_logits = forward_op(train_logits, params)
    train_probs = torch.softmax(train_logits, dim=1)

    train_probs = torch.softmax(train_logits, dim=1)
    train_probs_pos = train_probs[train_labels == 0]
    train_probs_neg = train_probs[train_labels == 1]
    d_pos = torch.einsum("ij,ik->ijk", train_probs_pos, train_probs_pos).mean(dim=0)
    d_neg = torch.einsum("ij,ik->ijk", train_probs_neg, train_probs_neg).mean(dim=0)
    d = -(1 - lbd) * d_pos + lbd * d_neg
    d = torch.tril(d, diagonal=-1)
    d = d + d.T
    d = torch.relu(d) + 1e-8
    d = d / d.norm()
    score = torch.diag(probs @ d @ probs.T)
    acc, outer_auc, outer_fpr = evaluate(pred, test_targets, score)

    # save best temperature to file
    params = str([p.item() for p in params])
    row = [
        args.model_name,
        args.calibration,
        params,
        before_cal_nll,
        before_cal_ece,
        after_cal_nll,
        after_cal_ece,
        args.r,
        args.seed,
        acc,
        "cal_doctor",
        doctor_fpr,
        doctor_auc,
        lbd,
    ]
    results = dict(zip(columns, row))
    append_results_to_file(results, "results/calibration.csv")

    row = [
        args.model_name,
        args.calibration,
        params,
        before_cal_nll,
        before_cal_ece,
        after_cal_nll,
        after_cal_ece,
        args.r,
        args.seed,
        acc,
        "cal_outer",
        outer_fpr,
        outer_auc,
        lbd,
    ]
    results = dict(zip(columns, row))
    append_results_to_file(results, "results/calibration.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="densenet121_cifar10", help="model name")
    parser.add_argument("--calibration", type=str, default="temperature")
    parser.add_argument("--r", type=int, default=10, help="ratio")
    parser.add_argument("--lbd", type=float, default=0.0, help="lambda")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    args = parser.parse_args()

    main()
