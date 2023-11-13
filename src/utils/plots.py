import argparse
import os
import torch
import numpy as np
import seaborn as sns
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import pandas as pd
import eval

sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1)
sns.set_style("whitegrid", {"axes.grid": False})
sns.set_palette("colorblind")


def plot_roc_curve(scores, targets, method_names, title, fname="roc_curve.pdf"):
    markers = ["o", "x", "*", "^", "s"]
    linestyles = ["-", "--", "-.", ":", "-"]
    plt.figure(figsize=(4, 3), dpi=180)
    for i, (s, t, l) in enumerate(zip(scores, targets, method_names)):
        print(l)
        m = markers[i % len(markers)]
        ls = linestyles[i % len(linestyles)]
        fpr, tpr, _ = metrics.roc_curve(t, s)
        auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{l} (AUC={auc:.2f})", marker=m, linestyle=ls, markevery=0.1)
    plt.plot([0, 1], [0, 1], color="k", linestyle="--")
    plt.xlim([-0.01, 1.01])
    plt.ylim([0.80, 1.01])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    plt.savefig(os.path.join("images", fname))


def plot_risk_coverage_curve(scores, preds, targets, method_names, title, fname="risk_coverage_curve.pdf"):
    markers = ["o", "x", "*", "^", "s"]
    linestyles = ["-", "--", "-.", ":", "-"]
    plt.figure(figsize=(4, 3), dpi=180)
    for i, (s, p, t, l) in enumerate(zip(scores, preds, targets, method_names)):
        risks, coverages, thrs = eval.risks_coverages_selective_net(s, p, t)
        # filter coverages > 0.1
        f = 0.1
        risks = 100 * risks[coverages > f]
        coverages = coverages[coverages > f]
        m = markers[i % len(markers)]
        ls = linestyles[i % len(linestyles)]
        aurc = metrics.auc(coverages, risks)
        plt.plot(coverages, risks, marker=m, linestyle=ls, markevery=0.1, label=f"{l} (AUC={aurc:.2f})")
    plt.xlim([0.5, 1.01])
    plt.xlabel("Coverage")
    plt.ylabel(r"100 $\times$ Risk")
    plt.legend(loc="upper left")
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    plt.savefig(os.path.join("images", fname))


def plot_d_matrix(D, fname="d_matrix.pdf"):
    plt.figure(figsize=(4, 4), dpi=180)
    sns.heatmap(D, cmap="Blues")
    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    plt.savefig(os.path.join("images", fname))


def plot_confusion_matrix(preds, targets, fname="confusion_matrix.pdf"):
    # plot confusion matrix
    cm = metrics.confusion_matrix(targets, preds)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(4, 4), dpi=180)
    sns.heatmap(cm, cmap="Blues")
    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    plt.savefig(os.path.join("images", fname))


def main():
    model_name = args.model_name
    style = args.style
    r = args.r
    seed = args.seed
    df = pd.read_csv("results/results.csv")
    print(df.columns)
    # query best config for each method
    df = df.query("model_name == @model_name and style == @style and r == @r")
    # df = df.query("magnitude == 0")
    # df = df.sort_values("fpr", ascending=True)
    df = df.sort_values("auc", ascending=False)
    best_msp = df.query("method == 'msp'").iloc[0].to_dict()
    print("MSP", best_msp)
    best_odin = df.query("method == 'odin'").iloc[0].to_dict()
    print("Odin", best_odin)
    best_doctor = df.query("method == 'doctor'").iloc[0].to_dict()
    print("Doctor", best_doctor)
    best_mlp = df.query("method == 'mlp'").iloc[0].to_dict()
    print("MLP", best_mlp)
    best_ours = df.query("method == 'metric_lagrange'").iloc[0].to_dict()
    print("Ours", best_ours)
    score_configs = {
        "msp": {"temperature": 1, "magnitude": 0, "lbd": 0.0, "seed": best_msp["seed"]},
        "odin": {
            "temperature": best_odin["temperature"],
            "magnitude": best_odin["magnitude"],
            "lbd": 0.0,
            "seed": best_odin["seed"],
        },
        "doctor": {
            "temperature": best_doctor["temperature"],
            "magnitude": best_doctor["magnitude"],
            "lbd": 0.0,
            "seed": best_doctor["seed"],
        },
        "mlp": {"temperature": 1, "magnitude": 0, "lbd": 0, "seed": best_mlp["seed"]},
        "metric_lagrange": {
            "temperature": best_ours["temperature"],
            "magnitude": best_ours["magnitude"],
            "lbd": best_ours["lbd"],
            "seed": best_ours["seed"],
        },
    }
    pretty_names = {
        "msp": "MSP",
        "odin": "ODIN",
        "doctor": "DOCTOR",
        "mlp": "MLP",
        "metric_lagrange": "Ours",
        "densenet121_cifar10": "DenseNet-121 (CIFAR-10)",
        "resnet34_cifar10": "ResNet-34 (CIFAR-10)",
        "densenet121_cifar100": "DenseNet-121 (CIFAR-100)",
        "resnet34_cifar100": "ResNet-34 (CIFAR-100)",
        "ce": "CE",
    }
    score_names = []
    for sc, v in score_configs.items():
        score_names.append(
            f"{model_name}_{style}_{sc}_{v['temperature']:.1f}_{v['magnitude']:.4f}_{r}_{v['lbd']:.2f}_{v['seed']}_"
        )

    root = os.environ.get("TENSORS_DIR", "tensors/")
    preds = []
    targets = []
    bin_labels = []
    scores = []
    for sn in score_names:
        data = np.load(os.path.join(root, sn + "main.npz"))
        p = data["preds"].reshape(-1)
        t = data["targets"].reshape(-1)
        s = data["scores"].reshape(-1)
        preds.append(p)
        targets.append(t)
        scores.append(s)
        if "metric_lagrange" in sn:
            D = torch.load(os.path.join(root, sn + "D.pt"), map_location="cpu").numpy()
        bin_labels.append(preds[-1] != targets[-1])

    method_names = [pretty_names[sc] for sc in score_configs.keys()]
    fname = f"{model_name}_{style}_{r}_roc.pdf"
    title = f"{pretty_names[model_name]} ({pretty_names[style]})"
    plot_roc_curve(scores, bin_labels, method_names, title, fname=fname)
    plot_d_matrix(D, fname=fname.replace("_roc.pdf", "_d_matrix.pdf"))
    plot_risk_coverage_curve(
        scores, preds, targets, method_names, title, fname=fname.replace("_roc.pdf", "_risk_coverage.pdf")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="densenet121_cifar10", help="model name")
    parser.add_argument("--style", type=str, default="ce", help="training style")
    parser.add_argument("-r", "--r", type=int, default=2, help="ratio")
    parser.add_argument("--lbd", type=float, default=0.5, help="lambda")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    args = parser.parse_args()

    for mn in [
        "densenet121_cifar10",
        "densenet121_cifar100",
        "resnet34_cifar10",
        "resnet34_cifar100",
    ]:
        for s in [
            "ce",
            "lognorm",
            "mixup",
            "regmixup",
            "openmix",
        ]:
            for r in [
                2,
                4,
                5,
                8,
                10,
            ]:
                args.model_name = mn
                args.style = s
                args.r = r
                try:
                    main()
                except Exception as e:
                    print(e)
                    continue
