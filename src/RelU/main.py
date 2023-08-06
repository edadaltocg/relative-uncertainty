import os
import argparse
import itertools
import json
import random
import numpy as np
from torch.autograd import Variable
import torch
import torch.utils.data
from tqdm import tqdm
from src.RelU.methods import add_dropout_layer, enable_dropout, get_method
from src.utils.datasets import get_dataset
from src.utils.helpers import append_results_to_file
from src.utils.models import _get_openmix_cifar100_transforms, _get_openmix_cifar10_transforms, get_model_essentials
import torchvision
import timm
import timm.data
from src.utils.eval import evaluate

DATA_DIR = os.environ.get("DATA_DIR", "data/")


def get_model_and_dataset(model_name):
    CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", "checkpoints/")
    CHECKPOINTS_DIR = os.path.join(CHECKPOINTS_DIR, args.style)

    # load model
    if len(model_name.split("_")) == 1 or model_name.split("_")[0] == "tv":
        model = timm.create_model(model_name, pretrained=True)

        # transform
        data_config = timm.data.resolve_data_config(model.default_cfg)
        data_config["is_training"] = False
        test_transform = timm.data.create_transform(**data_config)

        root = os.environ["IMAGENET_ROOT"]
        dataset = torchvision.datasets.ImageNet(root=root, split="val", transform=test_transform)
    else:
        model_essentials = get_model_essentials(model_name)
        model = model_essentials["model"]
        test_transform = model_essentials["test_transforms"]
        if "openmix" in args.style and "cifar10" in model_name:
            _, test_transform = _get_openmix_cifar10_transforms()
        if "openmix" in args.style and "cifar100" in model_name:
            _, test_transform = _get_openmix_cifar100_transforms()
        try:
            w = torch.load(
                os.path.join(CHECKPOINTS_DIR, args.model_name, str(args.seed), "best.pth"), map_location="cpu"
            )
        except:
            w = torch.load(os.path.join(CHECKPOINTS_DIR, args.model_name, "last.pt"), map_location="cpu")
        w = {k.replace("module.", ""): v for k, v in w.items()}
        if "openmix" in args.style:
            # add one class to model output
            model._modules[list(model._modules.keys())[-1]] = torch.nn.Linear(
                model._modules[list(model._modules.keys())[-1]].in_features,
                model._modules[list(model._modules.keys())[-1]].out_features + 1,
            )

        model.load_state_dict(w)
        # load data
        dataset_name = model_name.split("_")[-1]
        dataset = get_dataset(
            dataset_name=dataset_name, root=DATA_DIR, train=False, transform=test_transform, download=True
        )

    return model, dataset


def main(temperature, magnitude):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    model, dataset = get_model_and_dataset(args.model_name)
    model = model.to(device)
    model.eval()
    if args.method == "mc_dropout":
        add_dropout_layer(model)
        enable_dropout(model)

    # randomly permutate dataset
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    dataset = torch.utils.data.Subset(dataset, indices)

    # split data
    dataset_name = args.model_name.split("_")[-1]
    num_classes = {"cifar10": 10, "svhn": 10, "cifar100": 100, "imagenet": 1000}[dataset_name]
    n = len(dataset)
    num_train_samples = n // args.r
    train_dataset = torch.utils.data.Subset(dataset, range(0, num_train_samples))
    test_dataset = torch.utils.data.Subset(dataset, range(num_train_samples, n))
    val_dataset = torch.utils.data.Subset(test_dataset, range(0, len(test_dataset) // 5))
    test_dataset = torch.utils.data.Subset(test_dataset, range(len(test_dataset) // 5, len(test_dataset)))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=6, prefetch_factor=2
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=6, prefetch_factor=2
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=6, prefetch_factor=2
    )

    # get train probs
    method = get_method(args.method, temperature=temperature, model=model, lbd=args.lbd, num_classes=num_classes)
    method.fit(train_dataloader, val_dataloader)

    test_preds, test_targets, test_scores = [], [], []
    for inputs, targets in tqdm(test_dataloader, total=len(test_dataloader)):
        inputs = inputs.to(device)
        pred = None
        if magnitude > 0:
            inputs = Variable(inputs, requires_grad=True)
            # compute output
            outputs = model(inputs)
            if args.style == "openmix":
                outputs = outputs[:, :-1]
            pred = torch.argmax(outputs, dim=1)
            # compute perturbation
            scores = method(outputs)
            scores = torch.log(scores)
            scores.sum().backward()
            inputs = inputs - magnitude * torch.sign(-inputs.grad)
            # inputs = torch.clamp(inputs, 0, 1)
            inputs = Variable(inputs, requires_grad=False)

        with torch.no_grad():
            if args.method == "mc_dropout":
                k=10
                logits = torch.zeros(k, inputs.shape[0], num_classes).to(device)
                for i in range(k):
                    logits[i] = model(inputs)
                logits = logits.mean(dim=0)
                scores = method(logits)
            else:
                logits = model(inputs)
                if args.style == "openmix":
                    logits = logits[:, :-1]
                scores = method(logits)

        if pred is None:
            pred = torch.argmax(logits, dim=1)

        test_preds.append(pred.cpu())
        test_targets.append(targets.cpu())
        test_scores.append(scores.cpu())

    test_preds = torch.concat(test_preds)
    test_targets = torch.concat(test_targets)
    test_scores = torch.concat(test_scores)

    model_acc, roc_auc, fpr, aurc = evaluate(test_preds, test_targets, test_scores)
    results = {
        "model_name": args.model_name,
        "style": args.style,
        "temperature": temperature,
        "magnitude": magnitude,
        "method": args.method,
        "accuracy": model_acc,
        "fpr": fpr,
        "auc": roc_auc,
        "aurc": aurc,
        "r": args.r,
        "lbd": args.lbd,
        "seed": args.seed,
    }

    print(json.dumps(results, indent=2))
    append_results_to_file(results, "results/results.csv")

    # save scores
    root = os.environ.get("TENSORS_DIR", "tensors/")
    os.makedirs(root, exist_ok=True)
    save_path = os.path.join(
        root,
        f"{args.model_name}_{args.style}_{args.method}_{temperature:.1f}_{magnitude:.4f}_{args.r}_{args.lbd:.2f}_{args.seed}_",
    )
    save_obj = dict(
        preds=test_preds.detach().cpu().numpy(),
        targets=test_targets.detach().cpu().numpy(),
        scores=test_scores.detach().cpu().numpy(),
    )
    np.savez(save_path + "main.npz", **save_obj)

    # torch.save(test_preds, save_path + "preds.pt")
    # torch.save(test_targets, save_path + "targets.pt")
    # torch.save(test_scores, save_path + "scores.pt")

    # save D matrix
    if args.method == "metric_lagrange":
        D = method.export_matrix()
        torch.save(D, save_path + "D.pt")

    print(f"Files saved to memory for config: {args}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="densenet121_cifar10", help="model name")
    parser.add_argument("--method", type=str, default="msp", help="method")
    parser.add_argument("--style", type=str, default="ce", help="training style")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-r", "--r", type=int, default=10, help="ratio")
    parser.add_argument("--lbd", type=float, default=0.5, help="lambda")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    args = parser.parse_args()

    TEMPERATURES = [0.2, 0.4, 0.6, 0.8, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2]
    MAGNITUDES = [
        0,
        0.0005,
        0.001,
        0.0015,
        0.002,
        0.0025,
        0.003,
        0.0035,
        0.0040,
        # 0.0036,
        # 0.0038,
        # 0.004,
        # 0.0042,
        # 0.0044,
        # 0.0046,
        # 0.0048,
        # 0.005,
        # 0,
    ]

    if args.method == "msp" or args.method == "mlp" or args.method == "mc_dropout":
        main(temperature=1.0, magnitude=0.0)
    else:
        for temperature, magnitude in itertools.product(TEMPERATURES, MAGNITUDES):
            main(temperature, magnitude)
