import torch
import openood
from openood.networks import ResNet18_32x32
from openood.networks import ResNet50, ViT_B_16
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
from torch.hub import load_state_dict_from_url
from openood.evaluation_api.datasets import (
    get_id_ood_dataloader,
    get_default_preprocessor,
)
from torch.utils.data import DataLoader
from pprint import pprint
from tqdm import tqdm
import argparse

import os

def store_latents(
    dataset, resnet=True, dataroot="./data", device="cuda:0", batch_size=5000
):
    if dataset in ["cifar10", "cifar100"]:
        models = [
            ResNet18_32x32(num_classes=100 if dataset == "cifar100" else 10)
            for _ in range(3)
        ]
        [
            model.load_state_dict(
                torch.load(
                    f"./OpenOOD/results/{dataset}_resnet18_32x32_base_e100_lr0.1_default/s{i}/best.ckpt"
                )
            )
            for i, model in enumerate(models)
        ]

    elif resnet:
        net = ResNet50()
        weights = ResNet50_Weights.IMAGENET1K_V1
        net.load_state_dict(load_state_dict_from_url(weights.url))
        models = [net]

    else:
        model = ViT_B_16()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model.load_state_dict(load_state_dict_from_url(weights.url))
        models = [model]
    preprocessor = get_default_preprocessor(dataset)
    id_ood_loader = get_id_ood_dataloader(
        dataset, dataroot, preprocessor, "histo", batch_size=batch_size
    )
    del id_ood_loader["csid"]
    id_ood_latents = [
        {
            k: {
                k_: (
                    [] if isinstance(v_, DataLoader) else {k__: [] for k__ in v_.keys()}
                )
                for k_, v_ in v.items()
            }
            for k, v in id_ood_loader.items()
        }
        for _ in models
    ]
    for i, model in enumerate(models):
        for id_ood_val, dsets in id_ood_loader.items():
            model.eval()
            model = model.to(device)
            with torch.no_grad():
                for dname, dset in dsets.items():
                    if isinstance(dset, dict):
                        for k, v in dset.items():
                            for sample in tqdm(
                                v, desc=f"model:{i} {id_ood_val} {dname} {k}"
                            ):
                                # latents = model(
                                #     sample["data"].to(device), return_feature=True
                                # )[1].cpu()
                                logits = model(
                                    sample["data"].to(device)).cpu()
                                id_ood_latents[i][id_ood_val][dname][k].append(logits)
                    else:
                        for sample in tqdm(
                            dset, desc=f"model:{i} {id_ood_val} {dname}"
                        ):
                            logits = model(
                                sample["data"].to(device)).cpu()
                            id_ood_latents[i][id_ood_val][dname].append(logits)
            # concatenate all the latents
        id_ood_latents[i] = {
            k: {
                k_: (
                    torch.cat(v_, dim=0)
                    if isinstance(v_, list)
                    else {k__: torch.cat(v__) for k__, v__ in v_.items()}
                )
                for k_, v_ in v.items()
            }
            for k, v in id_ood_latents[i].items()
        }
    os.makedirs("./storage", exist_ok=True)
    if resnet:
        torch.save(id_ood_latents, f"./storage/{dataset}_custom_latents.pth")
    else:
        torch.save(id_ood_latents, f"./storage/{dataset}_vit_latents.pth")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    all_datasets = ["cifar10", "cifar100", "imagenet", "imagenet_vit"]
    cifar_datasets = ["cifar10", "cifar100"]
    args.add_argument("--datasets", type=str, default="cifar")
    args.add_argument("--batch_size", type=int, default=5000)
    args.add_argument("--device", type=int, default=0)
    args = args.parse_args()
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    if args.datasets == "cifar":
        datasets = cifar_datasets
    else:
        datasets = all_datasets
    for dataset in datasets:
        if "vit" in dataset:
            store_latents(
                "imagenet", resnet=False, device=device, batch_size=args.batch_size
            )
        else:
            store_latents(dataset, device=device, batch_size=args.batch_size)
