import os

N_THREADS = 1
os.environ["OMP_NUM_THREADS"]   = f"{N_THREADS}"
os.environ["MKL_NUM_THREADS"]   = f"{N_THREADS}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{N_THREADS}"
os.environ["NUMEXPR_NUM_THREADS"]  = f"{N_THREADS}"
import json
import torch

from code.utils.models import get_model

from code.utils.datasets import  get_dataset
from itertools import product

import numpy as np

import pandas as pd
import joblib
import random
from typing import Dict, Any, List, Tuple
import warnings
from copy import deepcopy
from code.utils.detection.methods import EvaluatorAblation
from code.utils.helper import make_config_list, _prepare_config_for_results, setup_seeds
from code.utils.datasets.dataloader import prepare_ablation_dataloaders
from code.utils.config import Config

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*force_all_finite.*"
)


# N_THREADS = 1
# os.environ["OMP_NUM_THREADS"]   = f"{N_THREADS}"
# os.environ["MKL_NUM_THREADS"]   = f"{N_THREADS}"
# os.environ["OPENBLAS_NUM_THREADS"] = f"{N_THREADS}"
# os.environ["NUMEXPR_NUM_THREADS"]  = f"{N_THREADS}"

torch.set_num_threads(N_THREADS)
torch.set_num_interop_threads(N_THREADS)

# # 4. Verify settings
print("OMP_NUM_THREADS =", os.getenv("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS =", os.getenv("MKL_NUM_THREADS"))
print("torch.get_num_threads() =", torch.get_num_threads())
print("torch.get_num_interop_threads() =", torch.get_num_interop_threads())

CHECKPOINTS_DIR_BASE = os.environ.get("CHECKPOINTS_DIR", "checkpoints/")
DATA_DIR = os.environ.get("DATA_DIR", "./data")


        

def main(seed_split, n_cal):

 
    dataset = get_dataset(
                dataset_name=data_cfg["name"], 
                model_name=model_cfg["model_name"], 
                root=DATA_DIR,
                preprocess=model_cfg["preprocessor"],
                shuffle=False)
                

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    model = get_model(model_name=model_cfg["model_name"], 
                    dataset_name=data_cfg["name"],
                    n_classes=data_cfg["num_classes"],
                    model_seed=model_cfg["seed"],
                    checkpoint_dir = os.path.join(CHECKPOINTS_DIR_BASE, model_cfg["preprocessor"]),
                    )
    
    model = model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)  # freeze permanently
        
    # # for name, m in model.named_modules():
    # # #     print(name, m)
    # # exit()
    # model = None
 
    print(f"Running seed split {seed_split}...")

    setup_seeds(args.seed, seed_split)

    res_loader, cal_loader, test_loader = prepare_ablation_dataloaders(
        dataset = dataset,
        seed_split=seed_split, 
        n_res=data_cfg["n_samples"]["res"],
        n_cal=n_cal, 
        n_test=data_cfg["n_samples"]["test"],
        batch_size_train=data_cfg["batch_size_train"], 
        batch_size_test=data_cfg["batch_size_test"], 
        cal_transform=cfg_detection["experience_args"]["transform"]["cal"], 
        res_transform=cfg_detection["experience_args"]["transform"]["res"], 
        data_name=data_cfg["name"],
        model_name=model_cfg["model_name"],
    )
    # print("res_loader 0", res_loader.dataset[0])
    # print("cal_loader 0", cal_loader.dataset[0])
    # print("test_loader 0", test_loader.dataset[0])
    # exit()
    # ds = cal_loader.dataset  # likely a Subset
    # first10 = [ds[i] for i in range(10)]  # integer indexing only
    # torch.save(first10, "test_my_inputs.pt")
    # exit()


    latent_dir = os.path.join(args.latent_dir, f"seed-split-{seed_split}")
    latent_paths = {
        "res": os.path.join(latent_dir, f"res_n-samples-{data_cfg['n_samples']['res']}_transform-{cfg_detection['experience_args']['transform']['res']}_n-epochs-{data_cfg['n_epochs']['res']}.pt"),
        "cal": os.path.join(latent_dir, f"cal_n-samples-{n_cal}_transform-{cfg_detection['experience_args']['transform']['cal']}_n-epochs-{data_cfg['n_epochs']['cal']}.pt"),
        # "cal": "storage_latent/______cifar100_resnet34_ce_r-0.5_seed-split-9/probits_calib_n-epochs1_transform-test.pt",
        "test": os.path.join(latent_dir, f"test_n-samples-{data_cfg['n_samples']['test']}.pt"),
    }   

    evaluator = EvaluatorAblation(
        model=model,
        cfg_detection=cfg_detection, 
        cfg_dataset=data_cfg,
        device=device, 
        res_loader=res_loader,
        cal_loader=cal_loader, 
        test_loader=test_loader,
        result_folder=args.root_dir,
        metric ="fpr",
        latent_paths=latent_paths,
        n_cal=n_cal,
        seed_split=seed_split,
        # is_relu=True
    )
    evaluator.run()

            
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_dataset', 
        type=str, 
        default="configs/datasets/cifar10/cifar10_ablation.yml",
        help='Path to the dataset config file'
        )
    parser.add_argument(
        '--config_model', 
        type=str, 
        default="configs/models/cifar10_resnet34.yml",
        help='Path to the model config file'
        )
    parser.add_argument(
        '--config_detection', 
        type=str, 
        default="configs/detection/clustering.yml",
        help='Path to the detection config file'
        )
    parser.add_argument(
        '--root_dir', 
        type=str, 
        default="./results/ablation/cifar10_n_cal/",
        help='Root directory to save results'
        )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=1,
        help='Random seed for reproducibility'
        )
    parser.add_argument(
        '--gpu_id', 
        type=int, 
        default=0,
        help='GPU ID to use'
        )
    parser.add_argument(
        '--latent_dir', 
        type=str, 
        default="./latent/ablation/cifar10_n_cal/",
        help='Directory to save latent representations'
        )
    args = parser.parse_args()

    data_cfg = Config(args.config_dataset)
    model_cfg = Config(args.config_model)
    cfg_detection = Config(args.config_detection)


    
    # seed_splits = data_cfg["seed_split"][-1]
    # n_cal1 = data_cfg["n_samples"]["cal1"][-1]
    # print(data_cfg["seed_split"])
    # print(data_cfg["n_samples"]["cal1"])
    # for seed_split, n_cal in product(data_cfg["seed_split"], data_cfg["n_samples"]["cal"]):
    #     print(f"seed_split: {seed_split}, n_cal: {n_cal}")
    #     main(seed_split, n_cal)
    main(9, 15000)



