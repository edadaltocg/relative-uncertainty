#!/usr/bin/env python

import os
from copy import deepcopy
from typing import Dict, Tuple

import torch

from code.evaluators import HyperparamsSearch
from code.utils.datasets import get_dataset
from code.utils.datasets.dataloader import prepare_ablation_dataloaders
from code.utils.helper import setup_seeds
from code.utils.models import get_model
from code.utils.config import Config


N_THREADS = 1
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_THREADS)

torch.set_num_threads(N_THREADS)
torch.set_num_interop_threads(N_THREADS)

CHECKPOINTS_DIR_BASE = os.environ.get("CHECKPOINTS_DIR", "checkpoints/")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
RESULTS_ROOT = "results_scikit"
SEED_SPLIT = 9
BASE_SEED = 1
CAL_SAMPLES = 5000
TEST_SAMPLES = 5000


SCIKIT_METHODS = [
    "dummy_classifier",
    "logistic_regression",
    "logistic_regression_cv",
    "k_neighbors_classifier",
    "decision_tree_classifier",
    "extra_tree_classifier",
    "random_forest_classifier",
    "extra_trees_classifier",
    "gradient_boosting_classifier",
    "hist_gradient_boosting_classifier",
    "ada_boost_classifier",
    "bagging_classifier",
    "gaussian_nb",
    "multinomial_nb",
    "bernoulli_nb",
    "complement_nb",
    "linear_discriminant_analysis",
    "quadratic_discriminant_analysis",
    "mlp_classifier",
    "calibrated_classifier_cv",
]


MODEL_CONFIGS: Dict[Tuple[str, str], str] = {
    ("cifar10", "resnet34"): "configs/models/cifar10_resnet34.yml",
    ("cifar10", "densenet121"): "configs/models/cifar10_densenet121.yml",
    ("cifar100", "resnet34"): "configs/models/cifar100_resnet34.yml",
    ("cifar100", "densenet121"): "configs/models/cifar100_densenet121.yml",
}

LATENT_BASE: Dict[Tuple[str, str], str] = {
    ("cifar10", "resnet34"): "latent/ablation/cifar10_resnet34_n_cal",
    ("cifar10", "densenet121"): "latent/ablation/cifar10_densenet121_n_cal",
    ("cifar100", "resnet34"): "latent/ablation/cifar100_resnet34_n_cal",
    ("cifar100", "densenet121"): "latent/ablation/cifar100_densenet121_n_cal",
}


def build_data_cfg(dataset_name: str) -> Dict:
    if dataset_name not in {"cifar10", "cifar100"}:
        raise ValueError(f"Unsupported dataset '{dataset_name}'")

    num_classes = 10 if dataset_name == "cifar10" else 100
    return {
        "name": dataset_name,
        "num_classes": num_classes,
        "n_samples": {"res": 0, "cal": CAL_SAMPLES, "test": TEST_SAMPLES},
        "seed_split": [SEED_SPLIT],
        "batch_size_train": 252,
        "batch_size_test": 252,
    }


def build_detection_cfg(estimator_key: str) -> Dict:
    return {
        "name": "scikit",
        "postprocessor_args": {
            "space": "probits",
            "temperature": 1.0,
            "reorder_embs": True,
            "estimator": estimator_key,
        },
        "experience_args": {
            "n_folds": 5,
            "n_epochs": {"res": None, "cal": 1},
            "transform": {"res": None, "cal": "test"},
        },
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    for (dataset_name, model_name), model_cfg_path in MODEL_CONFIGS.items():
        data_cfg = build_data_cfg(dataset_name)
        model_cfg = Config(model_cfg_path)

        print(f"\n=== Dataset: {dataset_name} | Model: {model_name} ===")

        dataset = get_dataset(
            dataset_name=data_cfg["name"],
            model_name=model_cfg["model_name"],
            root=DATA_DIR,
            preprocess=model_cfg["preprocessor"],
            shuffle=False,
        )

        model = get_model(
            model_name=model_cfg["model_name"],
            dataset_name=data_cfg["name"],
            n_classes=data_cfg["num_classes"],
            model_seed=model_cfg["seed"],
            checkpoint_dir=os.path.join(CHECKPOINTS_DIR_BASE, model_cfg["preprocessor"]),
        )
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        setup_seeds(BASE_SEED, SEED_SPLIT)
        res_loader, cal_loader, test_loader = prepare_ablation_dataloaders(
            dataset=dataset,
            seed_split=SEED_SPLIT,
            n_res=data_cfg["n_samples"]["res"],
            n_cal=data_cfg["n_samples"]["cal"],
            n_test=data_cfg["n_samples"]["test"],
            batch_size_train=data_cfg["batch_size_train"],
            batch_size_test=data_cfg["batch_size_test"],
            cal_transform="test",
            res_transform=None,
            data_name=data_cfg["name"],
            model_name=model_cfg["model_name"],
        )

        latent_base = LATENT_BASE[(dataset_name, model_name)]
        latent_dir = os.path.join(latent_base, f"seed-split-{SEED_SPLIT}")
        latent_paths = {
            "res": os.path.join(
                latent_dir,
                f"res_n-samples-{data_cfg['n_samples']['res']}_transform-None_n-epochs-None.pt",
            ),
            "cal": os.path.join(
                latent_dir,
                f"cal_n-samples-{data_cfg['n_samples']['cal']}_transform-test_n-epochs-1.pt",
            ),
            "test": os.path.join(
                latent_dir, f"test_n-samples-{data_cfg['n_samples']['test']}.pt"
            ),
        }

        for estimator_key in SCIKIT_METHODS:
            detection_cfg = build_detection_cfg(estimator_key)
            result_folder = os.path.join(
                RESULTS_ROOT,
                estimator_key,
                f"{dataset_name}_{model_name}",
                f"seed-split-{SEED_SPLIT}",
            )
            os.makedirs(result_folder, exist_ok=True)

            print(f" -> {estimator_key}")
            evaluator = HyperparamsSearch(
                model=model,
                cfg_detection=deepcopy(detection_cfg),
                cfg_dataset=deepcopy(data_cfg),
                device=device,
                res_loader=res_loader,
                cal_loader=cal_loader,
                test_loader=test_loader,
                result_folder=result_folder,
                metric="fpr",
                latent_paths=latent_paths,
                seed_split=SEED_SPLIT,
                mode="evaluation",
                verbose=False,
            )
            evaluator.run()

        # release GPU memory between model switches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

