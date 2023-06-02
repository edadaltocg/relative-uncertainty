from typing import Any, Dict, Type

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder, ImageNet

datasets_registry: Dict[str, Any] = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "svhn": SVHN,
    "imagenet": ImageNet,
}


def get_dataset(dataset_name: str, root: str, **kwargs) -> Dataset:
    if dataset_name is not None:
        return datasets_registry[dataset_name](root, **kwargs)
    else:
        try:
            return ImageFolder(root, **kwargs)
        except:
            raise ValueError(f"Dataset {root} not found")


def get_dataset_cls(dataset_name: str) -> Type[Dataset]:
    try:
        return datasets_registry[dataset_name]
    except:
        return ImageFolder


def get_datasets_names():
    return list(datasets_registry.keys())
