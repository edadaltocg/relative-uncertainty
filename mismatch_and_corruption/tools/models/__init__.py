import logging
from typing import Any, Dict

from torchvision import transforms

from . import densenet, resnet, vgg

logger = logging.getLogger(__name__)


def _get_default_cifar10_transforms():
    statistics = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize(*statistics),
        ]
    )
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(*statistics),
        ]
    )
    return train_transforms, test_transforms


def _get_default_cifar100_transforms():
    statistics = ((0.4914, 0.482158, 0.446531), (0.247032, 0.243486, 0.261588))
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize(*statistics),
        ]
    )
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(*statistics),
        ]
    )
    return train_transforms, test_transforms


def _get_default_svhn_transforms():
    statistics = ((0.437682, 0.44377, 0.472805), (0.19803, 0.201016, 0.197036))
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize(*statistics),
        ]
    )
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(*statistics),
        ]
    )
    return train_transforms, test_transforms


def DenseNet121Cifar10(features_nodes=None):
    model = densenet.DenseNet121Small(10)
    train_transforms, test_transforms = _get_default_cifar10_transforms()
    input_dim = (3, 32, 32)
    if features_nodes is None:
        features_nodes = {"view": "features", "linear": "linear"}
    return {
        "model": model,
        "features_nodes": features_nodes,
        "input_dim": input_dim,
        "test_transforms": test_transforms,
        "train_transforms": train_transforms,
    }


def DenseNet121Cifar100(features_nodes=None):
    model = densenet.DenseNet121Small(100)
    train_transforms, test_transforms = _get_default_cifar100_transforms()
    input_dim = (3, 32, 32)
    if features_nodes is None:
        features_nodes = {"view": "features", "linear": "linear"}
    return {
        "model": model,
        "features_nodes": features_nodes,
        "input_dim": input_dim,
        "test_transforms": test_transforms,
        "train_transforms": train_transforms,
    }


def DenseNet121SVHN(features_nodes=None):
    model = densenet.DenseNet121Small(10)
    train_transforms, test_transforms = _get_default_svhn_transforms()
    input_dim = (3, 32, 32)
    if features_nodes is None:
        features_nodes = {"view": "features", "linear": "linear"}
    return {
        "model": model,
        "features_nodes": features_nodes,
        "input_dim": input_dim,
        "test_transforms": test_transforms,
        "train_transforms": train_transforms,
    }


def VGG16Cifar10(features_nodes=None):
    model = vgg.VGG16(10)
    train_transforms, test_transforms = _get_default_cifar10_transforms()
    input_dim = (3, 32, 32)
    if features_nodes is None:
        features_nodes = {"view": "features", "classifier": "linear"}
    return {
        "model": model,
        "features_nodes": features_nodes,
        "input_dim": input_dim,
        "test_transforms": test_transforms,
        "train_transforms": train_transforms,
    }


def VGG16Cifar100(features_nodes=None):
    model = vgg.VGG16(100)
    train_transforms, test_transforms = _get_default_cifar100_transforms()
    input_dim = (3, 32, 32)
    if features_nodes is None:
        features_nodes = {"view": "features", "classifier": "linear"}
    return {
        "model": model,
        "features_nodes": features_nodes,
        "input_dim": input_dim,
        "test_transforms": test_transforms,
        "train_transforms": train_transforms,
    }


def VGG16SVHN(features_nodes=None, download=False, url=None, *args, **kwargs):
    model = vgg.VGG16(10)
    train_transforms, test_transforms = _get_default_svhn_transforms()
    input_dim = (3, 32, 32)
    if features_nodes is None:
        features_nodes = {"view": "features", "classifier": "linear"}
    return {
        "model": model,
        "features_nodes": features_nodes,
        "input_dim": input_dim,
        "test_transforms": test_transforms,
        "train_transforms": train_transforms,
    }


def ResNet34Cifar10(features_nodes=None):
    model = resnet.ResNet34(10)
    train_transforms, test_transforms = _get_default_cifar10_transforms()
    input_dim = (3, 32, 32)
    if features_nodes is None:
        features_nodes = {"view": "features", "linear": "linear"}
    return {
        "model": model,
        "features_nodes": features_nodes,
        "input_dim": input_dim,
        "test_transforms": test_transforms,
        "train_transforms": train_transforms,
    }


def ResNet34Cifar100(features_nodes=None):
    model = resnet.ResNet34(100)
    train_transforms, test_transforms = _get_default_cifar100_transforms()
    input_dim = (3, 32, 32)
    if features_nodes is None:
        features_nodes = {"view": "features", "linear": "linear"}
    return {
        "model": model,
        "features_nodes": features_nodes,
        "input_dim": input_dim,
        "test_transforms": test_transforms,
        "train_transforms": train_transforms,
    }


def ResNet34SVHN(features_nodes=None):
    model = resnet.ResNet34(10)
    train_transforms, test_transforms = _get_default_svhn_transforms()
    input_dim = (3, 32, 32)
    if features_nodes is None:
        features_nodes = {"view": "features", "linear": "linear"}
    return {
        "model": model,
        "features_nodes": features_nodes,
        "input_dim": input_dim,
        "test_transforms": test_transforms,
        "train_transforms": train_transforms,
    }


models_registry = {
    "densenet121_cifar10": DenseNet121Cifar10,
    "densenet121_cifar100": DenseNet121Cifar100,
    "densenet121_svhn": DenseNet121SVHN,
    "vgg16_cifar10": VGG16Cifar10,
    "vgg16_cifar100": VGG16Cifar100,
    "vgg16_svhn": VGG16SVHN,
    "resnet34_cifar10": ResNet34Cifar10,
    "resnet34_cifar100": ResNet34Cifar100,
    "resnet34_svhn": ResNet34SVHN,
}


def get_model_essentials(model_name, features_nodes=None) -> Dict[str, Any]:
    if model_name not in models_registry:
        raise ValueError("Unknown model name: {}".format(model_name))
    return models_registry[model_name](features_nodes=features_nodes)
