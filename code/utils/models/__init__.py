
# -------------------------------
from typing import Any, Dict
import os
import torch
from torchvision import transforms
import json 
from . import resnet, densenet
from .models import ThresholdClassifier, BayesClassifier, MLPClassifier
from transformers import ViTForImageClassification, ViTImageProcessor
import timm

DATA_DIR = os.environ.get("DATA_DIR", "./data")
CHECKPOINTS_DIR_BASE = os.environ.get("CHECKPOINTS_DIR", "checkpoints/")

__all__ = ["ThresholdClassifier", "BayesClassifier", "MLPClassifier"]


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


def _get_default_imagenet_transforms():
    # Standard ImageNet normalization and resizing to 224×224
    statistics = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(*statistics),
    ])
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(*statistics),
    ])
    return train_transforms, test_transforms

class ViTLogitsOnly(ViTForImageClassification):
    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Call parent forward to get the ImageClassifierOutput
        outputs = super().forward(*args, **kwargs)
        # Return only the logits tensor
        return outputs.logits

def ViTBase16ImageNet(features_nodes=None):
    train_transforms, test_transforms = _get_default_imagenet_transforms()
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTLogitsOnly.from_pretrained(
        "google/vit-base-patch16-224")
    input_dim = (3, 224, 224)
    if features_nodes is None:
        features_nodes = {"view": "pooler", "classifier": "classifier"}
    return {
        "model": model,
        "features_nodes": features_nodes,
        "input_dim": input_dim,
        "test_transforms": processor,
        "train_transforms": train_transforms,
    }

def TimmViTBase16ImageNet(features_nodes=None):
    train_transforms, test_transforms = _get_default_imagenet_transforms()
    
    model = timm.create_model('vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=True)

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    input_dim = (3, 224, 224)
    if features_nodes is None:
        features_nodes = {"view": "pooler", "classifier": "classifier"}
    return {
        "model": model,
        "features_nodes": features_nodes,
        "input_dim": input_dim,
        "test_transforms": transforms,
        "train_transforms": train_transforms,
    }


def TimmViTTiny16ImageNet(features_nodes=None):
    train_transforms, test_transforms = _get_default_imagenet_transforms()
    
    model = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k', pretrained=True)

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    # print("data_config", data_config)
    # print("transforms", transforms)
    # exit()
    input_dim = (3, 224, 224)
    if features_nodes is None:
        features_nodes = {"view": "pooler", "classifier": "classifier"}
    return {
        "model": model,
        "features_nodes": features_nodes,
        "input_dim": input_dim,
        "test_transforms": transforms,
        "train_transforms": train_transforms,
    }


def ViTLarge16ImageNet(features_nodes=None):
    train_transforms, test_transforms = _get_default_imagenet_transforms()
    model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224-in21k")
    input_dim = (3, 224, 224)
    if features_nodes is None:
        features_nodes = {"view": "pooler", "classifier": "classifier"}
    return {
        "model": model,
        "features_nodes": features_nodes,
        "input_dim": input_dim,
        "test_transforms": test_transforms,
        "train_transforms": train_transforms,
    }

def ViTHuge14ImageNet(features_nodes=None):
    train_transforms, test_transforms = _get_default_imagenet_transforms()
    model = ViTForImageClassification.from_pretrained("google/vit-huge-patch14-224-in21k")
    input_dim = (3, 224, 224)
    if features_nodes is None:
        features_nodes = {"view": "pooler", "classifier": "classifier"}
    return {
        "model": model,
        "features_nodes": features_nodes,
        "input_dim": input_dim,
        "test_transforms": test_transforms,
        "train_transforms": train_transforms,
    }


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






models_registry = {
    "resnet34_cifar10": ResNet34Cifar10,
    "resnet34_cifar100": ResNet34Cifar100,
    "densenet121_cifar10": DenseNet121Cifar10,
    "densenet121_cifar100": DenseNet121Cifar100,
    "vit_base16_imagenet": ViTBase16ImageNet,
    "vit_large16_imagenet": ViTLarge16ImageNet,
    "vit_huge14_imagenet": ViTHuge14ImageNet,
    "timm_vit_base16_imagenet": TimmViTBase16ImageNet,
    "timm_vit_tiny16_imagenet": TimmViTTiny16ImageNet,
}


def get_model_essentials(model, dataset, features_nodes=None) -> Dict[str, Any]:

    name = "_".join([model, dataset])
    
    if name not in models_registry:
        raise ValueError("Unknown model name: {}".format(name))
    return models_registry[name](features_nodes=features_nodes)





def get_model(model_name: str, 
              dataset_name: str, 
              n_classes: int,
              input_dim=None,
              model_seed=1, 
              checkpoint_dir=None,
              desired_indices=None) -> torch.nn.Module:
    # print("desired_indices", desired_indices)
  
    if model_name == "mlp_synth_dim-10_classes-7":
       
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        config_model_path = os.path.join(checkpoint_dir, "config.json")

        if not os.path.exists(config_model_path):
            raise FileNotFoundError(f"Configuration file not found at {config_model_path}")
        # Load the configuration file
        with open(config_model_path, "r") as f:
            config_model = json.load(f)
        
        # Instantiate the MLP classifier
        model = MLPClassifier(
            input_dim=config_model["dim"], 
            hidden_size=config_model["hidden_dims"][0], 
            num_hidden_layers=config_model["num_hidden_layers"], 
            dropout_p=config_model["dropout_p"], 
            num_classes=config_model["n_classes"]
            )
        
        # Load the model weights
        
        checkpoint_path = os.path.join(checkpoint_dir, "best_mlp.pth")

    elif (model_name == "resnet34") and (dataset_name == "gaussian_mixture"):

        checkpoint_dir = os.path.join(checkpoint_dir, 
                                      model_name + f"_synth_dim-{input_dim}_classes-{n_classes}")
        # config_model_path = os.path.join(checkpoint_dir, "config.json")

        # if not os.path.exists(config_model_path):
        #     raise FileNotFoundError(f"Configuration file not found at {config_model_path}")
        # # Load the configuration file
        # with open(config_model_path, "r") as f:
        #     config_model = json.load(f)
        
        # Instantiate the MLP classifier
        model = resnet.ResNet34(n_classes)
        
        # Load the model weights
        checkpoint_path = os.path.join(checkpoint_dir, "best_mlp.pth")


    elif (model_name in ["resnet34", "densenet121"]) and (dataset_name in["cifar10", "cifar100"]):
        model_essentials = get_model_essentials(model_name, dataset_name)
        model = model_essentials["model"]
        checkpoint_path = os.path.join(checkpoint_dir, "_".join([model_name, dataset_name]), str(model_seed), "best.pth")
        try:
            checkpoint_path = os.path.join(checkpoint_dir, "_".join([model_name, dataset_name]), str(model_seed), "best.pth")
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu"
            )
        except:
            checkpoint_path = os.path.join(checkpoint_dir, "_".join([model_name, dataset_name]), "last.pt")
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu"
            )
           # w = torch.load(os.path.join(checkpoint_dir, args.model_name, "last.pt"), map_location="cpu")
       
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)
        return model
    # Handle ViT variants
    elif model_name.startswith(("vit_", "timm_vit_")) and dataset_name == "imagenet":
        essentials = get_model_essentials(model_name, dataset_name)
        model = essentials["model"]
        return model
    


        
    if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
  
    # print("checkpoint keys", checkpoint.keys())
    # print("model keys before loading", model.state_dict().keys())
    if "openmix" in checkpoint_dir:
        # add one class to model output
        model._modules[list(model._modules.keys())[-1]] = torch.nn.Linear(
            model._modules[list(model._modules.keys())[-1]].in_features,
            model._modules[list(model._modules.keys())[-1]].out_features + 1,
        )
    model.load_state_dict(checkpoint)
    # missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    # print("Missing:", missing_keys)
    # print("Unexpected:", unexpected_keys)

    if desired_indices is not None:
        print("Subsetting model to classes:", desired_indices)
        model = SubsetLogitWrapper(model, desired_indices)
    
    return model


class SubsetLogitWrapper(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, desired_indices: list[int]):
        """
        Wraps any classifier that outputs logits over N classes,
        and on forward() returns only the logits of `desired_indices`.
        """
        super().__init__()
        self.model = base_model
        print("desired_indices", desired_indices)
    
        self.indices = torch.tensor(desired_indices, dtype=torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get full logits from base model: shape [B, N]
        logits = self.model(x)
        # Select only desired class logits: shape [B, K]
        logits = logits[:, self.indices]
        print("logits shape after subset", logits.shape)
        return logits[:, self.indices]