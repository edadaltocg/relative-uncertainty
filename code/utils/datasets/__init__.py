import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from torch.distributions import MultivariateNormal
import time
from tqdm import tqdm
DATA_DIR = os.environ.get("DATA_DIR", "./data")
CHECKPOINTS_DIR_BASE = os.environ.get("CHECKPOINTS_DIR", "checkpoints/")
from torchvision import transforms


# -------------------------------
# 1. Data Generation: Gaussian Mixture Dataset
# -------------------------------



# class GaussianMixtureDataset(Dataset):
#     """
#     Dataset for a mixture of Gaussians with chunked pre-generation.

#     Args:
#         n_samples (int): Total number of samples to generate.
#         means (Tensor): Tensor of shape [n_components, dim].
#         covs (Tensor): Tensor of shape [n_components, dim, dim].
#         weights (Tensor): Tensor of shape [n_components].
#         seed (int, optional): Random seed for reproducibility. Default: None.
#         block_size (int, optional): Number of samples to generate per block to
#             control peak memory usage. Default: 5000.
#     """
#     def __init__(self, n_samples, means, covs, weights, seed=None, block_size=5000):
#         super().__init__()
#         self.n_samples = n_samples
#         self.block_size = block_size

#         # Keep parameters on CPU
#         self.means = means.cpu()
#         self.covs = covs.cpu()
#         self.weights = weights.cpu()
#         self.cov_chols   = torch.linalg.cholesky(covs.cpu())  # [n_classes, dim, dim]
#         # Reproducible RNG on CPU
#         # self.rng = torch.Generator(device='cpu')
#         # if seed is not None:
#         #     self.rng.manual_seed(seed)

#         # Sample component indices for all samples
#         components = torch.multinomial(
#             self.weights,
#             self.n_samples,
#             replacement=True
#             # generator=self.rng
#         )

#         # Pre-generate samples and labels in blocks
#         samples_list = []
#         labels_list = []
#         for start in tqdm(range(0, self.n_samples, self.block_size), desc="Generating samples"):
#         # for start in range(0, self.n_samples, self.block_size):
#             end         = min(start + self.block_size, self.n_samples)
#             comps_block = components[start:end]  # [block_size]
#             B           = comps_block.size(0)

#             # Index means and covs for this block
#             means_block = self.means[comps_block]  # [block_size, dim]
#             # covs_block = self.covs[comps_block]    # [block_size, dim, dim]
#             L_block     = self.cov_chols[comps_block]

#             # Sample from the block of MVNs

#             # sample standard normals and apply transform
#             z           = torch.randn(B, self.means.size(1))
#             samples_block  = means_block + torch.bmm(L_block, z.unsqueeze(-1)).squeeze(-1)
#             # mvn = MultivariateNormal(
#             #     loc=means_block,
#             #     covariance_matrix=covs_block,
#             #     validate_args=False
#             # )
#             # samples_block = mvn.sample(sample_shape=())  # [block_size, dim] #, generator=self.rng

#             samples_list.append(samples_block)
#             labels_list.append(comps_block)

#         # Concatenate all blocks into final buffers
#         self.samples = torch.cat(samples_list, dim=0)  # [n_samples, dim]
#         self.labels = torch.cat(labels_list, dim=0)    # [n_samples]

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, idx):
#         # Return a single sample and its label
#         return self.samples[idx], self.labels[idx]

import os
import torch
from torch.utils.data import Dataset



class GaussianMixtureDataset(Dataset):
    """
    Dataset for a mixture of Gaussians with fully pre-generated samples for
    speed and reproducibility.

    Either:
      - If `samples_path` and `labels_path` exist under `data_dir`, loads them from disk.
      - Otherwise, generates `n_samples` on the fly (deterministic, single-shot).

    Args:
        data_dir (str): Directory where pre-generated `samples.pt` and `labels.pt` live.
                        If files are missing, generation will occur in-memory.
        n_samples (int): Number of samples to generate if pre-generated files are absent.
        means (Tensor): [n_classes, dim] tensor of mixture means.
        covs (Tensor):  [n_classes, dim, dim] tensor of mixture covariances.
        weights (Tensor): [n_classes] tensor of class weights (sum to 1).
        seed (int, optional): RNG seed for reproducibility. Default: None.
    """
    def __init__(self, data_dir = None, 
                 n_samples = None,
                #   means=None, covs=None, weights=None, 
                #   seed=None,
                #  device="cpu"
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.n_samples = n_samples
        self.device = torch.device('cpu')

        samples_path = os.path.join(data_dir, 'samples.pt')
        labels_path  = os.path.join(data_dir, 'labels.pt')
        if os.path.exists(samples_path) and os.path.exists(labels_path):
            # Load pre-generated data
            self.samples = torch.load(samples_path)
            self.labels  = torch.load(labels_path)
        else:
            raise FileNotFoundError(
                f"Pre-generated samples not found at {samples_path} or {labels_path}. "
                "Please generate the dataset first."
            )
        # else:
        #     # Pre-generate all samples in one shot
        #     dim = means.size(1)
        #     n_classes = means.size(0)

        #     means_cpu   = means.to(device)
        #     covs_cpu    = covs.to(device)
        #     weights_cpu = weights.to(device)
        #     L_chols     = torch.linalg.cholesky(covs_cpu)

        #     # Deterministic CPU RNG for component draws
        #     gen = torch.Generator(device=device)
        #     if seed is not None:
        #         gen.manual_seed(seed)
        #     comps = torch.multinomial(weights_cpu, self.n_samples, replacement=True, generator=gen)

        #     # Allocate buffers
        #     samples = torch.empty(self.n_samples, dim, dtype=torch.float32)
        #     labels  = comps.clone()

        #     # Batch-by-class sampling
        #     for i in range(n_classes):
        #         idx = (comps == i).nonzero(as_tuple=True)[0]
        #         if idx.numel() == 0:
        #             continue
        #         # GPU-backed noise generation for speed
        #         z = torch.randn(idx.numel(), dim, device=device)
        #         block = (L_chols[i] @ z.T).T + means[i]
        #         samples[idx] = block.cpu()

        #     self.samples = samples
        #     self.labels  = labels
        #     # Optionally: save for future runs
        #     os.makedirs(data_dir, exist_ok=True)
        #     torch.save(self.samples, samples_path)
        #     torch.save(self.labels, labels_path)

        # Final shapes
        print(f"self.n_samples", self.n_samples)
        print("self.samples.size(0)", self.samples.size(0))
        assert self.samples.size(0) == self.n_samples 

        assert self.labels.size(0)  == self.n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # reshape if needed (e.g., for images):
        return self.samples[idx].view(3,32,32), self.labels[idx]


# class GaussianMixtureDataset(Dataset):
#     """
#     Dataset for a mixture of Gaussians with chunked pre-generation.

#     Args:
#         n_samples (int): Total number of samples to generate.
#         means (Tensor): Tensor of shape [n_components, dim].
#         covs (Tensor): Tensor of shape [n_components, dim, dim].
#         weights (Tensor): Tensor of shape [n_components].
#         seed (int, optional): Random seed for reproducibility. Default: None.
#         block_size (int, optional): Number of samples to generate per block to
#             control peak memory usage. Default: 5000.
#     """
#     def __init__(self, n_samples, means, covs, weights, seed=None):
#         super().__init__()
#         self.n_samples = n_samples

#         # Keep parameters on CPU
#         self.dim = means.size(1)  # Dimension of each sample
#         self.means = means.cpu()
#         self.covs = covs.cpu()
#         self.weights = weights.cpu()
#         self.cov_chols   = torch.linalg.cholesky(covs.cpu())  # [n_classes, dim, dim]
#         # Reproducible RNG on CPU
#         self.gen = torch.Generator(device='cpu')
#         if seed is not None:
#             self.gen.manual_seed(seed)

#         # Sample component indices for all samples
#         self.components = torch.multinomial(
#             self.weights,
#             self.n_samples,
#             replacement=True
#             # generator=self.rng
#         )

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, idx):
#         comp = self.components[idx]
#         mean = self.means[comp]  # [block_size, dim]
#         L     = self.cov_chols[comp]  # [block_size, dim, dim]

#         # sample standard normals and apply transform
#         z       = torch.randn(self.dim)
#         sample  = mean + L @ z
#         # Return a single sample and its label
#         sample = sample.view(3, 32, 32)
#         return sample, comp



# class GaussianMixtureDataset(Dataset):
#     def __init__(self, n_samples, means, stds, weights, seed=None, device="cpu"):
#         """
#         Generates samples from a Gaussian mixture model.
        
#         Args:
#             n_samples (int): Number of samples to generate.
#             means (torch.Tensor): Tensor of shape [n_classes, dim]
#                                   e.g., [7, 10]
#             stds (torch.Tensor): Tensor of shape [n_classes, dim]
#                                  e.g., [7, 10]
#             weights (torch.Tensor): Tensor of shape [n_classes]
#                                     e.g., [7]
#         """
#         # if seed is not None:
#         #     torch.manual_seed(seed)
#         #     torch.cuda.manual_seed_all(seed)  # For GPU support if needed
#         #     torch.backends.cudnn.deterministic = True
#         # gen = torch.Generator(device=device)
#         # gen.manual_seed(seed)
        
#         self.n_samples = n_samples
#         self.means = means      # [n_classes, dim]
#         self.stds = stds        # [n_classes, dim]
#         self.weights = weights  # [n_classes]
#         self.n_classes, self.dim = means.shape

#         # Sample component indices using multinomial sampling
#         # components: [n_samples]
#         self.components = torch.multinomial(self.weights, n_samples, replacement=True
#                                             # , generator=gen
#                                             )
#         # Select means and stds for each sampled component
#         # chosen_means: [n_samples, dim]
#         chosen_means = self.means[self.components]
#         # chosen_stds: [n_samples, dim]
#         chosen_stds = self.stds[self.components]
#         # Sample from the normal distribution (elementwise).
#         # samples: [n_samples, dim]
#         self.samples = torch.normal(mean=chosen_means, std=chosen_stds
#                                     # , generator=gen
#                                     ).cpu()
#         # Labels are the indices of the components (classes)
#         # labels: [n_samples]
#         self.labels = self.components.cpu()

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, idx):
#         # Returns:
#         #  sample: [dim] and label: scalar
#         return self.samples[idx], self.labels[idx]


def load_data_config(data_folder):
    """
    Loads the dataset configuration from the config.json file in data_folder.
    """
    config_path = os.path.join(data_folder, "config.json")
    with open(config_path, "r") as f:
        data_config = json.load(f)
    return data_config

def load_generating_params(data_folder):
    """
    Loads the generating means and standard deviation from dataset_stats.csv.
    
    Assumes that dataset_stats.csv contains one row per class for a given dataset.
    We look for rows corresponding to the test dataset (if not found, fallback to "train").
    Each row's "mean" and "std" are stored as semicolon–separated numbers.
    
    Returns:
        means (ndarray): Array of shape (n_classes, dim) containing the generating means.
        std (float): The standard deviation (assumed to be constant across classes).
    """
    stats_path = os.path.join(data_folder, "dataset_stats.csv")
    df = pd.read_csv(stats_path)
    
    # Use rows for "test_dataset" if available; otherwise fall back to "train"
    df_subset = df[df["dataset"] == "test_dataset"]
    if df_subset.empty:
        df_subset = df[df["dataset"] == "train"]
    
    df_subset = df_subset.sort_values(by="class")
    means_list = []
    std_list = []
    for _, row in df_subset.iterrows():
        # Strip the square brackets and split on any whitespace.
        mean_str = row["mean"].strip("[]")
        # Split by whitespace and convert each piece to float.
        mean_vals = np.array([float(v) for v in mean_str.split()])
        std_val = float(row["std"])
        means_list.append(mean_vals)
        std_list.append(std_val)
    
    means = np.stack(means_list, axis=0)
    # Assume the standard deviation is identical across classes.
    std = std_list[0]
    return means, std

def bayes_proba(X_input, means_input, std_input):
    """
    For each input x (rows of X_input), returns the probability distribution over classes,
    computed via:
    
        P(y=i | x) = exp(-||x - mu_i||^2/(2 std_input^2)) / sum_j exp(-||x - mu_j||^2/(2 std_input^2))
    
    Parameters:
        X_input (ndarray): Shape (n_samples, dim)
        means_input (ndarray): Shape (n_classes, dim)
        std_input (float): Standard deviation.
        
    Returns:
        probs (ndarray): Shape (n_samples, n_classes)
    """
    # Compute squared Euclidean distances.
    dists = np.sum((X_input[:, None, :] - means_input[None, :, :]) ** 2, axis=2)
    logits = -dists / (2 * std_input**2)
    # For numerical stability, subtract the maximum logit per sample.
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs



from typing import Any, Dict, Type
import numpy as np	
import os
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageNet
from PIL import Image
from code.utils.models import get_model_essentials

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


class TorchvisionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.data_dir = data_dir
        self.data = None
        self.labels = None

    def load_data(self):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

from typing import Iterable, List, Dict, Tuple, Optional, Union
LabelSpec = Union[Iterable[str], Iterable[int]]

class FilteredCIFAR100(Dataset):
    """
    CIFAR-100 wrapper that excludes a set of classes and remaps labels to 0..K-1.

    Parameters
    ----------
    root : str
        Root directory for CIFAR-100 (same as torchvision).
    train : bool
        Train split (True) or test split (False).
    exclude : Iterable[str] or Iterable[int]
        Classes to remove. You can pass fine-label names or their integer ids.
    transform : callable, optional
        Transform applied to PIL image (same as torchvision).
    target_transform : callable, optional
        Additional transform applied to the *remapped* label.
    download : bool
        Download CIFAR-100 if needed.

    Attributes
    ----------
    base : torchvision.datasets.CIFAR100
        The underlying CIFAR-100 dataset (unfiltered).
    indices : List[int]
        Indices kept from the underlying dataset.
    classes : List[str]
        Fine-class names of the *kept* classes, in the remapped order.
    class_to_idx : Dict[str, int]
        Mapping from kept class name -> new label id in [0..K-1].
    orig_to_new : Dict[int, int]
        Mapping from original CIFAR-100 label -> new label id (only for kept labels).
    new_to_orig : List[int]
        Inverse of orig_to_new for the kept classes: new id -> original id.
    targets : List[int]
        Remapped labels for the filtered dataset (aligned with __getitem__ order).
    """
    def __init__(
        self,
        root: str,
        train: bool = False,
        exclude = [
             'bus',
             'camel',
             'cattle',
             'fox',
             'leopard',
             'lion',
             'pickup_truck',
             'streetcar',
             'tank',
             'tiger',
             'tractor',
             'train',
             'wolf'
             ],
        download: bool = False,
    ):
        super().__init__()
        train_transform, test_transform = _get_default_cifar100_transforms()
        transform = train_transform if train == "train" else test_transform
        self.base = CIFAR100(root=root, train=train, transform=transform, download=download)

        # --- Normalize 'exclude' to a set of original integer ids ---
        all_names: List[str] = list(self.base.classes)           # length 100
        name_to_idx: Dict[str, int] = dict(self.base.class_to_idx)
        if len(exclude) == 0:
            exclude_idx: set = set()
        else:
            first = next(iter(exclude))
            if isinstance(first, str):
                bad = [n for n in exclude if n not in name_to_idx]
                if bad:
                    raise ValueError(f"Unknown class names in exclude: {bad}")
                exclude_idx = {name_to_idx[n] for n in exclude}   # by names
            else:
                exclude_idx = set(int(i) for i in exclude)        # by ids
                bad = [i for i in exclude_idx if not (0 <= i < 100)]
                if bad:
                    raise ValueError(f"Class ids out of range 0..99: {bad}")

        # --- Decide which original labels to keep, and their new ids ---
        kept_orig_ids: List[int] = sorted(set(range(100)) - exclude_idx)
        if len(kept_orig_ids) == 0:
            raise ValueError("All classes were excluded; nothing left to keep.")

        orig_to_new: Dict[int, int] = {orig: new for new, orig in enumerate(kept_orig_ids)}
        new_to_orig: List[int] = kept_orig_ids[:]  # inverse map

        kept_names: List[str] = [all_names[orig] for orig in kept_orig_ids]
        class_to_idx_filtered: Dict[str, int] = {name: i for i, name in enumerate(kept_names)}

        # --- Select indices of samples whose original label is kept ---
        # CIFAR100 stores original integer labels in self.base.targets (list[int])
        targets = self.base.targets
        self.indices: List[int] = [i for i, y in enumerate(targets) if y in orig_to_new]

        # --- Precompute remapped targets in dataloader order (after filtering) ---
        self.targets: List[int] = [orig_to_new[targets[i]] for i in self.indices]

        # Store transforms and mappings
        self.transform = self.base.transform
        self.target_transform = target_transform
        self.classes: List[str] = kept_names
        self.class_to_idx: Dict[str, int] = class_to_idx_filtered
        self.orig_to_new: Dict[int, int] = orig_to_new
        self.new_to_orig: List[int] = new_to_orig

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        base_idx = self.indices[idx]
        img, orig_y = self.base.data[base_idx], self.base.targets[base_idx]  # raw numpy + int

        # torchvision's CIFAR100.__getitem__ converts to PIL and applies transform; reuse that:
        # We replicate its behavior by calling the base's transform on a PIL image.
        # Convert numpy HWC uint8 -> PIL
        from PIL import Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        y = self.orig_to_new[orig_y]
        if self.target_transform is not None:
            y = self.target_transform(y)

        return img, y


# class CIFAR100Torchvision(TorchvisionDataset):
#     def __init__(self, data_dir):
#         super(CIFAR100Torchvision, self).__init__(data_dir, transform)
#         self.trainset = torchvision.datasets.CIFAR100(
#             root=self.data_dir, train=True, download=True, transform=transform)
#         self.testset = torchvision.datasets.CIFAR100(
#             root=self.data_dir, train=False, download=True, transform=transform)

#     def get_trainset(self):
#         return self.trainset

#     def get_testset(self):
#         return self.testset

#     def get_num_classes(self):
#         return 100


class TImmImageNet(Dataset):
    def __init__(self, root, train=False, transform=None, **kwargs):
        split = "train" if train else "val"
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "ILSVRC/imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC/ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
    def __len__(self):
            return len(self.samples)
    def __getitem__(self, idx):
            
        img = Image.open(self.samples[idx]).convert("RGB")
        label = self.targets[idx]
        
        # 4) return exactly two things:
        return self.transform(img), label       # (Tensor[C,H,W], int)

class ImageNetKaggle(Dataset):
    def __init__(self, root, train=False, transform=None, **kwargs):
        split = "train" if train else "val"
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "ILSVRC/imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC/ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
    def __len__(self):
            return len(self.samples)
    def __getitem__(self, idx):
            
        img = Image.open(self.samples[idx]).convert("RGB")
        label = self.targets[idx]
        
        # 2) preprocess → returns {"pixel_values": Tensor[1,3,224,224]}
        enc = self.transform(img, return_tensors="pt")
        # 3) squeeze out the dummy batch‐dim → (3,224,224)
        pix = enc["pixel_values"].squeeze(0)
        
        # 4) return exactly two things:
        return pix, label       # (Tensor[C,H,W], int)

        # x = Image.open(self.samples[idx]).convert("RGB")
        # if self.transform:
        #     x = self.transform(x)
        # return x, self.targets[idx]


datasets_registry: Dict[str, Any] = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "filtered_cifar100": FilteredCIFAR100,
    "svhn": SVHN,
    "imagenet": TImmImageNet,
}


# def get_synthetic_dataset(model_name: str, # to get the test transform
#                  checkpoint_dir: str = os.path.join(CHECKPOINTS_DIR_BASE, "ce"),
#                  n_samples_train = 1000,
#                 n_samples_test = 1000,
#                  seed_train: int = 0,
#                  seed_test: int = 1,
#                  device: str = "cpu",
#                  **kwargs) -> Dataset:
    
#     checkpoints_dir = os.path.join(checkpoints_dir, model_name)
#     config_model_path = os.path.join(checkpoints_dir, "config.json")

#     if not os.path.exists(config_model_path):
#         raise FileNotFoundError(f"Configuration file not found at {config_model_path}")
#     # Load the configuration file
#     with open(config_model_path, "r") as f:
#         config_model = json.load(f)

#     means = torch.tensor(config_model["means"]).to(device)
#     stds = torch.tensor(config_model["stds"]).to(device)
#     weights = torch.tensor(config_model["weights"]).to(device)

#     # Generate Dataset
#         # Create training and validation datasets.
#     train_dataset = GaussianMixtureDataset(n_samples_train, means, stds, weights, seed=seed_train)
#     val_dataset = GaussianMixtureDataset(n_samples_test, means, stds, weights, seed=seed_test)

#     return train_dataset, val_dataset

def get_synthetic_dataset(
        # model_name: str, # to get the test transform
                          data_name: str = "gaussian_mixture",
                #  checkpoint_dir: str = os.path.join(CHECKPOINTS_DIR_BASE, "ce"),
                 n_samples = 1000,
                 dim: int = 10,
                 n_classes: int = 7,
                 seed: int = 0,
                #  device: str = "cpu"
                 ) -> Dataset:
    


    # checkpoint_dir = os.path.join(checkpoint_dir, 
    #                                 model_name + f"_synth_dim-{input_dim}_classes-{n_classes}")
    
    # data_parameters = np.load(os.path.join(checkpoint_dir, "data_parameters.npz"))
    # means =  torch.from_numpy(data_parameters["means"].astype(np.float32))
    # covs =  torch.from_numpy(data_parameters["covs"].astype(np.float32))
    # weights =  torch.from_numpy(data_parameters["weights"].astype(np.float32))
    data_dir = os.path.join(DATA_DIR, data_name, 
                        f"dim-{dim}_classes-{n_classes}-seed-{seed}")
    return GaussianMixtureDataset(data_dir=data_dir, n_samples=n_samples)


def _get_openmix_cifar10_transforms():
    statistics = ((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))
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


def _get_openmix_cifar100_transforms():
    statistics = ((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
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



def get_dataset(dataset_name: str, 
                 model_name: str, # to get the test transform
                 root: str,
                 shuffle: bool = False,
                 random_state: int = 0,
                 train=False,
                 transform="test",
                 preprocess: bool = True,
                 **kwargs) -> Dataset:
    
    if dataset_name not in datasets_registry.keys():
        raise ValueError(f"Dataset {dataset_name} not found")

    if dataset_name == "filtered_cifar100":
         return datasets_registry[dataset_name](
            root, train=train, 
            download=True) 
    transform = get_model_essentials(model_name, dataset_name)[f"{transform}_transforms"]
    if ("openmix" == preprocess) and ("cifar10" == dataset_name):
            _, transform = _get_openmix_cifar10_transforms()
    if ("openmix" == preprocess) and ("cifar100" == dataset_name):
            _, transform = _get_openmix_cifar100_transforms()
    # print("transform", transform)
    if not shuffle:
        return datasets_registry[dataset_name](
            root, train=train, 
            transform=transform, 
            download=True) 
    
    else:
        dataset = datasets_registry[dataset_name](
                root, train=train, 
                transform=transform, 
                download=True) 
        # reproducible permutation
        gen = torch.Generator()
        gen.manual_seed(random_state)
        perm = torch.randperm(len(dataset), generator=gen).tolist()

        return Subset(dataset, perm)

        