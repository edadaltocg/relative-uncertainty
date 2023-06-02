import yaml
import torch
import pickle
import logging
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from tools.cifar_c_class import CORRUPTIONS, CIFAR10_C, CIFAR100_C

# define an abstract class to load datasets from torchvision


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

# define class for dataset CIFAR10 inherited from TorchvisionDataset


class CIFAR10Torchvision(TorchvisionDataset):
    def __init__(self, data_dir, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])):
        super(CIFAR10Torchvision, self).__init__(data_dir, transform)
        self.trainset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=transform)
        self.testset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=transform)

    def get_trainset(self):
        return self.trainset

    def get_testset(self):
        return self.testset

    def get_num_classes(self):
        return 10

# define class for dataset CIFAR100 inherited from TorchvisionDataset


class CIFAR100Torchvision(TorchvisionDataset):
    def __init__(self, data_dir, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.482158, 0.446531),
                             (0.247032, 0.243486, 0.261588))
    ])):
        super(CIFAR100Torchvision, self).__init__(data_dir, transform)
        self.trainset = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=True, download=True, transform=transform)
        self.testset = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=False, download=True, transform=transform)

    def get_trainset(self):
        return self.trainset

    def get_testset(self):
        return self.testset

    def get_num_classes(self):
        return 100


class CIFAR10C(TorchvisionDataset):
    def __init__(self, data_dir,
                 corruption,
                 intensity,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                 ])):
        super(CIFAR10C, self).__init__(data_dir, transform)
        self.trainset = None
        self.testset = CIFAR10_C(root=self.data_dir, split=corruption,
                                 intensity=intensity, download=True, transform=transform)

    def get_trainset(self):
        return self.trainset

    def get_testset(self):
        return self.testset

    def get_num_classes(self):
        return 100


class CIFAR100C(TorchvisionDataset):
    def __init__(self, data_dir,
                 corruption,
                 intensity,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(
                         (0.4914, 0.482158, 0.446531), (0.247032, 0.243486, 0.261588))
                 ])):
        super(CIFAR100C, self).__init__(data_dir, transform)
        self.trainset = None
        self.testset = CIFAR100_C(root=self.data_dir, split=corruption,
                                  intensity=intensity, download=True, transform=transform)

    def get_trainset(self):
        return self.trainset

    def get_testset(self):
        return self.testset

    def get_num_classes(self):
        return 100


# define a method to return dataset from string


def get_dataset(dataset_name, data_dir, **kwargs):
    if dataset_name == 'cifar10':
        return CIFAR10Torchvision(data_dir=data_dir)
    elif dataset_name == 'cifar100':
        return CIFAR100Torchvision(data_dir=data_dir)
    elif dataset_name == 'cifar10c':
        return CIFAR10C(data_dir=data_dir,
                        corruption=kwargs['corruption'],
                        intensity=kwargs['intensity'])
    elif dataset_name == 'cifar100c':
        return CIFAR100C(data_dir=data_dir,
                         corruption=kwargs['corruption'],
                         intensity=kwargs['intensity'])
    else:
        raise NotImplementedError


def get_data(dataset_name, data_path, **kwargs):
    # load dataset
    dataset = get_dataset(dataset_name, data_path, **kwargs)
    trainset = dataset.get_trainset()
    testset = dataset.get_testset()
    num_classes = dataset.get_num_classes()
    return trainset, testset, num_classes


def read_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_logger(logger_name, log_file=None, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    # add file handler if log_file is not None, otherwise only print to console
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def fpr_at_fixed_tpr(fprs, tprs, thresholds, tpr_level: float = 0.95):
    if all(tprs < tpr_level):
        raise ValueError(f"No threshold allows for TPR at least {tpr_level}.")
    idxs = [i for i, x in enumerate(tprs) if x >= tpr_level]
    idx = min(idxs)
    return fprs[idx], tprs[idx], thresholds[idx]


def get_CIFAR10_class_names(data_dir):
    with open(f'{data_dir}/cifar-10-batches-py/batches.meta', 'rb') as f:
        meta_file = pickle.load(f)
    fine_label_names = meta_file['label_names']
    return fine_label_names


def get_CIFAR100_class_names(data_dir):
    with open(f'{data_dir}/cifar-100-python/meta', 'rb') as f:
        meta_file = pickle.load(f)
    fine_label_names = meta_file['fine_label_names']
    return fine_label_names
