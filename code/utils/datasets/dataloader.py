import torch
from typing import Tuple
from torch.utils.data import  DataLoader, Subset
import random
from code.utils.models import get_model_essentials







def prepare_ablation_dataloaders(
    dataset: torch.utils.data.Dataset, 
    seed_split=None, 
    n_cal=0.5, 
    n_res=0., 
    n_test=0.5,
    batch_size_train=252, 
    batch_size_test=252, 
    cal_transform=None, 
    res_transform=None,
    data_name="cifar10",
    model_name="resnet34",
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits the dataset and prepares train, validation, and test DataLoaders.

    Args:
        dataset (torch.utils.data.Dataset): The full dataset.
        config (Dict[str, Any]): The data configuration.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test dataloaders.


    """
    def _resolve_count(x: float | int, total: int, name: str) -> int:
        if isinstance(x, float):
            if not (0 < x <= 1):
                raise ValueError(f"{name} ratio must be in (0,1], got {x}")
            return int(round(total * x))
        elif isinstance(x, int):
            if x < 0:
                raise ValueError(f"{name} count must be >= 0, got {x}")
            return x
        else:
            raise TypeError(f"{name} must be float (ratio) or int (count), got {type(x).__name__}")

    def _change_transform(dataset, transform):
            transform = get_model_essentials(
                model_name,
                data_name
            )[f"{transform}_transforms"]
            dataset.dataset.transform = transform
            
    n = len(dataset)

    n_cal_samples = _resolve_count(n_cal, n, "n_cal")
    n_test_samples = _resolve_count(n_test, n, "n_test")
    n_res_samples = _resolve_count(n_res, n, "n_res")

    perm = list(range(n))
    if seed_split is not None:
        # Use a generator for local reproducibility of the shuffle
        random.shuffle(perm)


    if n_res_samples == 0:
        cal_idx = perm[:n_cal_samples]
        test_idx = perm[n - n_test_samples:]

        cal_dataset = Subset(dataset, cal_idx)
        test_dataset = Subset(dataset, test_idx)
        if cal_transform is not None:
            _change_transform(cal_dataset, cal_transform)


        cal_loader = DataLoader(
            cal_dataset, batch_size=batch_size_train, shuffle=False, pin_memory=True, num_workers=10
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size_test, shuffle=False, pin_memory=True, num_workers=10
        )
        print("Length of Cal dataset:", len(cal_loader.dataset))
        print("Length of test dataset:", len(test_loader.dataset))

        return None, cal_loader, test_loader

    else:

        
        cal_idx = perm[:n_cal_samples]
        res_idx = perm[n_cal_samples:n_cal_samples + n_res_samples]
        test_idx = perm[n - n_test_samples:]

        cal_dataset = Subset(dataset, cal_idx)
        res_dataset = Subset(dataset, res_idx)
        test_dataset = Subset(dataset, test_idx)

        if cal_transform is not None:
            _change_transform(cal_dataset, cal_transform)
        if res_transform is not None:
            _change_transform(res_dataset, res_transform)

        cal_loader = DataLoader(
            cal_dataset, batch_size=batch_size_train, shuffle=False, pin_memory=True, num_workers=10
        )
        res_loader = DataLoader(
            res_dataset, batch_size=batch_size_train, shuffle=False, pin_memory=True, num_workers=10
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size_test, shuffle=False, pin_memory=True, num_workers=10
        )
        print("Length of Cal dataset:", len(cal_loader.dataset))
        print("Length of Res dataset:", len(res_loader.dataset))
        print("Length of test dataset:", len(test_loader.dataset))
        return res_loader, cal_loader, test_loader

       