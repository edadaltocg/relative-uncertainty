import os
import torch
import torch.utils.data
from tqdm import tqdm


from src.utils.datasets import get_dataset
from src.utils.models import get_model_essentials

CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", "checkpoints/")
CHECKPOINTS_DIR = os.path.join(CHECKPOINTS_DIR, "regmixup/")


def main(model_name, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_essential = get_model_essentials(model_name)
    checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR, model_name, str(seed), "best.pth"), map_location="cpu")
    # replace module. from keys
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model = model_essential["model"]
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # transform
    transform = model_essential["test_transforms"]
    dataset_name = model_name.split("_")[-1]
    dataset = get_dataset(dataset_name, os.environ.get("DATA_DIR", ""), train=False, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, pin_memory=True, num_workers=6)
    num_classes = 10 if dataset_name == "cifar10" else 100
    logits = torch.empty((len(dataset), num_classes), dtype=torch.float32)
    targets = torch.empty((len(dataset),), dtype=torch.int64)
    idx = 0
    for batch in tqdm(dataloader):
        x, y = batch
        x = x.to(device)
        with torch.no_grad():
            out = model(x).cpu()
        logits[idx : idx + x.shape[0]] = out
        targets[idx : idx + x.shape[0]] = y
        idx += x.shape[0]

    os.makedirs(root, exist_ok=True)
    torch.save(logits, os.path.join(root, "test_logits.pt"))
    torch.save(targets, os.path.join(root, "test_targets.pt"))
    acc = torch.mean((logits.argmax(dim=1) == targets).float()).item()
    print("Accuracy:", acc)


if __name__ == "__main__":
    model_names = [
        "resnet34_cifar10",
        "resnet34_cifar100",
        "densenet121_cifar10",
        "densenet121_cifar100",
    ]
    for model_name in model_names:
        print(model_name)
        for seed in range(1, 11, 1):
            root = os.path.join(CHECKPOINTS_DIR, model_name, str(seed))
            if not os.path.isfile(os.path.join(root, "test_logits.pt")):
                try:
                    main(model_name, seed)
                except Exception as e:
                    print(e)
