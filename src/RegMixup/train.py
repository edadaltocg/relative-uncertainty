import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data
from tqdm import tqdm

from src.utils.datasets import get_dataset
from src.utils.models import get_model_essentials

CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", "checkpoints/")
CHECKPOINTS_DIR = os.path.join(CHECKPOINTS_DIR, "regmixup/")


def regmixup_data(x, y, alpha=10.0, beta=10.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, beta)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def main(args):
    use_cuda = torch.cuda.is_available()

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    torch.manual_seed(args.seed)

    # Data
    print("==> Preparing data..")
    if args.augment:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    dataset_name = args.model_name.split("_")[-1]
    alpha = 20 if dataset_name == "cifar10" else 10
    beta = alpha
    trainset = get_dataset(dataset_name, os.environ.get("DATA_DIR", ""), train=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    testset = get_dataset(dataset_name, os.environ.get("DATA_DIR", ""), train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

    # Model
    net = get_model_essentials(args.model_name)["model"]

    logname = os.path.join(CHECKPOINTS_DIR, args.model_name, str(args.seed), "results.csv")
    if not os.path.exists(os.path.dirname(logname)):
        os.makedirs(os.path.dirname(logname))

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)
        print(torch.cuda.device_count())
        cudnn.benchmark = True
        print("Using CUDA..")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

    def train():
        net.train()
        train_loss = 0
        reg_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            mixup_x, targets_a, targets_b, lam = regmixup_data(inputs, targets, alpha, beta, use_cuda)
            # inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

            targets_a = torch.cat([targets, targets_a])
            targets_b = torch.cat([targets, targets_b])
            inputs = torch.cat([inputs, mixup_x], dim=0)

            outputs = net(inputs)

            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (
                lam * predicted.eq(targets_a.data).cpu().sum().float()
                + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return (train_loss / batch_idx, reg_loss / batch_idx, 100.0 * correct / total)

    def test():
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            with torch.no_grad():
                outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        acc = 100.0 * correct / total
        return (test_loss / batch_idx, acc)

    def save_checkpoint(name):
        # Save checkpoint.
        path = os.path.join(CHECKPOINTS_DIR, args.model_name, str(args.seed), name)
        if not os.path.isdir(os.path.basename(path)):
            os.makedirs(os.path.basename(path))
        torch.save(net.state_dict(), path)

    def adjust_learning_rate(optimizer, epoch):
        """decrease the learning rate at 100 and 150 epoch"""
        lr = args.lr
        if epoch >= 100:
            lr /= 10
        if epoch >= 250:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    if not os.path.exists(logname):
        with open(logname, "w") as logfile:
            logwriter = csv.writer(logfile, delimiter=",")
            logwriter.writerow(["epoch", "train loss", "reg loss", "train acc", "test loss", "test acc"])

    best_acc = 0  # best test accuracy
    for epoch in tqdm(range(start_epoch, args.epoch)):
        train_loss, reg_loss, train_acc = train()
        test_loss, test_acc = test()
        adjust_learning_rate(optimizer, epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint("best.pth")

        with open(logname, "a") as logfile:
            logwriter = csv.writer(logfile, delimiter=",")
            logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss, test_acc])

    save_checkpoint("last.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MixUp Training")
    parser.add_argument("--model_name", default="resnet34_cifar10", type=str, help="model type")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument("--batch-size", default=128, type=int, help="batch size")
    parser.add_argument("--epoch", default=350, type=int, help="total epochs to run")
    parser.add_argument(
        "--no-augment", dest="augment", action="store_false", help="use standard augmentation (default: True)"
    )
    parser.add_argument("--decay", default=5e-4, type=float, help="weight decay")
    # parser.add_argument("--alpha", default=10.0, type=float, help="mixup interpolation coefficient (default: 1)")
    # parser.add_argument("--beta", default=10.0, type=float, help="mixup interpolation coefficient (default: 1)")
    args = parser.parse_args()

    main(args)
