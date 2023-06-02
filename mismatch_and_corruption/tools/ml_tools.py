import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch import nn as nn
from tools.models.resnet import ResNet34
import torch.nn.functional as torch_func
from torchvision import models as models
from tools.models.densenet import DenseNet121Small


def get_accuracy(predictions, targets):
    """
    Compute the accuracy given predictions and targets
    :param predictions: predictions
    :param targets: targets
    :return: accuracy
    """
    return torch.sum(predictions == targets).item() / len(targets)


def get_logits_labels_preds_data(model: torch.nn.Module,
                                 dataloader: torch.utils.data.DataLoader,
                                 device: torch.device,
                                 *args, **kwargs):
    """
    Compute the logits given input data loader and model
    :param model: model utilized for the logits computation
    :param dataloader: loader for the training data
    :param device: device used for computation
    :return: logits, predictions, targets, and data
    """
    logits_lst = []
    targets_lst = []
    predictions_lst = []
    data_lst = []

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(dataloader),
                                              desc='Computation ongoing...',
                                              ascii=True,
                                              total=len(dataloader)):
            logits = model(data.to(device))
            logits_lst.append(logits.detach().cpu())

            predictions = torch.argmax(
                torch_func.softmax(logits, dim=1), dim=1)
            predictions_lst.append(predictions.detach().cpu().reshape(-1, 1))

            targets_lst.append(target.detach().cpu().reshape(-1, 1))

            data_lst.append(data.detach().cpu())

    logits = torch.vstack(logits_lst)
    data = torch.vstack(data_lst)
    predictions = torch.vstack(predictions_lst).reshape(-1)
    targets = torch.vstack(targets_lst).reshape(-1)
    if 'save_to_folder' in kwargs:
        try:
            os.makedirs(kwargs['save_to_folder'], exist_ok=True)
            # torch.save(obj=logits, f=kwargs['save_to_folder'] + 'logits.pt')
            np.save(kwargs['save_to_folder'] + 'logits.npy', logits.detach().cpu().numpy(), allow_pickle=False)
            # torch.save(obj=predictions,
            #            f=kwargs['save_to_folder'] + 'predictions.pt')
            np.save(kwargs['save_to_folder'] + 'predictions.npy', predictions.detach().cpu().numpy(), allow_pickle=False)
            # torch.save(obj=targets, f=kwargs['save_to_folder'] + 'targets.pt')
            np.save(kwargs['save_to_folder'] + 'targets.npy', targets.detach().cpu().numpy(), allow_pickle=False)
            # torch.save(obj=data, f=kwargs['save_to_folder'] + 'data.pt')
            np.save(kwargs['save_to_folder'] + 'data.npy', data.detach().cpu().numpy(), allow_pickle=False)
        except FileNotFoundError as e:
            print(e)
    return logits, targets, predictions, data

# define an abstract class for torchvision model


class TorchvisionModel(nn.Module):
    def __init__(self):
        super(TorchvisionModel, self).__init__()
        self.model = None

    def get_model(self):
        pass

    def forward(self, x):
        pass

# define class densenet121 for torch model densenet121 inherited from TorchModel


class DenseNet121Torchvision(TorchvisionModel):
    def __init__(self, pretrained=True):
        super(DenseNet121Torchvision, self).__init__()
        self.model = models.densenet121(pretrained=pretrained)

    def get_model(self):
        return self.model

    def forward(self, x):
        return self.model(x)


class DenseNet121Custom(TorchvisionModel):
    def __init__(self, checkpoint_path=None, num_classes=10):
        super(DenseNet121Custom, self).__init__()
        self.model = DenseNet121Small(num_classes=num_classes)
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    def get_model(self):
        return self.model

    def forward(self, x):
        return self.model(x)


class ResNet34Custom(TorchvisionModel):
    def __init__(self, checkpoint_path=None, num_classes=10):
        super(ResNet34Custom, self).__init__()
        self.model = ResNet34(num_classes=num_classes)
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    def get_model(self):
        return self.model

    def forward(self, x):
        return self.model(x)

# define method to return model from string


def get_model_(model_name, **kwargs):
    if model_name == 'densenet121':
        return DenseNet121Torchvision(pretrained=kwargs['pretrained'])
    elif model_name == 'densenet121_custom' or model_name == 'densenet121_custom_lognorm' or model_name == 'densenet121_custom_mixup' or model_name == 'densenet121_custom_regmixup':
        return DenseNet121Custom(checkpoint_path=kwargs['checkpoint_path'], num_classes=kwargs['num_classes'])
    elif model_name == 'resnet34_custom' or model_name == 'resnet34_custom_lognorm' or model_name == 'resnet34_custom_mixup' or model_name == 'resnet34_custom_regmixup':
        return ResNet34Custom(checkpoint_path=kwargs['checkpoint_path'], num_classes=kwargs['num_classes'])
    else:
        raise NotImplementedError
    
def get_model(model_name, num_classes, checkpoint_path):
    net = get_model_(
        model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)
    return net


def set_seed(seed):
    """
    Set seed for reproducibility
    :param seed: seed
    :return: generator for torch data loader
    """
    # set seed for cuda and numpy
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch_gen = torch.Generator()
    torch_gen.manual_seed(seed)
    return torch_gen
