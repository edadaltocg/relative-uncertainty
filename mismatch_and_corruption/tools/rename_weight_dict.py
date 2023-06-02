import torch
from collections import OrderedDict

if __name__ == '__main__':

    path_list = [
        'resnet34_custom_mixup/cifar10/1/old_best.pth',
        'resnet34_custom_mixup/cifar100/1/old_best.pth',
        'resnet34_custom_regmixup/cifar10/1/old_best.pth',
        'resnet34_custom_regmixup/cifar100/1/old_best.pth',
        'densenet121_custom_mixup/cifar10/1/old_best.pth',
        'densenet121_custom_mixup/cifar100/1/old_best.pth',
        'densenet121_custom_regmixup/cifar10/1/old_best.pth',
        'densenet121_custom_regmixup/cifar100/1/old_best.pth',
    ]

    for path in path_list:
        # # load the state dict
        # state_dict = torch.load(path, map_location='cpu')
        # # create a new state dict
        # new_state_dict = OrderedDict()
        # # iterate over the keys
        # for key in state_dict.keys():
        #     # if the key starts with module
        #     if key.startswith('module.'):
        #         # remove the module. prefix
        #         new_key = key[7:]
        #     # if the key doesn't start with module
        #     else:
        #         # keep the key as is
        #         new_key = key
        #     # add the new key and value to the new state dict
        #     new_state_dict[new_key] = state_dict[key]
        # # save the new state dict
        # old_best_path = path.replace('best.pth', 'old_best.pth')
        w = torch.load(path, map_location='cpu')
        w = {k.replace("module.", ""): v for k, v in w.items()}
        if torch.cuda.is_available():
            w = {k: v.cuda() for k, v in w.items()}
        else:
            w = {k: v.cpu() for k, v in w.items()}
        torch.save(w, path.replace('old_best.pth', 'best.pth'))