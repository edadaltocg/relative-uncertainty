import os
import torch
import numpy as np
from tools import ml_tools
from tools.data_tools import get_data, get_CIFAR100_class_names


def get_id_classes_to_eliminate(names: list, dict_classes: dict) -> list:
    """ Get id of classes to eliminate
    Args:
        names (list): list of class names to eliminate
        dict_classes (dict): dictionary with key as class id and value as class name

    Returns:
        id_classes_to_eliminate (list): list of class ids to eliminate
    """
    id_classes_to_eliminate = []
    # assert that names are in dict_classes
    for name in names:
        assert name in dict_classes.values(
        ), f'Name {name} not in dict_classes'

    # get id of classes to eliminate
    for key, value in dict_classes.items():
        if value in names:
            id_classes_to_eliminate.append(key)

    return id_classes_to_eliminate


def eliminate_classes(data: torch.Tensor, labels: torch.Tensor, classes: list):
    """ Eliminate classes from data and labels
    Args:
        data (list): list of data
        labels (list): list of labels
        classes (list): list of classes to eliminate

    Returns:
        new_data (list): list of data without classes to eliminate
        new_labels (list): list of labels without classes to eliminate
    """
    # assert that data and labels have the same length
    assert len(data) == len(
        labels), f'Length of data {len(data)} and labels {len(labels)} are different'

    # get indexes of classes to eliminate
    indexes_to_eliminate = []
    for i in range(len(labels)):
        if labels[i] in classes:
            indexes_to_eliminate.append(i)

    # eliminate classes from data and labels
    new_data = np.delete(data, indexes_to_eliminate, axis=0)
    new_labels = np.delete(labels, indexes_to_eliminate, axis=0)

    return new_data, new_labels


def main():
    trainset, testset, _ = get_data('cifar100', 'data')
    # get class names
    class_names = get_CIFAR100_class_names('data')
    # create dictionary for class names where key is class id and value is class name
    class_names_dict = {}
    for i in range(len(class_names)):
        class_names_dict[i] = class_names[i]

    names = ['bus',
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
             'wolf']

    id_classes_to_eliminate = get_id_classes_to_eliminate(
        names=names, dict_classes=class_names_dict)

    net = ml_tools.get_model(model_name='densenet121_custom', num_classes=100,
                             checkpoint_path=f"densenet121_custom/cifar100/1/best.pth").eval()

    ###
    ###
    ###

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    _, train_labels, train_predictions, train_data = ml_tools.get_logits_labels_preds_data(
        net, torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=False), device=device)
    print(f'Train data shape: {train_data.shape}')
    print(f'Train labels shape: {train_labels.shape}')
    print(
        f'Accuracy on train set: {(train_labels == train_predictions).sum() / len(train_labels)}')

    # eliminate classes and create new train data
    new_train_data, new_train_labels = eliminate_classes(
        train_data, train_labels, id_classes_to_eliminate)
    print(f'New train data shape: {new_train_data.shape}')
    print(f'New train labels shape: {new_train_labels.shape}')

    new_trainset = torch.utils.data.TensorDataset(
        new_train_data, new_train_labels)
    _, new_train_labels, new_train_predictions, _ = ml_tools.get_logits_labels_preds_data(
        net, torch.utils.data.DataLoader(new_trainset, batch_size=1000, shuffle=False), device=device)
    print(
        f'New accuracy on train set: {(new_train_labels == new_train_predictions).sum() / len(new_train_labels)}')

    ###

    _, test_labels, test_predictions, test_data = ml_tools.get_logits_labels_preds_data(
        net, torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False), device=device)
    print(f'Test data shape: {test_data.shape}')
    print(f'Test labels shape: {test_labels.shape}')
    print(
        f'Accuracy on test set: {(test_labels == test_predictions).sum() / len(test_labels)}')

    # create new test data
    new_test_data, new_test_labels = eliminate_classes(
        test_data, test_labels, id_classes_to_eliminate)
    print(f'New test data shape: {new_test_data.shape}')
    print(f'New test labels shape: {new_test_labels.shape}')

    new_testset = torch.utils.data.TensorDataset(
        new_test_data, new_test_labels)
    _, new_test_labels, new_test_predictions, _ = ml_tools.get_logits_labels_preds_data(
        net, torch.utils.data.DataLoader(new_testset, batch_size=1000, shuffle=False), device=device)
    print(
        f'New accuracy on test set: {(new_test_labels == new_test_predictions).sum() / len(new_test_labels)}')

    ###
    ###
    ###

    dest_folder = 'data/mismatch_data/cifar100'
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # save new train data and labels and new trainset using torch
    torch.save(new_train_data, os.path.join(dest_folder, 'train_data.pt'))
    torch.save(new_train_labels, os.path.join(
        dest_folder, 'train_labels.pt'))
    torch.save(new_trainset, os.path.join(dest_folder, 'trainset.pt'))

    # save new test data and labels and new testset using torch
    torch.save(new_test_data, os.path.join(dest_folder, 'test_data.pt'))
    torch.save(new_test_labels, os.path.join(
        dest_folder, 'test_labels.pt'))
    torch.save(new_testset, os.path.join(dest_folder, 'testset.pt'))


if __name__ == '__main__':
    main()
