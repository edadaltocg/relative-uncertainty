import torch
from tools import data_tools
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc


def eval(lbd, device, model_op, params, test_labels, test_data, logger):
    # validate
    with torch.no_grad():
        scores = model_op(test_data.to(device), params).cpu().numpy()
    fprs, tprs, thrs = roc_curve(test_labels, scores)
    roc_auc = auc(fprs, tprs)
    fpr, trp, threshold = data_tools.fpr_at_fixed_tpr(fprs, tprs, thrs, 0.95)
    logger.info(f"lbd: {lbd}, AUC: {roc_auc:.4f}, FPR: {fpr:.4f}")

    return lbd, roc_auc, fpr, fprs, tprs, thrs, trp, threshold


def model_op(probs, params):
    # double checking constraints are satisfied
    params = torch.tril(params, diagonal=-1)
    params = params + params.T
    params = params / params.norm()
    return torch.diag(probs @ params @ probs.T)


def matrix_D(train_data, train_labels, device, lbd, logger):
    data_pos = train_data[train_labels == 0]
    data_neg = train_data[train_labels == 1]
    params = (1 - lbd) * torch.einsum("ij,ik->ijk", data_pos, data_pos).mean(dim=0).to(
            device
        ) - lbd * torch.einsum("ij,ik->ijk", data_neg, data_neg).mean(dim=0).to(device)
    params = torch.relu(-params)
    params = torch.tril(params, diagonal=-1).requires_grad_()
    params = params + params.T
    if torch.all(params <= 0):
        # default to gini
        params = torch.ones(params.size()).to(device)
        params = torch.tril(params, diagonal=-1)
        params = params + params.T
    params = params / params.norm()

    data_example = train_data[0].unsqueeze(0).to(device)
    assert torch.allclose(data_example.sum(1), torch.ones(1, device=device))
    logger.info(f"Input:\n{data_example}")
    output = model_op(data_example, params)
    logger.info(f"Output:\n{output.item()}")

    return params, None


def D_scores_func(test_data, device, params):
    if params.device != device:
        params = params.to(device)
    scores = model_op(test_data.to(device), params)
    return scores


def D_perturbation(dataloader: torch.utils.data.DataLoader,
                   device: torch.device,
                   magnitude: float,
                   net: torch.nn.Module = None,
                   generator=None,
                   params=None):
    # loop over the dataset
    net = net.eval()
    net = net.to(device)
    new_inputs_list = []
    list_targets = []
    for i, (inputs, targets) in enumerate(dataloader):
        # print(inputs.shape)
        inputs, targets = inputs.to(device), targets.to(device).reshape(-1, 1)
        inputs = Variable(inputs, requires_grad=True)
        # compute output
        outputs = net(inputs)
        outputs = torch.softmax(outputs, dim=1)
        # compute perturbation
        D_scores = D_scores_func(outputs, device, params)
        log_D_scores = torch.log(D_scores)
        log_D_scores.sum().backward()
        # print(inputs)
        # print('grad:', inputs.grad)
        # exit()
        new_inputs = inputs - magnitude * torch.sign(-inputs.grad)
        # new_inputs = inputs
        new_inputs_list.append(new_inputs.detach().cpu())
        list_targets.append(targets.detach().cpu())
    new_inputs = torch.vstack(new_inputs_list)
    targets = torch.vstack(list_targets)

    td = torch.utils.data.TensorDataset(new_inputs, targets)
    # new daataloader from td
    batch_size = dataloader.batch_size
    new_dataloader = torch.utils.data.DataLoader(
        td, batch_size=batch_size, shuffle=False, num_workers=2, generator=generator,)
    # print(new_inputs.shape)
    # print(targets.shape)
    return new_dataloader


def compute_perturbed_loaders(magnitude, match_testloader, mismatch_loader, device, net, torch_gen, params):
    new_match_testloader = None
    new_mismatch_loader = None

    if magnitude > 0.:
        new_match_testloader = D_perturbation(
            dataloader=match_testloader, device=device, magnitude=magnitude, net=net, generator=torch_gen, params=params)
        new_mismatch_loader = D_perturbation(
            dataloader=mismatch_loader, device=device, magnitude=magnitude, net=net, generator=torch_gen, params=params)
    else:
        new_match_testloader = match_testloader
        new_mismatch_loader = mismatch_loader

    return new_match_testloader, new_mismatch_loader
