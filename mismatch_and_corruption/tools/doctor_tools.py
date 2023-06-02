import os
import time
import torch
import logging
import numpy as np
from torch.autograd import Variable


def g(logits: torch.Tensor, temperature: float = 1.0):
    return torch.sum(torch.softmax(logits/temperature, dim=1) ** 2, dim=1)


def doctor(logits: torch.Tensor, temperature: float = 1.0):
    return 1 - g(logits=logits, temperature=temperature)


def doctor_perturbation(dataloader: torch.utils.data.DataLoader,
                        device: torch.device,
                        magnitude: float,
                        temperature: float = 1.0,
                        net: torch.nn.Module = None,
                        generator=None):
    # loop over the dataset
    net = net.eval()
    net = net.to(device)
    new_inputs_list = []
    list_targets = []
    for _, (inputs, targets) in enumerate(dataloader):
        # print(inputs.shape)
        inputs, targets = inputs.to(device), targets.to(device).reshape(-1, 1)
        inputs = Variable(inputs, requires_grad=True)
        # compute output
        outputs = net(inputs)
        # compute perturbation
        doctor_scores = doctor(
            logits=outputs, temperature=temperature)
        log_doctor_scores = torch.log(doctor_scores)
        # log_doctor_scores.backward(torch.ones_like(log_doctor_scores))
        log_doctor_scores.sum().backward()
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


def apply_perturbation(match_testloader: torch.utils.data.DataLoader,
                       mismatch_dataloader: torch.utils.data.DataLoader,
                       magnitude: float,
                       net: torch.nn.Module,
                       device: torch.device,
                       torch_gen: torch.Generator,
                       logger: logging.Logger) -> tuple:
    """
    Apply perturbation to the dataloader

    Args:
        match_testloader (torch.utils.data.DataLoader): dataloader for match test
        mismatch_dataloader (torch.utils.data.DataLoader): dataloader for mismatch test
        magnitude (float): magnitude of the perturbation
        net (torch.nn.Module): model to be perturbed
        device (torch.device): device to be used
        torch_gen (torch.Generator): torch generator for dataloader
        logger (logging.Logger): logger to log the perturbation

    Returns:
        new_match_testloader (torch.utils.data.DataLoader): perturbed dataloader for match test
        new_mismatch_loader (torch.utils.data.DataLoader): perturbed dataloader for mismatch test
    """
    new_match_testloader = match_testloader
    new_mismatch_loader = mismatch_dataloader

    if magnitude > 0.:
        # compute time here
        start_time = time.time()
        logger.info(f"Perturbing with magnitude {magnitude}")
        new_match_testloader = doctor_perturbation(
            dataloader=match_testloader, device=device, magnitude=magnitude, temperature=1.0, net=net, generator=torch_gen)
        new_mismatch_loader = doctor_perturbation(
            dataloader=mismatch_dataloader, device=device, magnitude=magnitude, temperature=1.0, net=net, generator=torch_gen)
        logger.info(
            f"Time taken for perturbation: {time.time() - start_time} seconds")

    return new_match_testloader, new_mismatch_loader


def compute_doctor_scores_for_magnitude_and_temperature(temperature,
                                                        magnitude,
                                                        logger,
                                                        new_match_ts_logits,
                                                        new_mismatch_logits,
                                                        dest_folder) -> None:
    magnitude_folder = '/'.join((dest_folder, f"magnitude_{magnitude}"))
    if not os.path.exists(magnitude_folder):
        os.makedirs(magnitude_folder, exist_ok=True)
    temperature_folder = '/'.join((magnitude_folder,
                                   f"/temperature_{temperature}"))
    if not os.path.exists(temperature_folder):
        os.makedirs(temperature_folder, exist_ok=True)
    logger.info(f"Magnitude {magnitude}\tTemperature: {temperature}")
    new_match_ts_doctor_scores = doctor(
        logits=new_match_ts_logits, temperature=temperature)
    new_mismatch_ts_doctor_scores = doctor(
        logits=new_mismatch_logits, temperature=temperature)

    # save doctor scores
    # torch.save(new_match_ts_doctor_scores,
    #            f"{temperature_folder}/match_ts_doctor_scores.pt")
    np.save(f"{temperature_folder}/match_ts_doctor_scores.npy",
            new_match_ts_doctor_scores.cpu().numpy(),
            allow_pickle=False)

    # torch.save(new_mismatch_ts_doctor_scores,
    #            f"{temperature_folder}/mismatch_doctor_scores.pt")
    np.save(f"{temperature_folder}/mismatch_doctor_scores.npy",
            new_mismatch_ts_doctor_scores.cpu().numpy(),
            allow_pickle=False)
            
