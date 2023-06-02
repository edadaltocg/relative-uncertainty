# import sys
# sys.path.append("..")
# sys.path.append("../..")

import os
import torch
import argparse
import numpy as np
from tools import ml_tools
from tools import data_tools
from corruption_analysis.d_matrix import compute_d_matrix
from tools.d_matrix_tools import compute_perturbed_loaders
from corruption_analysis.d_matrix import compute_eval_scores


if __name__ == "__main__":
    # get the config file from the command line using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, required=True)
    args = parser.parse_args()
    config = data_tools.read_config(args.config_file_path)


    ##################
    ##################
    ##################

    model_name = config["model_name"]
    match_dataset_name = config["match_dataset_name"]
    corrupted_dataset_name = config["corrupted_dataset_name"]
    model_seed = config["model_seed"]
    data_path = config["data_path"]
    magnitudes = config["magnitudes"]
    temperatures = config["temperatures"]
    batch_size = config["batch_size"]
    rs = config["rs"]
    seeds = config["seeds"]
    lbds = config["lbds"]
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    corruptions = config["corruptions"]
    intensities = config["intensities"]

    ##################
    ##################
    ##################


    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    torch_gen = ml_tools.set_seed(model_seed)

    ##################
    ##################
    ##################

    ROOT=os.environ.get("SCRATCH", ".")
    dest_folder = f"corruption_analysis/d_matrix/{match_dataset_name}_to_{corrupted_dataset_name}/{model_name}/model_seed_{model_seed}"
    dest_folder = os.path.join(ROOT, dest_folder)
    # create dest_folder if not exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    ##################
    ##################
    ##################

    logger = data_tools.get_logger(logger_name="d_matrix_script",
                                   log_file=os.path.join(dest_folder, "d_matrix_script.log"))

    ##################
    ##################
    ##################

    _, _, match_num_classes = data_tools.get_data(
        match_dataset_name, data_path)

    # get the model from the checkpoint of the match dataset
    checkpoint_path = f"{model_name}/{match_dataset_name}/{model_seed}/best.pth"
    net = ml_tools.get_model(model_name=model_name, num_classes=match_num_classes,
                             checkpoint_path=checkpoint_path)

    ##################
    ##################
    ##################

    match_trainset, match_testset, _ = data_tools.get_data(
        match_dataset_name, data_path)

    match_trainloader = torch.utils.data.DataLoader(
        match_trainset, batch_size=batch_size, shuffle=False, num_workers=2, generator=torch_gen)

    _, match_tr_labels, match_tr_predictions, _ = ml_tools.get_logits_labels_preds_data(
        net, match_trainloader, device)

    match_testloader = torch.utils.data.DataLoader(
        match_testset, batch_size=batch_size, shuffle=False, num_workers=2, generator=torch_gen)

    match_ts_logits, match_ts_targets, match_ts_predictions, match_ts_data = ml_tools.get_logits_labels_preds_data(
        net, match_testloader, device)
    # save the match test labels and predictions
    # torch.save(match_ts_targets, os.path.join(
    #     dest_folder, "match_ts_targets.pt"))
    np.save(os.path.join(dest_folder, "match_ts_targets.npy"), match_ts_targets, allow_pickle=False)
    # torch.save(match_ts_predictions, os.path.join(
    #     dest_folder, "match_ts_predictions.pt"))
    np.save(os.path.join(dest_folder, "match_ts_predictions.npy"), match_ts_predictions, allow_pickle=False)

    # print train and test accuracy of the in-distribution data
    logger.info(
        f"Match train accuracy: {(match_tr_labels==match_tr_predictions).sum().item()/len(match_tr_labels)}")
    logger.info(
        f"Match test accuracy: {(match_ts_targets==match_ts_predictions).sum().item()/len(match_ts_targets)}")

    ##################
    ##################
    ##################

    for seed in seeds:
        for r in rs:
            for lbd in lbds:
                compute_d_matrix.compute_d_matrix(seed=seed,
                                                  dest_folder=dest_folder,
                                                  match_ts_labels=match_ts_targets,
                                                  match_ts_predictions=match_ts_predictions,
                                                  match_ts_logits=match_ts_logits,
                                                  r=r,
                                                  lbd=lbd,
                                                  logger=logger,
                                                  device=device,
                                                  batch_size=batch_size,
                                                  lr=lr,
                                                  epochs=epochs)
    ##################
    ##################
    ##################

    # for corruption in corruptions:
    #     for intensity in intensities:

    #         _, corrupted_testset, _ = data_tools.get_data(
    #             corrupted_dataset_name, data_path, corruption=corruption, intensity=intensity)

    #         corrupted_testloader = torch.utils.data.DataLoader(
    #             corrupted_testset, batch_size=batch_size, shuffle=False, num_workers=2, generator=torch_gen)

    #         corrupted_ts_logits, corrupted_ts_labels, corrupted_ts_predictions, corrupted_ts_data = ml_tools.get_logits_labels_preds_data(
    #             net, corrupted_testloader, device)

    #         # save the mismatch test labels and predictions
    #         torch.save(corrupted_ts_labels, os.path.join(
    #             dest_folder, "mismatch_ts_targets.pt"))
    #         torch.save(corrupted_ts_predictions, os.path.join(
    #             dest_folder, "mismatch_ts_predictions.pt"))

    #         # print train and test accuracy of the out-distribution data
    #         logger.info(
    #             f"Mismatch test accuracy: {(corrupted_ts_labels==corrupted_ts_predictions).sum().item()/len(corrupted_ts_labels)}")

    #         # create mismatch dataloader from the concatenated mismatch data and mismatch labels
    #         corrupted_dataloader = torch.utils.data.DataLoader(
    #             torch.utils.data.TensorDataset(corrupted_ts_data, corrupted_ts_labels), batch_size=batch_size, shuffle=False, num_workers=2, generator=torch_gen)

    #         ##################
    #         ##################
    #         ##################

    #         for seed in seeds:
    #             for r in rs:
    #                 for lbd in lbds:
    #                     folder = f"{dest_folder}/seed_{seed}/r_{r}/lr_{lr}/epochs_{epochs}/lbd_{lbd}/{corruption}_{intensity}"
    #                     params_folder = f"{dest_folder}/seed_{seed}/r_{r}/lr_{lr}/epochs_{epochs}/lbd_{lbd}"
    #                     # load match and mismatch logits, labels, predictions and data
    #                     match_labels = match_ts_targets
    #                     match_predictions = match_ts_predictions
    #                     corrupted_labels = corrupted_ts_labels
    #                     corrupted_predictions = corrupted_ts_predictions

    #                     params = torch.load(
    #                         os.path.join(params_folder, "D_matrix.pt"))
    #                     for magnitude in magnitudes:
    #                         magnitude_folder = os.path.join(
    #                             folder, f"magnitude_{magnitude}")
    #                         if not os.path.exists(magnitude_folder):
    #                             os.makedirs(magnitude_folder)
    #                         new_match_testloader, new_corrupted_loader = compute_perturbed_loaders(magnitude=magnitude,
    #                                                                                                match_testloader=match_testloader,
    #                                                                                                mismatch_loader=corrupted_dataloader,
    #                                                                                                device=device,
    #                                                                                                net=net,
    #                                                                                                torch_gen=torch_gen,
    #                                                                                                params=params)
    #                         new_match_ts_logits, _, _, _ = ml_tools.get_logits_labels_preds_data(
    #                             net, new_match_testloader, device)

    #                         new_corrupted_logits, new_corrupted_labels, new_corrupted_predictions, _ = ml_tools.get_logits_labels_preds_data(
    #                             net, new_corrupted_loader, device)

    #                         n = len(match_ts_targets)
    #                         num_val_samples = n // r

    #                         ml_tools.set_seed(seed)
    #                         new_match_perm = torch.randperm(
    #                             len(match_ts_predictions))

    #                         new_match_val_logits = new_match_ts_logits[new_match_perm][:num_val_samples]
    #                         new_match_val_targets = match_labels[new_match_perm][:num_val_samples]
    #                         new_match_val_predictions = match_predictions[new_match_perm][:num_val_samples]

    #                         for temperature in temperatures:
    #                             compute_eval_scores.compute_eval_scores(device=device,
    #                                                                     logger=logger,
    #                                                                     magnitude=magnitude,
    #                                                                     magnitude_folder=magnitude_folder,
    #                                                                     new_match_val_logits=new_match_val_logits,
    #                                                                     new_match_val_targets=new_match_val_targets,
    #                                                                     new_match_val_predictions=new_match_val_predictions,
    #                                                                     params=params,
    #                                                                     seed=seed,
    #                                                                     temperature=temperature,
    #                                                                     new_corrupted_logits=new_corrupted_logits,
    #                                                                     new_corrupted_labels=new_corrupted_labels,
    #                                                                     new_corrupted_predictions=new_corrupted_predictions)
