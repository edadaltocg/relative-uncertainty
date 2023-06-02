import os
import torch
import argparse
import numpy as np
from tools import ml_tools
from tools import data_tools
from mismatch_analysis.d_matrix import compute_d_matrix
from mismatch_analysis.d_matrix import compute_eval_scores
from tools.d_matrix_tools import compute_perturbed_loaders


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
    mismatch_dataset_name = config["mismatch_dataset_name"]
    model_seed = config["model_seed"]
    data_path = config["data_path"]
    device_id = config["device_id"]
    magnitudes = config["magnitudes"]
    temperatures = config["temperatures"]
    batch_size = config["batch_size"]
    rs = config["rs"]
    seeds = config["seeds"]
    use_mismatch_val = config["use_mismatch_val"]
    lbds = config["lbds"]
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]

    ##################
    ##################
    ##################

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    torch_gen = ml_tools.set_seed(model_seed)

    ##################
    ##################
    ##################

    dest_folder = f"./mismatch_analysis/d_matrix/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}"
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
    checkpoint_path = f"./{model_name}/{match_dataset_name}/{model_seed}/best.pth"
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
    torch.save(match_ts_targets, os.path.join(
        dest_folder, "match_ts_targets.pt"))
    torch.save(match_ts_predictions, os.path.join(
        dest_folder, "match_ts_predictions.pt"))

    # print train and test accuracy of the in-distribution data
    logger.info(
        f"Match train accuracy: {(match_tr_labels==match_tr_predictions).sum().item()/len(match_tr_labels)}")
    logger.info(
        f"Match test accuracy: {(match_ts_targets==match_ts_predictions).sum().item()/len(match_ts_targets)}")

    ##################
    ##################
    ##################

    mismatch_trainset = torch.load(
        f'./data/mismatch_data/{mismatch_dataset_name}/trainset.pt')
    mismatch_testset = torch.load(
        f'./data/mismatch_data/{mismatch_dataset_name}/testset.pt')

    # mismatch_trainset, mismatch_testset, _ = data_tools.get_data(
    #     mismatch_dataset_name, data_path)

    mismatch_trainloader = torch.utils.data.DataLoader(
        mismatch_trainset, batch_size=batch_size, shuffle=False, num_workers=2, generator=torch_gen)

    mismatch_tr_logits, mismatch_tr_labels, mismatch_tr_predictions, mismatch_tr_data = ml_tools.get_logits_labels_preds_data(
        net, mismatch_trainloader, device)

    mismatch_testloader = torch.utils.data.DataLoader(
        mismatch_testset, batch_size=batch_size, shuffle=False, num_workers=2, generator=torch_gen)

    mismatch_ts_logits, mismatch_ts_labels, mismatch_ts_predictions, mismatch_ts_data = ml_tools.get_logits_labels_preds_data(
        net, mismatch_testloader, device)

    # print train and test accuracy of the out-distribution data
    logger.info(
        f"Mismatch train accuracy: {(mismatch_tr_labels==mismatch_tr_predictions).sum().item()/len(mismatch_tr_labels)}")
    logger.info(
        f"Mismatch test accuracy: {(mismatch_ts_labels==mismatch_ts_predictions).sum().item()/len(mismatch_ts_labels)}")

    # concatenate the data and labels mismatch train set and test set and save them using torch.save
    mismatch_labels = torch.cat(
        (mismatch_tr_labels, mismatch_ts_labels), dim=0)
    mismatch_logits = torch.cat(
        (mismatch_tr_logits, mismatch_ts_logits), dim=0)
    mismatch_data = torch.cat((mismatch_tr_data, mismatch_ts_data), dim=0)

    # create mismatch dataloader from the concatenated mismatch data and mismatch labels
    mismatch_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(mismatch_data, mismatch_labels), batch_size=batch_size, shuffle=False, num_workers=2, generator=torch_gen)

    ##################
    ##################
    ##################

    # for seed in seeds:
    #     for r in rs:
    #         for lbd in lbds:
    #             compute_d_matrix.compute_d_matrix(seed=seed,
    #                                               dest_folder=dest_folder,
    #                                               match_ts_labels=match_ts_targets,
    #                                               match_ts_predictions=match_ts_predictions,
    #                                               match_ts_logits=match_ts_logits,
    #                                               use_mismatch_val=use_mismatch_val,
    #                                               mismatch_labels=mismatch_labels,
    #                                               mismatch_logits=mismatch_logits,
    #                                               r=r,
    #                                               lbd=lbd,
    #                                               logger=logger,
    #                                               device=device,
    #                                               batch_size=batch_size,
    #                                               lr=lr,
    #                                               epochs=epochs)

    ##################
    ##################
    ##################

    for seed in seeds:
        for r in rs:
            for lbd in lbds:
                folder = f"{dest_folder}/seed_{seed}/r_{r}/lr_{lr}/epochs_{epochs}/lbd_{lbd}/use_mismatch_val_{use_mismatch_val}"
                # load match and mismatch logits, labels, predictions and data
                match_labels = match_ts_targets
                match_predictions = match_ts_predictions

                # params = torch.load(os.path.join(folder, "D_matrix.pt"))
                params = torch.Tensor(np.load(os.path.join(folder, "D_matrix.npy"), allow_pickle=False))
                for magnitude in magnitudes:
                    magnitude_folder = os.path.join(
                        folder, f"magnitude_{magnitude}")
                    if not os.path.exists(magnitude_folder):
                        os.makedirs(magnitude_folder)
                    new_match_testloader, new_mismatch_loader = compute_perturbed_loaders(magnitude=magnitude,
                                                                                          match_testloader=match_testloader,
                                                                                          mismatch_loader=mismatch_dataloader,
                                                                                          device=device,
                                                                                          net=net,
                                                                                          torch_gen=torch_gen,
                                                                                          params=params)
                    new_match_ts_logits, _, _, _ = ml_tools.get_logits_labels_preds_data(
                        net, new_match_testloader, device)

                    new_mismatch_logits, _, new_mismatch_predictions, _ = ml_tools.get_logits_labels_preds_data(
                        net, new_mismatch_loader, device)

                    n = len(match_ts_targets)
                    num_val_samples = n // r

                    ml_tools.set_seed(seed)
                    new_match_perm = torch.randperm(len(match_ts_predictions))
                    print(new_match_perm)

                    new_mismatch_perm = torch.randperm(
                        len(new_mismatch_predictions))
                    print(new_mismatch_perm)

                    new_match_val_logits = new_match_ts_logits[new_match_perm][:num_val_samples]
                    new_match_val_targets = match_labels[new_match_perm][:num_val_samples]
                    new_match_val_predictions = match_predictions[new_match_perm][:num_val_samples]

                    new_mismatch_val_logits = new_mismatch_logits[new_mismatch_perm][:num_val_samples][:len(
                        new_match_ts_logits)]

                    for temperature in temperatures:
                        compute_eval_scores.compute_eval_scores(device=device,
                                                                logger=logger,
                                                                magnitude=magnitude,
                                                                magnitude_folder=magnitude_folder,
                                                                new_match_val_logits=new_match_val_logits,
                                                                new_match_val_targets=new_match_val_targets,
                                                                new_match_val_predictions=new_match_val_predictions,
                                                                new_mismatch_val_logits=new_mismatch_val_logits,
                                                                new_match_ts_logits=new_match_ts_logits,
                                                                match_labels=match_labels,
                                                                params=params,
                                                                seed=seed,
                                                                match_predictions=match_predictions,
                                                                temperature=temperature,
                                                                new_mismatch_perm=new_mismatch_perm,
                                                                new_match_perm=new_match_perm,
                                                                num_val_samples=num_val_samples,
                                                                new_mismatch_logits=new_mismatch_logits,)
