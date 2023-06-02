import os
import torch
import argparse
from tools import ml_tools
from tools import data_tools
from tools import doctor_tools
from mismatch_analysis.doctor import eval_doctor_scores
from mismatch_analysis.doctor import compute_doctor_scores


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

    ##################
    ##################
    ##################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##################
    ##################
    ##################

    dest_folder = f"./mismatch_analysis/doctor/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}"
    # create dest_folder if not exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    ##################
    ##################
    ##################

    logger = data_tools.get_logger(logger_name="doctor_script",
                                   log_file=os.path.join(dest_folder, "doctor_script.log"))

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

    torch_gen = ml_tools.set_seed(model_seed)

    match_trainset, match_testset, _ = data_tools.get_data(
        match_dataset_name, data_path)

    match_trainloader = torch.utils.data.DataLoader(
        match_trainset, batch_size=batch_size, shuffle=False, num_workers=2, generator=torch_gen)

    _, match_tr_labels, match_tr_predictions, _ = ml_tools.get_logits_labels_preds_data(
        net, match_trainloader, device)

    match_testloader = torch.utils.data.DataLoader(
        match_testset, batch_size=batch_size, shuffle=False, num_workers=2, generator=torch_gen)

    _, match_ts_targets, match_ts_predictions, _ = ml_tools.get_logits_labels_preds_data(
        net, match_testloader, device)
    # save the match test targets and predictions
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

    _, mismatch_tr_labels, mismatch_tr_predictions, mismatch_tr_data = ml_tools.get_logits_labels_preds_data(
        net, mismatch_trainloader, device)

    mismatch_testloader = torch.utils.data.DataLoader(
        mismatch_testset, batch_size=batch_size, shuffle=False, num_workers=2, generator=torch_gen)

    _, mismatch_ts_labels, mismatch_ts_predictions, mismatch_ts_data = ml_tools.get_logits_labels_preds_data(
        net, mismatch_testloader, device)

    # print train and test accuracy of the out-distribution data
    logger.info(
        f"Mismatch train accuracy: {(mismatch_tr_labels==mismatch_tr_predictions).sum().item()/len(mismatch_tr_labels)}")
    logger.info(
        f"Mismatch test accuracy: {(mismatch_ts_labels==mismatch_ts_predictions).sum().item()/len(mismatch_ts_labels)}")

    # concatenate the data and labels mismatch train set and test set and save them using torch.save
    mismatch_labels = torch.cat(
        (mismatch_tr_labels, mismatch_ts_labels), dim=0)
    mismatch_data = torch.cat((mismatch_tr_data, mismatch_ts_data), dim=0)

    # create mismatch dataloader from the concatenated mismatch data and mismatch labels
    mismatch_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(mismatch_data, mismatch_labels), batch_size=batch_size, shuffle=False, num_workers=2, generator=torch_gen)

    ##################
    ##################
    ##################

    for magnitude in magnitudes:
        new_match_testloader, new_mismatch_loader = doctor_tools.apply_perturbation(
            match_testloader=match_testloader,
            mismatch_dataloader=mismatch_dataloader,
            magnitude=magnitude,
            net=net,
            device=device,
            torch_gen=torch_gen,
            logger=logger)

        new_match_ts_logits, _, _, _ = ml_tools.get_logits_labels_preds_data(
            net, new_match_testloader, device)

        new_mismatch_logits, _, _, _ = ml_tools.get_logits_labels_preds_data(
            net, new_mismatch_loader, device)

        for temperature in temperatures:
            compute_doctor_scores.compute_doctor_scores_for_magnitude_and_temperature(
                temperature=temperature,
                magnitude=magnitude,
                logger=logger,
                new_match_ts_logits=new_match_ts_logits,
                new_mismatch_logits=new_mismatch_logits,
                dest_folder=dest_folder)

    ##################
    ##################
    ##################

    for seed in seeds:
        for magnitude in magnitudes:
            for temperature in temperatures:
                for r in rs:
                    eval_doctor_scores.eval_doctor_scores(
                        magnitude=magnitude,
                        temperature=temperature,
                        r=r,
                        seed=seed,
                        model_name=model_name,
                        match_dataset_name=match_dataset_name,
                        mismatch_dataset_name=mismatch_dataset_name,
                        model_seed=model_seed,
                        logger=logger)
