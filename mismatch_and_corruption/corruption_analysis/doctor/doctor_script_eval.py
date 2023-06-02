import os
import torch
import argparse
from tools import ml_tools
from tools import data_tools
from tools import doctor_tools
from corruption_analysis.doctor import eval_doctor_scores
from tools.doctor_tools import compute_doctor_scores_for_magnitude_and_temperature


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
    device_id = config["device_id"]
    magnitudes = config["magnitudes"]
    temperatures = config["temperatures"]
    batch_size = config["batch_size"]
    rs = config["rs"]
    seeds = config["seeds"]
    intensities = config["intensities"]
    corruptions = config["corruptions"]

    ##################
    ##################
    ##################

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    ##################
    ##################
    ##################

    ROOT=os.environ.get("SCRATCH", ".")
    dest_folder = f"corruption_analysis/doctor/{match_dataset_name}_to_{corrupted_dataset_name}/{model_name}/model_seed_{model_seed}"
    dest_folder = os.path.join(ROOT, dest_folder)
    # create dest_folder if not exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder, exist_ok=True)

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
    checkpoint_path = f"{model_name}/{match_dataset_name}/{model_seed}/best.pth"
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
    if not os.path.exists(os.path.join(dest_folder, "match_ts_targets.pt")):
        torch.save(match_ts_targets, os.path.join(
            dest_folder, "match_ts_targets.pt"))
    if not os.path.exists(os.path.join(dest_folder, "match_ts_predictions.pt")):
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

    for corruption in corruptions:
        for intensity in intensities:
            dest_folder_corruption = os.path.join(
                dest_folder, f"{corruption}_{intensity}")
            # create dest_folder if not exists
            if not os.path.exists(dest_folder_corruption):
                os.makedirs(dest_folder_corruption, exist_ok=True)
            _, corrupted_testset, _ = data_tools.get_data(
                corrupted_dataset_name, data_path, intensity=intensity, corruption=corruption)

            corrupted_testloader = torch.utils.data.DataLoader(
                corrupted_testset, batch_size=batch_size, shuffle=False, num_workers=2, generator=torch_gen)

            _, corrupted_ts_labels, corrupted_ts_predictions, corrupted_ts_data = ml_tools.get_logits_labels_preds_data(
                net, corrupted_testloader, device)

            # save the corrupted test targets and predictions
            torch.save(corrupted_ts_labels, os.path.join(
                dest_folder_corruption, "corrupted_targets.pt"))
            torch.save(corrupted_ts_predictions, os.path.join(
                dest_folder_corruption, "corrupted_predictions.pt"))

            logger.info(
                f"corrupted test accuracy: {(corrupted_ts_labels==corrupted_ts_predictions).sum().item()/len(corrupted_ts_labels)}")

            # create corrupted dataloader from the concatenated corrupted data and corrupted labels
            corrupted_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(corrupted_ts_data, corrupted_ts_labels),
                                                               batch_size=batch_size,
                                                               shuffle=False,
                                                               num_workers=2,
                                                               generator=torch_gen)

            ##################
            ##################
            ##################

            # for magnitude in magnitudes:
            #     new_match_testloader, new_corrupted_loader = doctor_tools.apply_perturbation(
            #         match_testloader=match_testloader,
            #         mismatch_dataloader=corrupted_dataloader,
            #         magnitude=magnitude,
            #         net=net,
            #         device=device,
            #         torch_gen=torch_gen,
            #         logger=logger)

            #     new_match_ts_logits, _, _, _ = ml_tools.get_logits_labels_preds_data(
            #         net, new_match_testloader, device)

            #     new_corrupted_logits, _, _, _ = ml_tools.get_logits_labels_preds_data(
            #         net, new_corrupted_loader, device)

            #     for temperature in temperatures:
            #         compute_doctor_scores_for_magnitude_and_temperature(temperature=temperature,
            #                                                             magnitude=magnitude,
            #                                                             logger=logger,
            #                                                             new_match_ts_logits=new_match_ts_logits,
            #                                                             new_mismatch_logits=new_corrupted_logits,
            #                                                             dest_folder=dest_folder_corruption)

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
                                corrupted_dataset_name=corrupted_dataset_name,
                                model_seed=model_seed,
                                logger=logger,
                                intensity=intensity,
                                corruption=corruption)
    # import time
    # time.sleep(60*10)
