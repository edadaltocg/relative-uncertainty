import io
import os
import torch
import numpy as np
from tools import data_tools, ml_tools
from sklearn.metrics import auc, roc_curve
from tools.data_tools import fpr_at_fixed_tpr

# data pos = 0 in-distr and/or correct
# data neg = 1 out-distr and/or incorrect


def eval_doctor_scores(magnitude,
                       temperature,
                       r,
                       seed,
                       model_name,
                       match_dataset_name,
                       corrupted_dataset_name,
                       model_seed,
                       corruption,
                       intensity,
                       logger):
    ROOT=os.environ.get("SCRATCH", ".")
    source_folder = f'corruption_analysis/doctor/{match_dataset_name}_to_{corrupted_dataset_name}/{model_name}/model_seed_{model_seed}/{corruption}_{intensity}'
    source_folder = os.path.join(ROOT, source_folder)
    source_folder_no_corruption_no_intensity = f'corruption_analysis/doctor/{match_dataset_name}_to_{corrupted_dataset_name}/{model_name}/model_seed_{model_seed}'
    source_folder_no_corruption_no_intensity = os.path.join(ROOT, source_folder_no_corruption_no_intensity)
    # source_folder_no_corruption_no_intensity = '/'.join(source_folder.split('/')[:-1])
    dest_folder = f'{source_folder}/results/r_{r}/seed_{seed}'
    dest_folder = os.path.join(ROOT, dest_folder)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    source_magnitude_folder = '/'.join((source_folder,
                                        f"magnitude_{magnitude}"))
    source_temperature_folder = '/'.join((source_magnitude_folder,
                                          f"temperature_{temperature}"))

    # match_ts_doctor_scores = torch.load(
    #     f"{source_temperature_folder}/match_ts_doctor_scores.pt")
    match_ts_doctor_scores = torch.Tensor(np.load(
        f"{source_temperature_folder}/match_ts_doctor_scores.npy", allow_pickle=False))
    # match_ts_targets = torch.load(
    #     f"{source_folder_no_corruption_no_intensity}/match_ts_targets.pt")
    match_ts_targets = torch.Tensor(np.load(
        f"{source_folder_no_corruption_no_intensity}/match_ts_targets.npy", allow_pickle=False))
    # match_ts_predictions = torch.load(
    #     f"{source_folder_no_corruption_no_intensity}/match_ts_predictions.pt")
    match_ts_predictions = torch.Tensor(np.load(
        f"{source_folder_no_corruption_no_intensity}/match_ts_predictions.npy", allow_pickle=False))

    # corrupted_doctor_scores = torch.load(
    #     f"{source_temperature_folder}/mismatch_doctor_scores.pt")
    corrupted_doctor_scores = torch.Tensor(np.load(
        f"{source_temperature_folder}/mismatch_doctor_scores.npy", allow_pickle=False))
    # corrupted_targets = torch.load(
    #     f"{source_folder}/corrupted_targets.pt")
    corrupted_targets = torch.Tensor(np.load(
        f"{source_folder}/corrupted_targets.npy", allow_pickle=False))
    # corrupted_predictions = torch.load(
    #     f"{source_folder}/corrupted_predictions.pt")
    corrupted_predictions = torch.Tensor(np.load(
        f"{source_folder}/corrupted_predictions.npy", allow_pickle=False))



    n = len(match_ts_doctor_scores)
    num_val_samples = n // r

    _ = ml_tools.set_seed(seed)
    # perm arrays
    match_ts_perm = torch.randperm(len(match_ts_doctor_scores))
    match_ts_doctor_scores = match_ts_doctor_scores[match_ts_perm]
    match_ts_targets = match_ts_targets[match_ts_perm]
    match_ts_predictions = match_ts_predictions[match_ts_perm]

    # Validation only a subset of the non corrupted data
    doctor_val_scores = match_ts_doctor_scores[:num_val_samples]
    logger.info(
        f"Doctor baseline with temperature {temperature} and magnitude {magnitude}")
    logger.info(
        f"Number of val scores: {len(doctor_val_scores)}")
    doctor_val_binary_labels = match_ts_targets[:
                                                num_val_samples] != match_ts_predictions[:num_val_samples]

    doctor_val_fprs, doctor_val_tprs, doctor_val_thresholds = roc_curve(
        doctor_val_binary_labels, doctor_val_scores)
    doctor_val_fpr, doctor_val_tpr, doctor_val_threshold = fpr_at_fixed_tpr(
        doctor_val_fprs, doctor_val_tprs, doctor_val_thresholds, tpr_level=0.95)
    doctor_val_auc = auc(doctor_val_fprs, doctor_val_tprs)
    logger.info(
        f"Doctor validation baseline with temperature {temperature} and magnitude {magnitude}")
    logger.info(f"AUC val: {doctor_val_auc}")
    logger.info(f"FPR val: {doctor_val_fpr}")

    # save results
    final_dest_folder = f'{dest_folder}/magnitude_{magnitude}/temperature_{temperature}'
    if not os.path.exists(final_dest_folder):
        os.makedirs(final_dest_folder)
    torch.save(
        doctor_val_fprs, f'{final_dest_folder}/doctor_val_fprs.pt')
    # np.save(f'{final_dest_folder}/doctor_val_fprs.npy', doctor_val_fprs.detach().cpu().numpy(), allow_pickle=False)
    logger.info(f"Saved doctor_val_fprs.pt to {final_dest_folder}")
    torch.save(
        doctor_val_tprs, f'{final_dest_folder}/doctor_val_tprs.pt')
    # np.save(f'{final_dest_folder}/doctor_val_tprs.npy', doctor_val_tprs.detach().cpu().numpy(), allow_pickle=False)
    logger.info(f"Saved doctor_val_tprs.pt to {final_dest_folder}")
    torch.save(
        doctor_val_thresholds,  f'{final_dest_folder}/doctor_val_thresholds.pt')
    # np.save(f'{final_dest_folder}/doctor_val_thresholds.npy', doctor_val_thresholds.detach().cpu().numpy(), allow_pickle=False)
    logger.info(f"Saved doctor_val_thresholds.pt to {final_dest_folder}")
    torch.save(
        doctor_val_fpr, f'{final_dest_folder}/doctor_val_fpr.pt')
    # np.save(f'{final_dest_folder}/doctor_val_fpr.npy', doctor_val_fpr.detach().cpu().numpy(), allow_pickle=False)
    logger.info(f"Saved doctor_val_fpr.pt to {final_dest_folder}")
    torch.save(
        doctor_val_tpr, f'{final_dest_folder}/doctor_val_tpr.pt')
    # np.save(f'{final_dest_folder}/doctor_val_tpr.npy', doctor_val_tpr.detach().cpu().numpy(), allow_pickle=False)
    logger.info(f"Saved doctor_val_tpr.pt to {final_dest_folder}")
    torch.save(
        doctor_val_threshold, f'{final_dest_folder}/doctor_val_threshold.pt')
    # np.save(f'{final_dest_folder}/doctor_val_threshold.npy', doctor_val_threshold.detach().cpu().numpy(), allow_pickle=False)
    logger.info(f"Saved doctor_val_threshold.pt to {final_dest_folder}")
    torch.save(
        doctor_val_auc, f'{final_dest_folder}/doctor_val_auc.pt')
    # np.save(f'{final_dest_folder}/doctor_val_auc.npy', doctor_val_auc.detach().cpu().numpy(), allow_pickle=False)
    logger.info(f"Saved doctor_val_auc.pt to {final_dest_folder}")

    # Test: only the corrupted data

    doctor_test_scores = corrupted_doctor_scores

    logger.info(
        f"Doctor baseline with temperature {temperature} and magnitude {magnitude}")
    logger.info(
        f"Number of test scores: {len(doctor_test_scores)}")

    # assign labels for binary classification: 0 for in-test, 1 for out-test
    doctor_test_labels = corrupted_targets != corrupted_predictions
    doctor_test_fprs, doctor_test_tprs, doctor_test_thresholds = roc_curve(
        doctor_test_labels, doctor_test_scores)
    doctor_test_fpr, doctor_test_tpr, doctor_test_threshold = fpr_at_fixed_tpr(
        doctor_test_fprs, doctor_test_tprs, doctor_test_thresholds, tpr_level=0.95)
    doctor_test_auc = auc(doctor_test_fprs, doctor_test_tprs)
    logger.info(
        f"Doctor test baseline with temperature {temperature} and magnitude {magnitude}")
    print(f"AUC test: {doctor_test_auc}")
    print(f"FPR test: {doctor_test_fpr}")

    # save results
    torch.save(
        doctor_test_fprs, f'{final_dest_folder}/doctor_test_fprs.pt')
    # np.save(f'{final_dest_folder}/doctor_test_fprs.npy', doctor_test_fprs.detach().cpu().numpy(), allow_pickle=False)
    torch.save(
        doctor_test_tprs, f'{final_dest_folder}/doctor_test_tprs.pt')
    # np.save(f'{final_dest_folder}/doctor_test_tprs.npy', doctor_test_tprs.detach().cpu().numpy(), allow_pickle=False)
    torch.save(
        doctor_test_thresholds, f'{final_dest_folder}/doctor_test_thresholds.pt')
    # np.save(f'{final_dest_folder}/doctor_test_thresholds.npy', doctor_test_thresholds.detach().cpu().numpy(), allow_pickle=False)
    torch.save(
        doctor_test_fpr, f'{final_dest_folder}/doctor_test_fpr.pt')
    # np.save(f'{final_dest_folder}/doctor_test_fpr.npy', doctor_test_fpr.detach().cpu().numpy(), allow_pickle=False)
    torch.save(
        doctor_test_tpr, f'{final_dest_folder}/doctor_test_tpr.pt')
    # np.save(f'{final_dest_folder}/doctor_test_tpr.npy', doctor_test_tpr.detach().cpu().numpy(), allow_pickle=False)
    torch.save(
        doctor_test_threshold, f'{final_dest_folder}/doctor_test_threshold.pt')
    # np.save(f'{final_dest_folder}/doctor_test_threshold.npy', doctor_test_threshold.detach().cpu().numpy(), allow_pickle=False)
    torch.save(
        doctor_test_auc, f'{final_dest_folder}/doctor_test_auc.pt')
    # np.save(f'{final_dest_folder}/doctor_test_auc.npy', doctor_test_auc.detach().cpu().numpy(), allow_pickle=False)
