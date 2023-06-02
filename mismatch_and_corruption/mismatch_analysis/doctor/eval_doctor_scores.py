import os
import torch
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
                       mismatch_dataset_name,
                       model_seed,
                       logger):
    source_folder = f'./mismatch_analysis/doctor/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}'
    dest_folder = f'{source_folder}/results/r_{r}/seed_{seed}'
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    source_magnitude_folder = '/'.join((source_folder,
                                        f"magnitude_{magnitude}"))
    source_temperature_folder = '/'.join((source_magnitude_folder,
                                          f"temperature_{temperature}"))

    match_ts_doctor_scores = torch.load(
        f"{source_temperature_folder}/match_ts_doctor_scores.pt")
    match_ts_targets = torch.load(
        f"{source_folder}/match_ts_targets.pt")
    match_ts_predictions = torch.load(
        f"{source_folder}/match_ts_predictions.pt")

    mismatch_doctor_scores = torch.load(
        f"{source_temperature_folder}/mismatch_doctor_scores.pt")

    n = len(match_ts_doctor_scores)
    num_val_samples = n // r

    _ = ml_tools.set_seed(seed)
    # perm arrays
    match_ts_perm = torch.randperm(len(match_ts_doctor_scores))
    match_ts_doctor_scores = match_ts_doctor_scores[match_ts_perm]
    match_ts_targets = match_ts_targets[match_ts_perm]
    match_ts_predictions = match_ts_predictions[match_ts_perm]

    mismatch_perm = torch.randperm(len(mismatch_doctor_scores))
    mismatch_doctor_scores = mismatch_doctor_scores[mismatch_perm]

    # Validation
    match_val_doctor_scores = match_ts_doctor_scores[:num_val_samples]
    mismatch_val_doctor_scores = mismatch_doctor_scores[:num_val_samples]
    filter_match_val_correct = match_ts_targets[:
                                                num_val_samples] == match_ts_predictions[:num_val_samples]
    match_val_doctor_scores = match_val_doctor_scores[filter_match_val_correct]
    # balance the number of samples from match and mismatch
    mismatch_val_doctor_scores = mismatch_val_doctor_scores[:num_val_samples][:len(
        match_val_doctor_scores)]

    logger.info(
        f"Doctor baseline with temperature {temperature} and magnitude {magnitude}")
    logger.info(
        f"Number of match val scores: {len(match_val_doctor_scores)}")
    logger.info(
        f"Number of mismatch val scores: {len(mismatch_val_doctor_scores)}")

    doctor_val_scores = torch.cat(
        [match_val_doctor_scores, mismatch_val_doctor_scores])
    # assign labels for binary classification: 0 for in-test, 1 for out-test
    doctor_val_binary_labels = torch.cat(
        [torch.zeros_like(match_val_doctor_scores), torch.ones_like(mismatch_val_doctor_scores)])
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
    torch.save(
        doctor_val_tprs, f'{final_dest_folder}/doctor_val_tprs.pt')
    torch.save(
        doctor_val_thresholds,  f'{final_dest_folder}/doctor_val_thresholds.pt')
    torch.save(
        doctor_val_fpr, f'{final_dest_folder}/doctor_val_fpr.pt')
    torch.save(
        doctor_val_tpr, f'{final_dest_folder}/doctor_val_tpr.pt')
    torch.save(
        doctor_val_threshold, f'{final_dest_folder}/doctor_val_threshold.pt')
    torch.save(
        doctor_val_auc, f'{final_dest_folder}/doctor_val_auc.pt')

    # Test

    match_test_doctor_scores = match_ts_doctor_scores[num_val_samples:]
    mismatch_test_doctor_scores = mismatch_doctor_scores[num_val_samples:]

    filter_match_ts_correct = match_ts_targets[num_val_samples:
                                               ] == match_ts_predictions[num_val_samples:]
    match_test_doctor_scores = match_test_doctor_scores[filter_match_ts_correct]
    mismatch_test_doctor_scores = mismatch_test_doctor_scores[num_val_samples:][:len(
        match_test_doctor_scores)]

    logger.info(
        f"Doctor baseline with temperature {temperature} and magnitude {magnitude}")
    logger.info(
        f"Number of match test scores: {len(match_test_doctor_scores)}")
    logger.info(
        f"Number of mismatch test scores: {len(mismatch_test_doctor_scores)}")

    doctor_test_scores = torch.cat(
        [match_test_doctor_scores, mismatch_test_doctor_scores])
    # assign labels for binary classification: 0 for in-test, 1 for out-test
    doctor_test_labels = torch.cat(
        [torch.zeros_like(match_test_doctor_scores), torch.ones_like(mismatch_test_doctor_scores)])
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
    torch.save(
        doctor_test_tprs, f'{final_dest_folder}/doctor_test_tprs.pt')
    torch.save(
        doctor_test_thresholds, f'{final_dest_folder}/doctor_test_thresholds.pt')
    torch.save(
        doctor_test_fpr, f'{final_dest_folder}/doctor_test_fpr.pt')
    torch.save(
        doctor_test_tpr, f'{final_dest_folder}/doctor_test_tpr.pt')
    torch.save(
        doctor_test_threshold, f'{final_dest_folder}/doctor_test_threshold.pt')
    torch.save(
        doctor_test_auc, f'{final_dest_folder}/doctor_test_auc.pt')
