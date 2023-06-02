import os
import torch
from tools.data_tools import *
from sklearn.metrics import roc_curve, auc
from tools.d_matrix_tools import D_scores_func


def compute_eval_scores(device,
                        logger,
                        magnitude,
                        magnitude_folder,
                        new_match_val_logits,
                        new_match_val_targets,
                        new_match_val_predictions,
                        params,
                        seed,
                        temperature,
                        new_corrupted_logits,
                        new_corrupted_labels,
                        new_corrupted_predictions):
    logger.info(
        f"Seed: {seed}, magnitude: {magnitude}, temperature: {temperature}")
    temperature_folder = os.path.join(
        magnitude_folder, f"temperature_{temperature}")
    if not os.path.exists(temperature_folder):
        os.makedirs(temperature_folder)

    # Validation only on match samples val split
    new_match_val_sp = torch.softmax(
        new_match_val_logits/temperature, dim=1)
    new_match_val_D_scores = D_scores_func(
        test_data=new_match_val_sp, device=device, params=params)
    D_match_scores = torch.tensor(new_match_val_D_scores)

    # print for this temperature and magnitude the number of in-test and out-test scores
    logger.info(
        f"D matrix baseline with temperature {temperature} and magnitude {magnitude}")
    logger.info(
        f"Number of match val scores: {len(D_match_scores)}")

    D_scores = D_match_scores.detach().cpu().numpy()

    D_labels = new_match_val_targets != new_match_val_predictions

    D_fprs, D_tprs, thresholds = roc_curve(
        D_labels, D_scores)
    D_fpr, tpr, threshold = fpr_at_fixed_tpr(
        D_fprs, D_tprs, thresholds, tpr_level=0.95)
    D_auc = auc(D_fprs, D_tprs)
    print(
        f"D baseline with temperature {temperature} and magnitude {magnitude}")
    print(f"AUC val: {D_auc}")
    print(f"FPR val: {D_fpr}")

    # save results
    torch.save(D_fprs, '/'.join(
        [temperature_folder, f'D_fprs_val.pt']))
    torch.save(D_tprs, '/'.join(
        [temperature_folder, f'D_tprs_val.pt']))
    torch.save(thresholds, '/'.join(
        [temperature_folder, f'D_thresholds_val.pt']))
    torch.save(D_fpr, '/'.join(
        [temperature_folder, f'D_fpr_val.pt']))
    torch.save(tpr, '/'.join(
        [temperature_folder, f'D_tpr_val.pt']))
    torch.save(threshold, '/'.join(
        [temperature_folder, f'D_threshold_val.pt']))
    torch.save(D_auc, '/'.join(
        [temperature_folder, f'D_auc_val.pt']))

    # test only on corrupted samples

    logger.info(
        f"Seed: {seed}, magnitude: {magnitude}, temperature: {temperature}")
    temperature_folder = os.path.join(
        magnitude_folder, f"temperature_{temperature}")
    if not os.path.exists(temperature_folder):
        os.makedirs(temperature_folder)
    new_corrupted_test_sp = torch.softmax(
        new_corrupted_logits/temperature, dim=1)

    new_corrupted_test_D_scores = D_scores_func(
        test_data=new_corrupted_test_sp, device=device, params=params)
    
    logger.info(
        f"D matrix baseline with temperature {temperature} and magnitude {magnitude}")
    logger.info(
        f"Number of corrupted test scores: {len(new_corrupted_test_D_scores)}")
   
    D_scores = new_corrupted_test_D_scores.detach().cpu().numpy()
    D_labels = new_corrupted_labels != new_corrupted_predictions

    D_fprs, D_tprs, thresholds = roc_curve(
        D_labels, D_scores)
    D_fpr, tpr, threshold = fpr_at_fixed_tpr(
        D_fprs, D_tprs, thresholds, tpr_level=0.95)
    D_auc = auc(D_fprs, D_tprs)
    print(
        f"D baseline with temperature {temperature} and magnitude {magnitude}")
    print(f"AUC test: {D_auc}")
    print(f"FPR test: {D_fpr}")

    # save results
    torch.save(D_fprs, '/'.join(
        [temperature_folder, f'D_fprs_test.pt']))
    torch.save(D_tprs, '/'.join(
        [temperature_folder, f'D_tprs_test.pt']))
    torch.save(thresholds, '/'.join(
        [temperature_folder, f'D_thresholds_test.pt']))
    torch.save(D_fpr, '/'.join(
        [temperature_folder, f'D_fpr_test.pt']))
    torch.save(tpr, '/'.join(
        [temperature_folder, f'D_tpr_test.pt']))
    torch.save(threshold, '/'.join(
        [temperature_folder, f'D_threshold_test.pt']))
    torch.save(D_auc, '/'.join(
        [temperature_folder, f'D_auc_test.pt']))
