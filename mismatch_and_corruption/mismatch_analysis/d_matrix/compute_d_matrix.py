import os
import torch
import numpy as np
from tools import ml_tools
from tools.d_matrix_tools import matrix_D


def compute_d_matrix(seed,
                     dest_folder,
                     match_ts_labels,
                     match_ts_predictions,
                     match_ts_logits,
                     use_mismatch_val,
                     mismatch_labels,
                     mismatch_logits,
                     r,
                     lbd,
                     logger,
                     device,
                     batch_size,
                     lr,
                     epochs):
    torch_gen = ml_tools.set_seed(seed)
    dest_folder_seed = os.path.join(dest_folder, f"seed_{seed}/r_{r}/lr_{lr}/epochs_{epochs}/lbd_{lbd}/use_mismatch_val_{use_mismatch_val}")
    if not os.path.exists(dest_folder_seed):
        os.makedirs(dest_folder_seed)

    n = len(match_ts_labels)
    num_val_samples = n // r

    match_ts_perm = torch.randperm(len(match_ts_predictions))
    print("match_ts_perm", match_ts_perm)
    val_logits = match_ts_logits[match_ts_perm][:num_val_samples]
    val_labels = match_ts_labels[match_ts_perm][:num_val_samples]
    val_predictions = match_ts_predictions[match_ts_perm][:num_val_samples]

    # we want the misclassifications to be 1, i.e. true
    labs = val_labels != val_predictions

    if use_mismatch_val:
        mismatch_val_perm = torch.randperm(len(mismatch_labels))
        mismatch_val_logits = mismatch_logits[mismatch_val_perm][:num_val_samples]
        mismatch_val_labels = mismatch_labels[mismatch_val_perm][:num_val_samples]

        val_logits = torch.cat(
            (val_logits, mismatch_val_logits), dim=0)
        labs = torch.cat(
            (labs, torch.ones(len(mismatch_val_labels), dtype=torch.bool)), dim=0)

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.softmax(val_logits, dim=1), labs), batch_size=batch_size, shuffle=False, num_workers=2, generator=torch_gen)

    params, _= matrix_D(
        train_data=torch.softmax(val_logits, dim=1), train_labels=labs, device=device, lbd=lbd, logger=logger)

    # _, _, loss_history, loss_history_pos, loss_history_neg,  auc_history, fpr_at_95_tpr_history = train(
    #     lbd, epochs, device, optimizer, val_loader, model_op, params, test_labels=labs, test_data=torch.softmax(val_logits, dim=1), logger=logger)

    # save params
    # torch.save(
    #     params, '/'.join((dest_folder_seed, 'D_matrix.pt')))
    np.save(os.path.join(dest_folder_seed, "D_matrix.npy"), params.detach().cpu().numpy(), allow_pickle=False)