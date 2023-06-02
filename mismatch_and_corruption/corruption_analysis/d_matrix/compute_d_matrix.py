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
                     r,
                     lbd,
                     logger,
                     device,
                     batch_size,
                     lr,
                     epochs):
    dest_folder_seed = os.path.join(
        dest_folder, f"seed_{seed}/r_{r}/lr_{lr}/epochs_{epochs}/lbd_{lbd}")
    if not os.path.exists(dest_folder_seed):
        os.makedirs(dest_folder_seed)

    n = len(match_ts_labels)
    num_val_samples = n // r

    torch_gen = ml_tools.set_seed(seed)

    match_ts_perm = torch.randperm(len(match_ts_predictions))
    val_logits = match_ts_logits[match_ts_perm][:num_val_samples]
    val_labels = match_ts_labels[match_ts_perm][:num_val_samples]
    val_predictions = match_ts_predictions[match_ts_perm][:num_val_samples]

    # we want the misclassifications to be 1, i.e. true
    labs = val_labels != val_predictions

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.softmax(val_logits, dim=1), labs),
        batch_size=batch_size, shuffle=False, num_workers=2, generator=torch_gen)

    params, _= matrix_D(
        train_data=torch.softmax(val_logits, dim=1), train_labels=labs, device=device, lbd=lbd, logger=logger)


    # save params
    # torch.save(
    #     params, '/'.join((dest_folder_seed, 'D_matrix.pt')))
    np.save('/'.join((dest_folder_seed, 'D_matrix.npy')), params.detach().cpu().numpy(), allow_pickle=False)