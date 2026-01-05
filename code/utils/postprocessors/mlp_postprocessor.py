import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from code.utils.metrics import compute_all_metrics
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import random

def set_seed(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    # For CUDA matmul determinism (PyTorch docs)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enforce deterministic ops where possible
    torch.use_deterministic_algorithms(True, warn_only=True)

def read_probits(latent_path, n_samples=None, order=True, subclasses=None, temperature=2.0, space="probits"):
    pkg = torch.load(latent_path, map_location="cpu")
    all_logits = pkg["logits"].to(torch.float32)        # (N, C)
    all_labels = pkg["labels"]              # (N,)
    all_model_preds  = pkg["model_preds"]# (N,)
    all_detector_labels = (all_model_preds != all_labels).int()
    
    if subclasses is not None:
        sub_idx = [i for i, label in enumerate(all_labels)
             if label in subclasses]
        sub_idx = torch.tensor(sub_idx, dtype=torch.long)
        all_logits = all_logits.index_select(0, sub_idx)
        all_logits = all_logits[:, subclasses]
        all_detector_labels = all_detector_labels.index_select(0, sub_idx)
    
    if space == "logits":
        all_logits = all_logits / (all_logits ** 2).sum(dim=1, keepdim=True).sqrt()
    elif space == "probits":
        all_logits = torch.softmax(all_logits / temperature, dim=1)
    else:
        raise ValueError(f"Unknown space: {space}")

    probits = all_logits
    if order:
        probits = probits.sort(dim=1, descending=True)[0]

    if n_samples is None:
        return probits, all_detector_labels
   
    return probits[:n_samples], all_detector_labels[:n_samples]

DIC_DATASETS_SPLITS = {
    "cifar10": (2000, 3000, 5000),
    "cifar100":  (2000, 3000, 5000),
    "imagenet": (10000, 15000, 25000),
}

def get_dataset(data_name, model_name, seed_split=9, order=True, temperature=2.0, n_val=2500, batch_size=252, device="cpu", generator=None, space="probits"):
    

    size_splits = DIC_DATASETS_SPLITS[data_name]
   
    latent_path = f"./latent/{data_name}_{model_name}/seed-split-{seed_split}/res_n-samples-{size_splits[0]}_transform-test_n-epochs-1.pt"
    probits_res, errors_res = read_probits(latent_path, order=order, temperature=temperature, space=space)
    latent_path_cal = f"./latent/{data_name}_{model_name}/seed-split-{seed_split}/cal_n-samples-{size_splits[1]}_transform-test_n-epochs-1.pt"
    cal_probits, cal_errors = read_probits(latent_path_cal, order=order, temperature=temperature, space=space)
    latent_path_test = f"./latent/{data_name}_{model_name}/seed-split-{seed_split}/test_n-samples-{size_splits[2]}.pt"
    test_probits, test_errors = read_probits(latent_path_test, order=order, temperature=temperature, space=space)

    data_train = torch.cat([probits_res, cal_probits], dim=0)
    errors_train = torch.cat([errors_res, cal_errors], dim=0)
    data_val = test_probits[:n_val]
    errors_val = test_errors[:n_val]
    data_test = test_probits[n_val:]
    errors_test = test_errors[n_val:]

    train_ds = TensorDataset(data_train, errors_train)
    val_ds = TensorDataset(data_val, errors_val)
    test_ds = TensorDataset(data_test, errors_test)
    train_dataloader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10, generator=generator
        )
    val_dataloader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10, generator=generator
        )
    test_dataloader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10, generator=generator
        )
    return train_dataloader, val_dataloader, test_dataloader

# class MLPPostprocessor(nn.Module):
#     def __init__(
#             self, 
#             in_dim=10,
#             hid_dim=32,
#             dropout_p=0.0,
#             temperature=1.0, 
#             class_proportions=None,
#             order=True, 
#             space="probits",
#               device="cpu"):
#         super().__init__()
#         self.temperature = temperature
        
#         self.device = device
#         self.params = None
#         self.in_dim = in_dim
#         self.hid_dim = hid_dim
#         self.class_proportions = class_proportions
#         self.order = order
#         self.space = space
#         self.dropout_p = dropout_p
#         self.fc1 = nn.Linear(self.in_dim, self.hid_dim)
#         self.activ1 = nn.ReLU()
#         self.dropout1 = torch.nn.Dropout(dropout_p)
#         self.fc2 = nn.Linear(self.hid_dim, 1)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.dropout1(x)
#         logits = self.fc2(x)
#         return logits.squeeze(-1)

class MLPPostprocessor(nn.Module):
    def __init__(
            self, 
            in_dim=10,
            hid_dim=32,
            num_hidden_layers=0,
            dropout_p=0.0,
            temperature=1.0, 
            class_proportions=None,
            order=True, 
            space="probits",
              device="cpu"
              ):
        super().__init__()
        self.temperature = temperature
        
        self.device = device
        self.params = None
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.class_proportions = class_proportions
        self.order = order
        self.space = space
        self.dropout_p = dropout_p
        self.fc1 = nn.Linear(self.in_dim, self.hid_dim)
        self.activ1 = nn.ReLU()
        self.dropout1 = torch.nn.Dropout(dropout_p)
        self.classifier = nn.Linear(self.hid_dim, 1)
        self.num_hidden_layers = num_hidden_layers
        if self.num_hidden_layers > 0:
            self.hidden_layers = torch.nn.Sequential(
                *(
                    [
                        torch.nn.Linear(hid_dim, hid_dim),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(dropout_p),
                    ]
                    * num_hidden_layers
                ),
            )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        if self.num_hidden_layers > 0:
            x = self.hidden_layers(x)
        logits = self.classifier(x)
        return logits.squeeze(-1)


    

def evaluate(model, dataloader, device="cpu"):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
            all_labels.append(y_batch.cpu())
    all_preds = torch.cat(all_preds, dim=0).squeeze().numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
   
    fpr, tpr, thr, auroc, accuracy, aurc_value, aupr_in, aupr_out = compute_all_metrics(
            conf=all_preds,
            detector_labels=all_labels,
        )
 
    results = {}
    results["fpr"] = fpr
    results["tpr"] = tpr
    results["thr"] = thr
    results["roc_auc"] = auroc
    results["model_acc"] = accuracy
    results["aurc"] = aurc_value
    results["aupr_err"] = aupr_in
    results["aupr_success"] = aupr_out
    
    
    return pd.DataFrame([results])

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

   
if __name__ == "__main__":
    seed = 41
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "timm-vit-tiny16" # timm-vit-tiny16
    data_name = "imagenet"
    space = "logits"  # "logits" or "probits"
    seed_split = 9
    order = True
    batch_size = 512
    hid_dim = 2048
    dropout_p = 0.0
    lr = 1e-3
    lr_min=1e-5
    n_epochs = 200
    weight_decay = 0
    temperature = 1.0
    num_hidden_layers = 3
    balanced_logits = False
    results_list = {"train": [], "val": [], "test": []}

    save_folder  = f"./code/utils/postprocessors/mlp_results"
    file = (f"{data_name}_{model_name}_seed-split-{seed_split}_order-{order}_hid-{hid_dim}_num-hidden-{num_hidden_layers}"
            f"_dropout-{dropout_p}_lr-{lr}_epochs-{n_epochs}_wd-{weight_decay}_T-{temperature}_space-{space}_cosine-sched")
    if balanced_logits:
        file += "_balanced-logits"
    if lr_min != 0:
        file += f"_lr-min-{lr_min}"
    save_folder = os.path.join(save_folder, file, f"seed-{seed}")
    os.makedirs(save_folder, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataset(
        data_name=data_name,
        model_name=model_name,
        seed_split=seed_split,  
        order=order,
        temperature=temperature,
        n_val=12500,
        batch_size=batch_size,
        device=device,
        generator=g,
        space=space,
    )
    print("Train loader size:", len(train_loader.dataset))
    n_train_errors = torch.sum(torch.tensor(
        [label for _, label in train_loader.dataset if label == 1], dtype=torch.float32))
    
    n_train_no_errors = len(train_loader.dataset) - n_train_errors
    error_ratio = n_train_errors / n_train_no_errors
    print("Train error ratio:", error_ratio)
    print("Number of train errors:", n_train_errors.item())
    print("Number of train no-errors:", n_train_no_errors)


    # ----- Training setup -----
    model = MLPPostprocessor(in_dim=1000, hid_dim=hid_dim, num_hidden_layers=num_hidden_layers,
                             dropout_p=dropout_p, temperature=2.0).to(device)
    criterion = nn.BCEWithLogitsLoss()  # logits + float targets
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,   # number of epochs for one cosine cycle
        eta_min=lr_min       # final LR at the end of training
    )

    # ----- Model selection based on validation FPR -----
    best_val_fpr = float("inf")
    best_val_roc_auc = 0.0
    best_roc_auc_epoch = -1
    best_fpr_epoch = -1
    best_ckpt_path = os.path.join(save_folder, f"{file}_best_val_roc_auc_model.pt")
    best_fpr_ckpt_path = os.path.join(save_folder, f"{file}_best_fpr_model.pt")

    # ----- Training loop -----
    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).float()

            logits = model(x_batch)              # (B,)
            if balanced_logits:
                logits += torch.log(error_ratio)
            loss = criterion(logits, y_batch)    # scalar

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            total += y_batch.size(0)

        epoch_train_loss = running_loss / total

        # ----- Validation loss -----
        model.eval()
        val_running_loss = 0.0
        val_total = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device).float()
                logits_val = model(x_val)
                val_loss = criterion(logits_val, y_val)
                val_running_loss += val_loss.item() * x_val.size(0)
                val_total += y_val.size(0)
        epoch_val_loss = val_running_loss / val_total

        # ----- Training loss -----
        model.eval()
        train_running_loss = 0.0
        train_total = 0
        with torch.no_grad():
            for x_tr, y_tr in train_loader:
                x_tr = x_tr.to(device)
                y_tr = y_tr.to(device).float()
                logits_tr = model(x_tr)
                train_loss = criterion(logits_tr, y_tr)
                train_running_loss += train_loss.item() * x_tr.size(0)
                train_total += y_tr.size(0)
        epoch_train_loss = train_running_loss / train_total

        # ----- Metrics (AUROC, FPR, etc.) -----
        with torch.no_grad():
            results_train = evaluate(model, train_loader, device=device)
            results_val = evaluate(model, val_loader, device=device)
            results_test = evaluate(model, test_loader, device=device)

            # add losses to the metrics DataFrames
            results_train.loc[0, "loss"] = epoch_train_loss
            results_val.loc[0, "loss"] = epoch_val_loss
          

            results_list["train"].append(results_train)
            results_list["val"].append(results_val) 
            results_list["test"].append(results_test)

        train_auc = results_train["roc_auc"].iloc[0]
        val_auc = results_val["roc_auc"].iloc[0]

                # ----- Model selection: pick model with lowest validation FPR -----
        val_fpr = results_val["fpr"].iloc[0]
        if val_fpr < best_val_fpr:
            best_val_fpr = val_fpr
            best_fpr_epoch = epoch
            torch.save(model.state_dict(), best_fpr_ckpt_path)
        if val_auc > best_val_roc_auc:
            best_val_roc_auc = val_auc
            best_roc_auc_epoch = epoch
            torch.save(model.state_dict(), best_ckpt_path)
        

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={epoch_train_loss:.4f} | val_loss={epoch_val_loss:.4f} | "
            f"train_roc_auc={train_auc:.4f} | val_roc_auc={val_auc:.4f} | "
            f"best_val_fpr={best_val_fpr:.4f} (epoch {best_fpr_epoch}) | "
            f"best_val_roc_auc={best_val_roc_auc:.4f} (epoch {best_roc_auc_epoch})"
        )
        scheduler.step()


    # ----- Concatenate per-epoch results -----
    for split in ["train", "val", "test"]:
            results_list[split] = pd.concat(results_list[split], ignore_index=True)
            results_list[split].to_csv(f"{save_folder}/{file}_{split}_results.csv", index=False)

    # ----- Plots: ROC_AUC, FPR, LOSS (train vs val) -----
    metrics_to_plot = ["roc_auc", "fpr", "loss"]

    epochs = range(1, n_epochs + 1)
    for metric in metrics_to_plot:
        plt.figure()
        plt.plot(epochs, results_list["train"][metric], label="Train " + metric)
        plt.plot(epochs, results_list["val"][metric], label="Validation " + metric)
        if metric != "loss":
            plt.plot(epochs, results_list["test"][metric], label="Test " + metric)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_folder}/{file}_{metric}_curve.png")
        plt.close()

     # ----- Final evaluation on TEST set with best model (selected by val FPR) -----
    print(f"\nLoading best FPR model from epoch {best_fpr_epoch} "
          f"with validation FPR = {best_val_fpr:.4f}")
    model.load_state_dict(torch.load(best_fpr_ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        test_results = evaluate(model, test_loader, device=device)

    # extract scalar metrics
    test_roc_auc = test_results["roc_auc"].iloc[0]
    test_fpr = test_results["fpr"].iloc[0]

    print(f"Best-model Test ROC AUC = {test_roc_auc:.4f} | "
          f"Best-model Test FPR = {test_fpr:.4f}")

    # save full test results and also just FPR + ROC AUC
    test_results.to_csv(f"{save_folder}/{file}_test_results_best_fpr.csv", index=False)
    # test_results[["fpr", "roc_auc"]].to_csv(
    #     f"{save_folder}/{file}_test_fpr_rocauc.csv", index=False
    # )

    # ----- Final evaluation on TEST set with best model (selected by val ROC AUC) -----
    print(f"\nLoading best ROC AUC model from epoch {best_roc_auc_epoch} "
          f"with validation ROC AUC = {best_val_roc_auc:.4f}")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.to(device)
    model.eval()    
    with torch.no_grad():
        test_results = evaluate(model, test_loader, device=device)
    # extract scalar metrics
    test_roc_auc = test_results["roc_auc"].iloc[0]
    test_fpr = test_results["fpr"].iloc[0]
    print(f"Best-model Test ROC AUC = {test_roc_auc:.4f} | "
          f"Best-model Test FPR = {test_fpr:.4f}")
    # save full test results and also just FPR + ROC AUC
    test_results.to_csv(f"{save_folder}/{file}_test_results_best_roc_auc.csv", index=False)
    # test_results[["fpr", "roc_auc"]].to_csv(
    #     f"{save_folder}/{file}_test_fpr_rocauc.csv", index=False
    # )