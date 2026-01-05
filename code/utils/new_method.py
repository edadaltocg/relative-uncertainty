import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from code.utils.results_helper import setup_publication_style

setup_publication_style()

latent_path = "./latent/ablation/imagenet_timm-vit-tiny16_n_cal/seed-split-9/cal1_n-samples-25000_transform-test_n-epochs-1.pt"
out_dir = "./new_method"
def compute_means(all_logits, all_model_preds, out_dir,
                  filename="mean_max_logit_per_pred_class.png"):
    # # Max logit per sample
    # max_logits = all_logits.max(dim=1).values        # shape: (N,)

    # # Ensure integer class indices
    # preds = all_model_preds.to(torch.long)           # shape: (N,)
    # C = all_logits.size(1)

    # # Sum and count per predicted class
    # sum_per_pred = torch.bincount(preds, weights=max_logits, minlength=C)   # (C,)
    # cnt_per_pred = torch.bincount(preds, minlength=C)                       # (C,)

    # # Mean max-logit per predicted class (NaN for classes never predicted)
    # mean_per_pred = torch.where(
    #     cnt_per_pred > 0,
    #     sum_per_pred / cnt_per_pred,
    #     torch.nan
    # )
    # valid = cnt_per_pred > 0
    # order = torch.argsort(mean_per_pred[valid], dim=0)  # ascending
    # sorted_classes = torch.arange(C, device=mean_per_pred.device)[valid][order]
    # sorted_means   = mean_per_pred[valid][order]



    #  # --- plotting ---
    # Path(out_dir).mkdir(parents=True, exist_ok=True)
    # save_path = os.path.join(out_dir, filename)

    # plt.figure(figsize=(10, 4))
    # x = torch.arange(sorted_classes.numel()).cpu()
    # y = sorted_means.cpu()

    # plt.bar(x.numpy(), y.numpy())
    # plt.xlabel("Class index (sorted by mean max-logit)")
    # plt.ylabel("Mean max-logit")
    # plt.title("Mean max-logit per predicted class (ascending)")
    # # Optional: show the original class ids on x-axis at sparse ticks
    # if sorted_classes.numel() <= 50:
    #     plt.xticks(x.numpy(), sorted_classes.cpu().numpy(), rotation=90)
    # else:
    #     # sparse ticks for readability
    #     step = max(1, sorted_classes.numel() // 20)
    #     idx = x[::step]
    #     plt.xticks(idx.numpy(), sorted_classes.cpu().numpy()[::step], rotation=90)

    # plt.tight_layout()
    # plt.savefig(save_path, dpi=200)
    # plt.close()

    n_pre_clusters = 10
    N, C = all_logits.shape
    n_bins = int(n_pre_clusters)
    if n_bins <= 0:
        raise ValueError("self.n_pre_clusters must be a positive integer.")

    # ---- (1) classwise mean of per-sample max-logit ----
    max_per_sample = all_logits.max(dim=1).values                 # (N,)
    preds = all_model_preds.to(torch.long)                         # (N,)
    sum_per_pred = torch.bincount(preds, weights=max_per_sample, minlength=C)  # (C,)
    cnt_per_pred = torch.bincount(preds, minlength=C)                            # (C,)

    # Assumption: every class is predicted -> cnt_per_pred > 0 for all k
    mean_per_pred = sum_per_pred / cnt_per_pred               # (C,)

    # ---- (2) sort classes by increasing mean ----
    order = torch.argsort(mean_per_pred)                      # (C,)
    print("order:", order)
    sorted_means = mean_per_pred[order]                       # (C,)
    sorted_classes = order                                    # (C,)

    lo = sorted_means[0].item()
    hi = sorted_means[-1].item()
    if not (float("-inf") < lo < float("inf") and float("-inf") < hi < float("inf")):
        raise ValueError("Non-finite classwise means encountered.")
    if hi == lo:
        raise ValueError("Degenerate range: max(sorted_means) == min(sorted_means).")

    # ---- (3) equal-width bins from sorted_means only ----
    bin_edges = torch.linspace(lo, hi, steps=n_bins + 1, device=all_logits.device)  # (n_bins+1,)
    # Assign EACH CLASS to a bin using its classwise mean
    # Values in [edge[i], edge[i+1]) -> bin i, last edge closed by clamp
    cut_points = bin_edges[1:-1]
    class_bin_sorted = torch.bucketize(sorted_means, boundaries=cut_points, right=False)  # (C,)
    class_bin_sorted = class_bin_sorted.clamp(0, n_bins - 1)                               # (C,)

    # Map back to original class ids: class_bin[k] is bin of class k
    class_bin = torch.empty(C, dtype=torch.long, device=all_logits.device)
    class_bin[sorted_classes] = class_bin_sorted
    print("class_bin:", class_bin)


    return sorted_means
if __name__ == "__main__":
    if os.path.exists(latent_path):
        pkg = torch.load(latent_path, map_location="cpu")
    else:
        raise ValueError(f"File {latent_path} does not exist.")
    all_logits = pkg["logits"].to(torch.float32)        # (N, C)
    all_labels = pkg["labels"]              # (N,)
    all_model_preds  = pkg["model_preds"]# (N,)
    all_detector_labels = (all_model_preds != all_labels).float()

    print("Logits shape:", all_logits.shape)
    print("Labels shape:", all_labels.shape)
    print("Model preds shape:", all_model_preds.shape)

    all_logits = torch.softmax(all_logits , dim=1)
    
    compute_means(all_logits, all_model_preds, out_dir)

    
