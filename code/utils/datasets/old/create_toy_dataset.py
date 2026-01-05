import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_toy_dataset(dim=2, n_classes=3,
                         n_train_mlp=10000,
                         n_train_detector=2000,
                         n_concentration=2000,
                         n_test=10000,
                         overlap=1.5, seed=42):
    """
    Generates a synthetic dataset as a mixture of Gaussians and splits it into four datasets:
      - Training set for the MLP classifier (saved as "train.npz")
      - Training set for the error detector (saved as "train_detector_dataset.npz")
      - Dataset for fitting concentration (saved as "concentration_dataset.npz")
      - Test set (saved as "test_dataset.npz")
      
    In addition, this function:
      - Saves a configuration file ("config.json") in the same folder.
      - Computes and saves (in "dataset_stats.csv") the number of samples, the per‐class
        empirical mean, and variance (computed on the generated data) for each dataset.
      - Computes the Bayes classifier predictions (using the true generating parameters)
        on each dataset and saves the results in "bayes_results.csv".
      - If dim==2, also saves a scatter plot of the classifier training set.
    
    All files are saved in:
         data/synthetic/dim-{dim}_classes-{n_classes}
    
    Parameters:
      - dim: Dimension of the input space.
      - n_classes: Number of classes (Gaussians).
      - n_train_mlp: Number of samples for training the MLP classifier.
      - n_train_detector: Number of samples for training the error detector.
      - n_concentration: Number of samples for fitting concentration.
      - n_test: Number of test samples.
      - overlap: Controls the standard deviation of each Gaussian.
      - seed: Random seed for reproducibility.
    """
    np.random.seed(seed)
    # Create directory for saving the dataset.
    data_dir = f"data/synthetic/dim-{dim}_classes-{n_classes}"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save configuration parameters.
    config = {
        "dim": dim,
        "n_classes": n_classes,
        "n_train_mlp": n_train_mlp,
        "n_train_detector": n_train_detector,
        "n_concentration": n_concentration,
        "n_test": n_test,
        "overlap": overlap,
        "seed": seed
    }
    with open(os.path.join(data_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Generate class means.
    means = []
    for i in range(n_classes):
        if dim == 2:
            # Evenly place means on a circle.
            angle = 2 * np.pi * i / n_classes
            radius = 3.0
            mean = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        else:
            # For higher dimensions, sample uniformly from [-3, 3].
            mean = np.random.uniform(-3, 3, size=dim)
        means.append(mean)
    means = np.array(means)
    # Standard deviation for each class.
    std = 1.0 * overlap  # this will be used in generating all samples

    # Helper function: generate samples evenly per class.
    def generate_samples(total_samples):
        X_list = []
        y_list = []
        samples_per_class = total_samples // n_classes
        for class_idx in range(n_classes):
            samples = np.random.randn(samples_per_class, dim) * std + means[class_idx]
            X_list.append(samples)
            y_list.append(np.full(samples_per_class, class_idx))
        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)
        # Shuffle the samples.
        indices = np.arange(len(X_all))
        np.random.shuffle(indices)
        return X_all[indices], y_all[indices]
    
    # Generate each dataset.
    datasets = {
        "train": generate_samples(n_train_mlp),
        "train_detector_dataset": generate_samples(n_train_detector),
        "concentration_dataset": generate_samples(n_concentration),
        "test_dataset": generate_samples(n_test)
    }
    
    # Save datasets.
    for name, (X, y) in datasets.items():
        np.savez(os.path.join(data_dir, f"{name}.npz"), X=X, y=y)
    
    # Compute sample statistics for each dataset and save into a CSV.
    rows = []
    for name, (X, y) in datasets.items():
        for class_idx in range(n_classes):
            mask = (y == class_idx)
            n_samples = np.sum(mask)
            # Store as strings for simplicity.
            row = {
                "dataset": name,
                "class": class_idx,
                "n_samples": int(n_samples),
                "mean": means[class_idx],
                "std": std,
                "seed" : seed
            }
            rows.append(row)
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(data_dir, "dataset_stats.csv"), index=False)
    
    # Define the Bayes posterior probability function.
    # (See bayes_proba documentation below.)
    def bayes_proba(X_input, means_input, std_input):
        """
        For each input x (rows of X_input), returns the probability of each class.
        
        That is, for each sample x, computes:
        
            P(y=i | x) = exp(-||x - mu_i||^2/(2 std_input^2)) / sum_j exp(-||x - mu_j||^2/(2 std_input^2))
        
        Parameters:
            X_input (ndarray): Shape (n_samples, dim)
            means_input (ndarray): Shape (n_classes, dim)
            std_input (float): Standard deviation used in generation.
            
        Returns:
            probs (ndarray): Shape (n_samples, n_classes) with probabilities.
        """
        # Compute squared Euclidean distances.
        dists = np.sum((X_input[:, None, :] - means_input[None, :, :]) ** 2, axis=2)
        logits = -dists / (2 * std_input**2)
        # For numerical stability, subtract max logit per sample.
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs
    
    # For each dataset, compute the Bayes classifier predictions and overall accuracy.
    bayes_rows = []
    for name, (X, y_true) in datasets.items():
        probs = bayes_proba(X, means, std)
        y_pred = np.argmax(probs, axis=1)
        accuracy = np.mean(y_true == y_pred)
        bayes_rows.append({
            "dataset": name,
            "n_samples": X.shape[0],
            "bayes_accuracy": f"{accuracy:.4f}"
        })
    bayes_df = pd.DataFrame(bayes_rows)
    bayes_df.to_csv(os.path.join(data_dir, "bayes_results.csv"), index=False)
    
    # If data is 2D, create a scatter plot of the classifier training set.
    if dim == 2:
        plt.figure(figsize=(8, 6))
        X_train_mlp, y_train_mlp = datasets["train"]
        for class_idx in range(n_classes):
            plt.scatter(X_train_mlp[y_train_mlp == class_idx, 0],
                        X_train_mlp[y_train_mlp == class_idx, 1],
                        label=f"Class {class_idx}", alpha=0.6, edgecolor='k')
        plt.legend()
        plt.title("Toy Dataset for MLP Training")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)
        plt.savefig(os.path.join(data_dir, "dataset_plot.png"))
        plt.close()
    
    print(f"Dataset generated and saved in {data_dir}")
    print(f"Dataset statistics saved to {os.path.join(data_dir, 'dataset_stats.csv')}")
    print(f"Bayes classifier results saved to {os.path.join(data_dir, 'bayes_results.csv')}")
    
    # Also return the means, std, and the bayes_proba function for further use.
    return means, std, bayes_proba

if __name__ == "__main__":
    # Example call: adjust parameters as needed.
    generate_toy_dataset(dim=2, n_classes=3, n_train_mlp=20000,
                         n_train_detector=20000, n_concentration=20000,
                         n_test=20000, overlap=1.5, seed=42)


# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt

# def generate_toy_dataset(dim=2, n_classes=3,
#                          n_train_mlp=10000,
#                          n_train_detector=2000,
#                          n_concentration=2000,
#                          n_test=10000,
#                          overlap=1.5, seed=42):
#     """
#     Generates a synthetic dataset as a mixture of Gaussians and splits it into four datasets:
#       - Training set for the MLP classifier (saved as "train.npz")
#       - Training set for the error detector (saved as "train_detector_dataset.npz")
#       - Dataset for fitting concentration (saved as "concentration_dataset.npz")
#       - Test set (saved as "test_dataset.npz")
      
#     All files, along with a configuration file and (if dim==2) a scatter plot,
#     are saved in the folder:
#         synthetic_data/dim-{dim}_classes-{n_classes}/
    
#     Parameters:
#       - dim: Dimension of the input space.
#       - n_classes: Number of classes (Gaussians).
#       - n_train_mlp: Number of samples for training the MLP classifier.
#       - n_train_detector: Number of samples for training the error detector.
#       - n_concentration: Number of samples for fitting concentration.
#       - n_test: Number of test samples.
#       - overlap: Controls the standard deviation of each Gaussian.
#       - seed: Random seed for reproducibility.
#     """
#     np.random.seed(seed)
#     # Create directory for saving the dataset.
#     data_dir = f"synthetic_data/dim-{dim}_classes-{n_classes}"
#     os.makedirs(data_dir, exist_ok=True)
    
#     # Save configuration parameters.
#     config = {
#         "dim": dim,
#         "n_classes": n_classes,
#         "n_train_mlp": n_train_mlp,
#         "n_train_detector": n_train_detector,
#         "n_concentration": n_concentration,
#         "n_test": n_test,
#         "overlap": overlap,
#         "seed": seed
#     }
#     with open(os.path.join(data_dir, "config.json"), "w") as f:
#         json.dump(config, f, indent=4)
    
#     # Generate class means.
#     means = []
#     for i in range(n_classes):
#         if dim == 2:
#             # Evenly place means on a circle.
#             angle = 2 * np.pi * i / n_classes
#             radius = 3.0
#             mean = np.array([radius * np.cos(angle), radius * np.sin(angle)])
#         else:
#             # For higher dimensions, sample uniformly from [-3, 3].
#             mean = np.random.uniform(-3, 3, size=dim)
#         means.append(mean)
#     means = np.array(means)
    
#     std = 1.0 * overlap  # Standard deviation controlling the overlap.
    
#     # Helper: generate samples evenly per class.
#     def generate_samples(total_samples):
#         X_list = []
#         y_list = []
#         samples_per_class = total_samples // n_classes
#         for class_idx in range(n_classes):
#             samples = np.random.randn(samples_per_class, dim) * std + means[class_idx]
#             X_list.append(samples)
#             y_list.append(np.full(samples_per_class, class_idx))
#         X_all = np.vstack(X_list)
#         y_all = np.concatenate(y_list)
#         # Shuffle the samples.
#         indices = np.arange(len(X_all))
#         np.random.shuffle(indices)
#         return X_all[indices], y_all[indices]
    
#     # Generate each dataset.
#     X_train_mlp, y_train_mlp = generate_samples(n_train_mlp)
#     X_train_detector, y_train_detector = generate_samples(n_train_detector)
#     X_concentration, y_concentration = generate_samples(n_concentration)
#     X_test, y_test = generate_samples(n_test)
    
#     # Save datasets.
#     np.savez(os.path.join(data_dir, "train.npz"), X=X_train_mlp, y=y_train_mlp)
#     np.savez(os.path.join(data_dir, "train_detector_dataset.npz"), X=X_train_detector, y=y_train_detector)
#     np.savez(os.path.join(data_dir, "concentration_dataset.npz"), X=X_concentration, y=y_concentration)
#     np.savez(os.path.join(data_dir, "test_dataset.npz"), X=X_test, y=y_test)
    
#     # If 2D, create and save a scatter plot of the classifier training set.
#     if dim == 2:
#         plt.figure(figsize=(8, 6))
#         for class_idx in range(n_classes):
#             plt.scatter(X_train_mlp[y_train_mlp == class_idx, 0],
#                         X_train_mlp[y_train_mlp == class_idx, 1],
#                         label=f"Class {class_idx}", alpha=0.6, edgecolor='k')
#         plt.legend()
#         plt.title("Toy Dataset for MLP Training")
#         plt.xlabel("Feature 1")
#         plt.ylabel("Feature 2")
#         plt.grid(True)
#         plt.savefig(os.path.join(data_dir, "dataset_plot.png"))
#         plt.close()
    
#     print(f"Dataset generated and saved in {data_dir}")

# if __name__ == "__main__":
#     # Example call: adjust parameters as needed.
#     generate_toy_dataset(dim=2, n_classes=3, n_train_mlp=10000,
#                          n_train_detector=10000, n_concentration=10000,
#                          n_test=10000, overlap=1.5, seed=42)
