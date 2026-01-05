
import torch
from .base_scikit_postprocessor import BaseScikitPostprocessor

from sklearn.ensemble import RandomForestClassifier


class RandomForestPostprocessor(BaseScikitPostprocessor):
    def __init__(self, model, cfg, result_folder, device=torch.device('cpu')):
        """
        Args:
            classifier (nn.Module): A PyTorch model that takes an input tensor of shape [1, dim] and returns (logits, probs).
            weights (torch.Tensor): Tensor of shape [n_classes] (e.g., [7]).
            means (torch.Tensor): Tensor of shape [n_classes, dim] (e.g., [7, 10]).
            stds (torch.Tensor): Tensor of shape [n_classes, dim] (e.g., [7, 10]).
            n_cluster (int): Number of clusters to partition the error probability into.
            alpha (float): Confidence level parameter for interval widths.
            method (str): The method to compute the cluster. (Currently only "uniform" is supported.)
            seed (int): Random seed for data generation.
            device (torch.device): Device on which to run the classifier.
        """
        super().__init__(model, cfg, result_folder,device)

        self.scikit_kwargs = {
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "min_samples_split": cfg["min_samples_split"],
            "min_samples_leaf": cfg["min_samples_leaf"],
            "max_features": cfg["max_features"],
            "random_state": cfg["random_state"],
            "class_weight": cfg["class_weight"],
            # "criterion": cfg["criterion"],
        }
        self.regressor = RandomForestClassifier(**self.scikit_kwargs)



