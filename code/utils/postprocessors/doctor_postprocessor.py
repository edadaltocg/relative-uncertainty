
import torch
from .base_postprocessor import BasePostprocessor

def gini(logits, temperature=1.0, normalize=False):
    g =torch.sum(torch.softmax(logits / temperature, dim=1) ** 2, dim=1, keepdim=True)
    if normalize:
        return  (1 - g) / g 
    else:
        return 1 - g




class DoctorPostprocessor(BasePostprocessor):
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
        super().__init__(model, cfg, result_folder, device)

        self.temperature = cfg["temperature"]
        self.magnitude = cfg["magnitude"]
        self.normalize = cfg["normalize"]
    

    def __call__(self, inputs=None, logits=None):
        if logits is None:
            if inputs is None:
                raise ValueError("Either logits or inputs must be provided")
            logits = self.model(inputs)

        # return -torch.softmax(logits / self.temperature, dim=1).max(dim=1, keepdim=True)[0]  # [batch_size]
        return gini(logits, temperature=self.temperature, normalize=self.normalize).squeeze()  # [batch_size]