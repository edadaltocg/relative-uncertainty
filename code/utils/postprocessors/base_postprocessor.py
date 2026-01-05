
import torch
from code.utils.calibration import platt_logits

class BasePostprocessor:
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
        self.model = model
        self.cfg = cfg
        self.device = device
        self.result_folder = result_folder
        self.temperature = 1.0

    def calibrate(self, calib_loader=None, logits=None, targets=None):
        if calib_loader is None:
            if logits is None or targets is None:
                raise ValueError("Either calib_loader or (logits and targets) must be provided")
            dataset = torch.utils.data.TensorDataset(logits, targets)
            calib_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        self.temperature = platt_logits(calib_loader, device=self.device).to(self.device)


    def __call__(self, inputs=None, logits=None):
        if logits is None:
            if inputs is None:
                raise ValueError("Either logits or inputs must be provided")
            logits = self.model(inputs)

        # return -torch.softmax(logits / self.temperature, dim=1).max(dim=1, keepdim=True)[0]  # [batch_size]
        return - torch.softmax(logits / self.temperature, dim=1).max(dim=1)[0]  # [batch_size]