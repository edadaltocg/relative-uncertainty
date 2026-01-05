import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical


# -------------------------------

class ThresholdClassifier(nn.Module):
    def __init__(self, threshold=0.5):
        """
        A simple threshold classifier.
        
        Args:
            threshold (float): Threshold for classification.
        """
        super(ThresholdClassifier, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, dim].
        
        Returns:
            preds (torch.Tensor): Predicted class indices of shape [batch_size].
        """
        x = x.flatten()
        preds = torch.zeros((x.shape[0], 2)).to(x.device)
        preds[x > self.threshold, 1] = 1
        preds[x <= self.threshold, 0] = 1
        return preds, preds


class BayesClassifier(nn.Module):
    def __init__(self, means, covs, weights):
        """
        A Bayes classifier implemented as an nn.Module.
        
        Args:
            means (torch.Tensor): Tensor of shape [n_classes, dim] for the component means.
            covs (torch.Tensor): Tensor of shape [n_classes, dim, dim] for the component covariance matrices.
            weights (torch.Tensor): Tensor of shape [n_classes] for the mixture weights.
        """
        super(BayesClassifier, self).__init__()
        # Register buffers so they become part of the model's state but are not trainable.
        # means: [n_classes, dim]
        self.register_buffer('means', means)
        # covs: [n_classes, dim, dim]
        self.register_buffer('covs', covs)
        # weights: [n_classes]
        self.register_buffer('weights', weights)
        self.register_buffer('log_weights', torch.log(weights))
        self.n_classes, self.dim = means.shape

    def forward(self, x):
        """
        Computes the posterior probabilities and predicted classes.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, dim].
                              e.g., x: [batch_size, dim]
        
        Returns:
            preds (torch.Tensor): Predicted class indices of shape [batch_size].
                                  e.g., preds: [batch_size]
            posterior (torch.Tensor): Posterior probabilities for each class [batch_size, n_classes].
                                      e.g., posterior: [batch_size, n_classes]
        """
        batch_size = x.shape[0]  # batch_size
        x = x.view(batch_size, -1)  
        
        # Collect weighted probability densities for each class.
        logits = []
        for i in range(self.n_classes):
            # Create a multivariate normal distribution for class i using mean and covariance.
            # For each class i: mean: [dim], covariance matrix: [dim, dim]
            # eigs = torch.linalg.eigvalsh(self.covs[i])
            # print("eigs shape", eigs.shape)
       
            # print("Class", i, "eigenvalues:", eigs.min())
            mvn = MultivariateNormal(loc=self.means[i], covariance_matrix=self.covs[i])
            # Evaluate probability density for each sample in x.
            # pdf_vals: [batch_size]
            # pdf_vals = torch.exp(mvn.log_prob(x))  # log_prob: [batch_size], exp gives a numerical probability.
            log_pdf_vals = mvn.log_prob(x)  # log_prob: [batch_size], exp gives a numerical probability.
            # Multiply by the prior weight for class i: scalar * [batch_size] = [batch_size]
            logits.append(self.log_weights[i]  + log_pdf_vals)
            
        # Stack the probabilities for each class along dimension 1.
        # probs: [batch_size, n_classes]
        logits = torch.stack(logits, dim=1)
        # Normalize to obtain the posterior probability for each class.
        # posterior = probs / probs.sum(dim=1, keepdim=True)  # [batch_size, n_classes]
        # # Predicted class: index of maximum posterior probability.
        # preds = torch.argmax(posterior, dim=1)  # [batch_size]
        # return posterior, preds
        return logits


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size=128, num_hidden_layers=1, dropout_p=0, num_classes=7):
        """
        An MLP adapted for classifying samples drawn from a Gaussian mixture.
        
        Args:
            input_dim (int): Dimensionality of input features.
                             e.g., 10 → input: [batch_size, 10]
            hidden_size (int): Number of hidden units.
            num_hidden_layers (int): Number of additional hidden blocks.
            dropout_p (float): Dropout probability.
            num_classes (int): Number of classes, e.g., 7.
                             e.g., output: [batch_size, 7]
        """
        super(MLPClassifier, self).__init__()
        # First linear layer maps input_dim to hidden_size.
        # [batch_size, input_dim] -> [batch_size, hidden_size]
        self.layer0 = nn.Linear(input_dim, hidden_size)
        
        # Dropout applied after activation.
        self.dropout = nn.Dropout(dropout_p)
        
        # Build additional hidden layers (each with its own instance).
        if num_hidden_layers > 0:
            self.hidden_layers = nn.Sequential(
                *[ nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),  # [batch_size, hidden_size]
                        nn.ReLU(),                             # [batch_size, hidden_size]
                        nn.Dropout(dropout_p)
                   ) for _ in range(num_hidden_layers)
                ]
            )
        else:
            self.hidden_layers = None
        
        # Final classifier layer produces logits for each class.
        # [batch_size, hidden_size] -> [batch_size, num_classes]
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, input_dim]
                              e.g., [32, 10]
        
        Returns:
            logits (torch.Tensor): Raw output logits, [batch_size, num_classes]
            probs (torch.Tensor): Softmax probabilities, [batch_size, num_classes]
        """
        # Apply first layer and activation.
        x = torch.relu(self.layer0(x))  # [batch_size, hidden_size]
        x = self.dropout(x)             # [batch_size, hidden_size]
        if self.hidden_layers is not None:
            x = self.hidden_layers(x)   # [batch_size, hidden_size]
        logits = self.classifier(x)     # [batch_size, num_classes]
        # probs = torch.softmax(logits, dim=1)  # [batch_size, num_classes]
        # return logits, probs
        return logits

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], num_classes=3):
        """
        A simple multilayer perceptron.
        
        Args:
          - input_dim: Dimension of input features.
          - hidden_dims: List of hidden layer sizes.
          - num_classes: Number of output classes.
        """
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
