from functools import partial
import torch
import torch.utils.data
from tqdm import tqdm


def g(logits, temperature=1.0):
    return torch.sum(torch.softmax(logits / temperature, dim=1) ** 2, dim=1)


def doctor(logits: torch.Tensor, temperature: float = 1.0, **kwargs):
    g_out = g(logits=logits, temperature=temperature)
    return (1 - g_out) / g_out


def odin(logits: torch.Tensor, temperature: float = 1.0, **kwargs):
    return -torch.softmax(logits / temperature, dim=1).max(dim=1)[0]


def msp(logits: torch.Tensor, **kwargs):
    return -torch.softmax(logits, dim=1).max(dim=1)[0]


class MetricLearningLagrange:
    def __init__(self, model, lbd=0.5, temperature=1, **kwargs):
        self.model = model
        self.device = next(model.parameters()).device
        self.lbd = lbd
        self.temperature = temperature

        self.params = None

    def fit(self, train_dataloader, *args, **kwargs):
        # get train logits
        train_logits = []
        train_labels = []
        for data, labels in tqdm(train_dataloader, desc="Fitting metric"):
            data = data.to(self.device)
            with torch.no_grad():
                logits = self.model(data).cpu()
            if logits.shape[1] % 2 == 1:  # openmix
                logits = logits[:, :-1]
            train_logits.append(logits)
            train_labels.append(labels)
        train_logits = torch.cat(train_logits, dim=0)
        train_pred = train_logits.argmax(dim=1)
        train_labels = torch.cat(train_labels, dim=0)
        train_labels = (train_labels != train_pred).int()

        train_probs = torch.softmax(train_logits / self.temperature, dim=1)

        train_probs_pos = train_probs[train_labels == 0]
        train_probs_neg = train_probs[train_labels == 1]

        self.params = -(1 - self.lbd) * torch.einsum("ij,ik->ijk", train_probs_pos, train_probs_pos).mean(dim=0).to(
            self.device
        ) + self.lbd * torch.einsum("ij,ik->ijk", train_probs_neg, train_probs_neg).mean(dim=0).to(self.device)
        self.params = torch.tril(self.params, diagonal=-1)
        self.params = self.params + self.params.T
        self.params = torch.relu(self.params)
        if torch.all(self.params <= 0):
            # default to gini
            self.params = torch.ones(self.params.size()).to(self.device)
            self.params = torch.tril(self.params, diagonal=-1)
            self.params = self.params + self.params.T
        self.params = self.params / self.params.norm()

    def __call__(self, logits, *args, **kwds):
        probs = torch.softmax(logits / self.temperature, dim=1)
        params = torch.tril(self.params, diagonal=-1)
        params = params + params.T
        params = params / params.norm()
        return torch.diag(probs @ params @ probs.T)


class Wrapper:
    def __init__(self, method, *args, **kwargs):
        self.method = method

    def fit(self, train_dataloader, val_dataloader, **kwargs):
        if hasattr(self.method, "fit"):
            self.method.fit(train_dataloader, val_dataloader, **kwargs)
        return self

    def __call__(self, x):
        return self.method(x)


def get_method(method_name, *args, **kwargs):
    if method_name == "doctor":
        return Wrapper(partial(doctor, *args, **kwargs))
    if method_name == "odin":
        return Wrapper(partial(odin, *args, **kwargs))
    if method_name == "msp":
        return Wrapper(msp)
    if method_name == "metric_lagrange":
        return Wrapper(MetricLearningLagrange(*args, **kwargs))
    raise ValueError(f"Method {method_name} not supported")
