from functools import partial
from typing import Any
import torch
import torch.utils.data
from tqdm import tqdm
import copy
from src.utils.eval import evaluate


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


class MLP(torch.nn.Module):
    def __init__(self, num_classes, hidden_size=128, num_hidden_layers=1, *args, **kwargs) -> None:
        super().__init__()
        self.layer0 = torch.nn.Linear(num_classes, hidden_size)
        self.classifier = torch.nn.Linear(hidden_size, 1)
        self.hidden_layers = None
        if num_hidden_layers > 0:
            self.hidden_layers = torch.nn.Sequential(
                *([torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()] * num_hidden_layers)
            )

    def forward(self, x):
        x = torch.relu(self.layer0(x))
        if self.hidden_layers is not None:
            x = self.hidden_layers(x)
        return torch.sigmoid(self.classifier(x))


class MLPTrainer:
    def __init__(self, model, num_classes, epochs=100, hidden_size=256, num_hidden_layers=2, *args, **kwargs) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.net = MLP(num_classes, hidden_size, num_hidden_layers)
        self.net = self.net.to(self.device)
        self.criterion = torch.nn.BCELoss()
        self.epochs = epochs

    def fit(self, train_dataloader, val_dataloader, *args, **kwargs):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-1)
        best_acc = 0
        best_fpr = 1
        best_auc = 0
        loss = torch.inf
        best_weights = copy.deepcopy(self.net.to("cpu").state_dict())
        self.net = self.net.to(self.device)
        progress_bar = tqdm(range(self.epochs), desc="Fit", unit="epoch")
        for e in progress_bar:
            # train step
            self.net.train()
            for data, labels in train_dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    logits = self.model(data)
                model_pred = logits.argmax(dim=1)
                bin_labels = (model_pred != labels).float()
                y_pred = self.net(logits)
                loss = self.criterion(y_pred.view(-1), bin_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss = loss.item()

            # eval
            self.net.eval()
            scores = []
            targets = []
            preds = []
            for data, labels in val_dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    logits = self.model(data)
                    model_pred = logits.argmax(dim=1)
                    bin_labels = (model_pred != labels).int()
                    y_pred = self.net(logits)
                preds.append(y_pred.round())
                targets.append(bin_labels.view(-1))
                scores.append(y_pred.view(-1))
            targets = torch.cat(targets)
            scores = torch.cat(scores)
            preds = torch.cat(preds)
            acc, roc_auc, fpr = evaluate(preds, targets, scores)
            # if acc > best_acc:
            #     best_acc = acc
            #     best_weights = copy.deepcopy(self.net.to("cpu").state_dict())
            #     self.net = self.net.to(self.device)
            if fpr < best_fpr:
                best_fpr = fpr
                best_weights = copy.deepcopy(self.net.to("cpu").state_dict())
                self.net = self.net.to(self.device)

            progress_bar.set_postfix(loss=loss, acc=acc, fpr=fpr, best_fpr=best_fpr, auc=roc_auc)

        self.net.load_state_dict(best_weights)
        self.net = self.net.to(self.device)

    def __call__(self, logits, *args: Any, **kwds: Any) -> Any:
        logits_device = logits.device
        logits = logits.to(self.device)
        self.net.eval()
        return self.net(logits).to(logits_device)


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
    if method_name == "mlp":
        return Wrapper(MLPTrainer(*args, **kwargs))
    raise ValueError(f"Method {method_name} not supported")
