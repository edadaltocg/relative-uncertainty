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


def entropy(logits: torch.Tensor, **kwargs):
    probs = torch.softmax(logits, dim=1)
    return -torch.sum(probs * torch.log(probs), dim=1)


def enable_dropout(model):
    """Function to enable the dropout layers during test-time."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def add_dropout_layer(model, dropout_p=0.5):
    """Function to add dropout layers to the model.

    replace model.linear by a sequential model with dropout and the same linear layer
    """
    # get the last layer
    last_layer = model.linear
    # remove it
    model.linear = torch.nn.Sequential()
    # add dropout
    model.linear.add_module("dropout", torch.nn.Dropout(dropout_p))
    # add the last layer
    model.linear.add_module("linear", last_layer)


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

    def export_matrix(self):
        return self.params.cpu()


class MLP(torch.nn.Module):
    def __init__(self, num_classes, hidden_size=128, num_hidden_layers=1, dropout_p=0, *args, **kwargs) -> None:
        super().__init__()
        self.layer0 = torch.nn.Linear(num_classes, hidden_size)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.classifier = torch.nn.Linear(hidden_size, 1)
        self.hidden_layers = None
        if num_hidden_layers > 0:
            self.hidden_layers = torch.nn.Sequential(
                *(
                    [
                        torch.nn.Linear(hidden_size, hidden_size),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(dropout_p),
                    ]
                    * num_hidden_layers
                ),
            )

    def forward(self, x):
        x = torch.relu(self.layer0(x))
        x = self.dropout(x)
        if self.hidden_layers is not None:
            x = self.hidden_layers(x)
        return torch.sigmoid(self.classifier(x))


class MLPTrainer:
    def __init__(self, model, num_classes, epochs=100, hidden_size=128, num_hidden_layers=2, *args, **kwargs) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.net = MLP(num_classes, hidden_size, num_hidden_layers)
        self.net = self.net.to(self.device)
        self.criterion = torch.nn.BCELoss()
        self.epochs = epochs

    def fit(self, train_dataloader, val_dataloader, *args, **kwargs):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        best_acc = 0
        best_fpr = 1
        best_auc = 0
        loss = torch.inf
        best_weights = copy.deepcopy(self.net.to("cpu").state_dict())
        self.net = self.net.to(self.device)
        progress_bar = tqdm(range(self.epochs), desc="Fit", unit="e")
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
            acc, roc_auc, fpr, aurc = evaluate(preds, targets, scores)
            # if acc > best_acc:
            #     best_acc = acc
            #     best_weights = copy.deepcopy(self.net.to("cpu").state_dict())
            #     self.net = self.net.to(self.device)
            if fpr < best_fpr:
                best_fpr = fpr
                best_auc = roc_auc
                best_aurc = aurc
                best_weights = copy.deepcopy(self.net.to("cpu").state_dict())
                self.net = self.net.to(self.device)
            # if roc_auc > best_auc:
            #     best_auc = roc_auc
            #     best_weights = copy.deepcopy(self.net.to("cpu").state_dict())
            #     self.net = self.net.to(self.device)

            progress_bar.set_postfix(l=loss, acc=acc, fpr=fpr, b_auc=best_auc, b_fpr=best_fpr, auc=roc_auc)

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

    def export_matrix(self):
        return self.method.export_matrix()


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
    if method_name == "mc_dropout":
        return Wrapper(entropy)
    if method_name == "ensemble":
        return Wrapper(msp)
    raise ValueError(f"Method {method_name} not supported")
