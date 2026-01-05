import math
from typing import Iterable, List, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base_scikit_postprocessor import BaseScikitPostprocessor


def _ensure_list(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return list(value)
    return [int(value)]


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation '{name}'")


class _FeedForwardBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: str, dropout: float, use_batchnorm: bool):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_features, out_features)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(_make_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _MLPHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int],
        activation: str,
        dropout: float,
        use_batchnorm: bool,
    ):
        super().__init__()
        dims = [in_features] + list(hidden_dims)
        blocks: List[nn.Module] = []
        for idx in range(len(dims) - 1):
            blocks.append(
                _FeedForwardBlock(
                    in_features=dims[idx],
                    out_features=dims[idx + 1],
                    activation=activation,
                    dropout=dropout,
                    use_batchnorm=use_batchnorm,
                )
            )
        self.backbone = nn.Sequential(*blocks)
        self.classifier = nn.Linear(dims[-1], 1)

        self._reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone:
            x = self.backbone(x)
        return self.classifier(x)

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)


class MLPPostprocessor(BaseScikitPostprocessor):
    """
    Predict misclassification probabilities with a lightweight PyTorch MLP.

    The class follows the same feature-extraction interface as the scikit postprocessors
    (logits, softmax probabilities, or Gini features) but trains a neural head instead of
    handing control to sklearn.
    """

    def __init__(self, model, cfg, result_folder, device=torch.device("cpu")):
        super().__init__(model, cfg, result_folder, device)

        self.normalize_gini = cfg.get("normalize_gini", False)
        self.hidden_dims = _ensure_list(cfg.get("hidden_dims", [1024, 512]))
        self.activation = cfg.get("activation", "relu")
        self.dropout = float(cfg.get("dropout", 0.2))
        self.use_batchnorm = bool(cfg.get("use_batchnorm", False))

        self.num_epochs = int(cfg.get("num_epochs", 10))
        self.batch_size = int(cfg.get("batch_size", 256))
        self.learning_rate = float(cfg.get("lr", 1e-3))
        self.weight_decay = float(cfg.get("weight_decay", 1e-4))
        self.grad_clip_norm = float(cfg.get("grad_clip_norm", 0.0))
        self.positive_class_weight = cfg.get("positive_class_weight", None)
        self.scheduler_cfg = cfg.get("scheduler", None)

        self.model_head: Optional[_MLPHead] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.loss_history: List[float] = []

    def _build_head(self, input_dim: int) -> None:
        self.model_head = _MLPHead(
            in_features=input_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout,
            use_batchnorm=self.use_batchnorm,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model_head.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if self.scheduler_cfg:
            if self.scheduler_cfg.get("name", "").lower() == "cosine":
                t_max = int(self.scheduler_cfg.get("t_max", self.num_epochs))
                eta_min = float(self.scheduler_cfg.get("eta_min", 0.0))
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=t_max,
                    eta_min=eta_min,
                )
            else:
                raise ValueError(f"Unsupported scheduler '{self.scheduler_cfg.get('name')}'")

    def _make_dataloader(self, embs: torch.Tensor, labels: torch.Tensor) -> DataLoader:
        dataset = TensorDataset(embs.float(), labels.float().unsqueeze(1))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    def fit(self, logits=None, detector_labels=None, verbose: bool = False):
        if logits is None or detector_labels is None:
            raise ValueError("Both logits and detector_labels must be provided to fit the MLP postprocessor.")

        embs = self._extract_embeddings(logits=logits).to(torch.device("cpu"))
        labels = detector_labels.detach().to(torch.device("cpu"))

        if self.model_head is None:
            self._build_head(input_dim=embs.shape[1])

        if self.model_head is None or self.optimizer is None:
            raise RuntimeError("MLP head failed to initialise.")

        dataloader = self._make_dataloader(embs, labels)
        pos_weight_tensor = None
        if self.positive_class_weight is not None:
            pos_weight_tensor = torch.tensor(float(self.positive_class_weight), device=self.device)

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        self.model_head.train()
        self.loss_history.clear()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch_embs, batch_labels in dataloader:
                batch_embs = batch_embs.to(self.device)
                batch_labels = batch_labels.to(self.device)

                logits_pred = self.model_head(batch_embs)
                loss = loss_fn(logits_pred, batch_labels)

                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip_norm > 0.0:
                    nn.utils.clip_grad_norm_(self.model_head.parameters(), max_norm=self.grad_clip_norm)
                self.optimizer.step()

                epoch_loss += loss.item() * batch_embs.size(0)

            epoch_loss /= len(dataloader.dataset)
            self.loss_history.append(epoch_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            if verbose:
                print(f"[MLPPostprocessor] epoch {epoch + 1}/{self.num_epochs} loss={epoch_loss:.6f}")

        self.model_head.eval()
        return self

    @torch.no_grad()
    def __call__(self, x=None, logits=None):
        if self.model_head is None:
            raise RuntimeError("MLPPostprocessor must be fitted before calling.")

        if logits is None:
            if x is None:
                raise ValueError("Either logits or inputs must be provided")
            logits = self.model(x)

        embs = self._extract_embeddings(x=x, logits=logits).to(self.device)
        self.model_head.eval()
        logits_out = self.model_head(embs.float())
        return torch.sigmoid(logits_out).squeeze(-1)
