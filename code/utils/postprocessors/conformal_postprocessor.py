import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as tdata

from .base_postprocessor import BasePostprocessor
from code.utils.calibration import platt_logits

# ---------------------------------------------------------------------
# Wrapper: ConformalPostprocessor to plug into your pipeline
# ---------------------------------------------------------------------

class ConformalPostprocessor(BasePostprocessor):
    """
    Wrapper that exposes the RAPS-style conformal classifier as a postprocessor.

    - fit(logits, targets): calibrates T, k_reg, λ, and Q̂ on calibration set
      using their tuning procedure.
    - __call__(logits): returns |S(x)| (size of conformal prediction set) as score.
      Larger score = more uncertain / more error-prone.
    """

    def __init__(self, model, cfg, result_folder, device=torch.device("cpu")):
        super().__init__(model=model, cfg=cfg, result_folder=result_folder, device=device)

        # Main conformal hyperparameters
        self.alpha = float(cfg.get("alpha", 0.1))

        # Allow "auto" selection of kreg / lamda if not provided
        raw_kreg = cfg.get("kreg", None)
        self.kreg = None if raw_kreg is None else int(raw_kreg)

        raw_lamda = cfg.get("lamda", cfg.get("lambda", None))
        self.lamda = None if raw_lamda is None else float(raw_lamda)

        self.randomized = bool(cfg.get("randomized", True))
        self.allow_zero_sets = bool(cfg.get("allow_zero_sets", False))
        self.batch_size = int(cfg.get("batch_size", 32))

        # Tuning hyperparameters
        self.pct_paramtune = float(cfg.get("pct_paramtune", 0.3))
        self.lamda_criterion = cfg.get("lamda_criterion", "size")  # "size" or "adaptiveness"

        # Flags (kept for completeness; typically False in our use)
        self.naive = bool(cfg.get("naive", False))
        self.LAC = bool(cfg.get("LAC", False))

        self.cmodel = None

    # @torch.no_grad()
    def fit(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs):
        """
        Args:
            logits  : [n_cal, C] calibration logits
            targets : [n_cal]    calibration labels
        """
        logits_cpu = logits.detach().cpu()
        targets_cpu = targets.detach().cpu().long()

        calib_dataset = tdata.TensorDataset(logits_cpu, targets_cpu)
        
        # Automatic parameter tuning if needed (their method)
        if self.kreg is None or self.lamda is None:
        
            self.kreg, self.lamda, calib_subset = pick_parameters(
                model=self.model,
                calib_dataset=calib_dataset,
                alpha=self.alpha,
                kreg=self.kreg,
                lamda=self.lamda,
                randomized=self.randomized,
                allow_zero_sets=self.allow_zero_sets,
                pct_paramtune=self.pct_paramtune,
                batch_size=self.batch_size,
                lamda_criterion=self.lamda_criterion,
            )
        else:
            calib_subset = calib_dataset

        calib_loader = tdata.DataLoader(
            calib_subset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

        self.cmodel = ConformalModelLogits(
            model=self.model,
            calib_loader=calib_loader,
            alpha=self.alpha,
            kreg=self.kreg,
            lamda=self.lamda,
            randomized=self.randomized,
            allow_zero_sets=self.allow_zero_sets,
            naive=self.naive,
            LAC=self.LAC,
        )

@torch.no_grad()
def __call__(self, inputs=None, logits=None) -> torch.Tensor:
    """
    Returns:
        scores: [B] tensor, score(x) = second largest probability within S(x)
                (0 if |S(x)| = 1). Larger score = more ambiguous prediction.
    """
    if self.cmodel is None:
        raise RuntimeError("ConformalPostprocessor.fit must be called before __call__.")

    if logits is None:
        if inputs is None:
            raise ValueError("Either logits or inputs must be provided.")
        inputs = inputs.to(self.device)
        self.model.eval()
        logits = self.model(inputs)
    else:
        logits = logits.to(self.device)

    # Get conformal prediction sets S(x)
    _, S = self.cmodel(logits)   # S: list of index arrays (one per sample)

    # Use the same temperature as in conformal calibration
    probs = torch.softmax(logits / self.cmodel.T.item(), dim=1)  # [B, C]

    second_probs = []
    for i, idx in enumerate(S):
        idx = torch.as_tensor(idx, device=self.device, dtype=torch.long)
        if idx.numel() < 2:
            # Only one label in the set: no ambiguity → second prob = 0
            second_probs.append(torch.tensor(0.0, device=self.device))
        else:
            p_set = probs[i, idx]                 # probs restricted to S(x)
            top2, _ = torch.topk(p_set, k=2)      # largest and second largest
            second_probs.append(top2[-1])         # second largest

    second_probs = torch.stack(second_probs, dim=0)  # [B]

    # This is your detection *score*: larger = more ambiguous / likely misclass.
    return second_probs



# ---------------------------------------------------------------------
# Logits-based conformal model (adapted from ConformalModelLogits)
# ---------------------------------------------------------------------

class ConformalModelLogits(nn.Module):
    def __init__(
        self,
        model,
        calib_loader,
        alpha,
        kreg,
        lamda,
        randomized=True,
        allow_zero_sets=False,
        naive=False,
        LAC=False,
    ):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.randomized = randomized
        self.LAC = LAC
        self.allow_zero_sets = allow_zero_sets

        if model is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device("cpu")

        # 1) temperature on calibration logits
        self.T = platt_logits(calib_loader, device=self.device)

        # 2) penalties
        num_classes = calib_loader.dataset[0][0].shape[-1]
        self.penalties = np.zeros((1, num_classes))
        if (kreg is not None) and (lamda is not None) and (not naive) and (not LAC):
            self.penalties[:, kreg:] += lamda

        # 3) Qhat
        if naive:
            self.Qhat = 1 - alpha
        elif LAC:
            raise NotImplementedError("LAC mode not implemented in this wrapper.")
        else:
            self.Qhat = conformal_calibration_logits(self, calib_loader)

    def forward(self, logits, randomized=None, allow_zero_sets=None):
        if randomized is None:
            randomized = self.randomized
        if allow_zero_sets is None:
            allow_zero_sets = self.allow_zero_sets

        with torch.no_grad():
            logits = logits.to(self.device)
            scores_t = torch.softmax(logits / self.T.item(), dim=1)
            scores = scores_t.cpu().numpy()

            if not self.LAC:
                I, ordered, cumsum = sort_sum(scores)
                S = gcq(
                    scores,
                    self.Qhat,
                    I=I,
                    ordered=ordered,
                    cumsum=cumsum,
                    penalties=self.penalties,
                    randomized=randomized,
                    allow_zero_sets=allow_zero_sets,
                )
            else:
                raise NotImplementedError("LAC mode not implemented here.")

        return logits, S





# ---------------------------------------------------------------------
# Helper: sorting + cumulative sums (from their utils.sort_sum)
# ---------------------------------------------------------------------

def sort_sum(scores: np.ndarray):
    """
    scores: [B, C] numpy array of probabilities.

    Returns:
        I       : [B, C] indices of classes sorted by decreasing prob
        ordered : [B, C] sorted probabilities
        cumsum  : [B, C] cumulative sum of ordered along axis 1
    """
    I = scores.argsort(axis=1)[:, ::-1]
    ordered = np.sort(scores, axis=1)[:, ::-1]
    cumsum = np.cumsum(ordered, axis=1)
    return I, ordered, cumsum


# ---------------------------------------------------------------------
# Core conformal / RAPS functions (same math as their code)
# ---------------------------------------------------------------------

def gcq(scores, tau, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
    """
    Generalized conditional quantile function.

    scores   : [B, C]
    tau      : scalar in [0, 1]
    penalties: [1, C]
    """
    penalties_cumsum = np.cumsum(penalties, axis=1)
    sizes_base = ((cumsum + penalties_cumsum) <= tau).sum(axis=1) + 1  # 1..C+1
    sizes_base = np.minimum(sizes_base, scores.shape[1])               # 1..C

    if randomized:
        V = np.zeros(sizes_base.shape)
        for i in range(sizes_base.shape[0]):
            V[i] = 1.0 / ordered[i, sizes_base[i] - 1] * (
                tau
                - (cumsum[i, sizes_base[i] - 1] - ordered[i, sizes_base[i] - 1])
                - penalties_cumsum[0, sizes_base[i] - 1]
            )

        sizes = sizes_base - (np.random.random(V.shape) >= V).astype(int)
    else:
        sizes = sizes_base

    if tau == 1.0:
        sizes[:] = cumsum.shape[1]  # avoid numerical issues when alpha == 0

    if not allow_zero_sets:
        sizes[sizes == 0] = 1  # enforce non-empty sets if requested

    S = []
    for i in range(I.shape[0]):
        S.append(I[i, 0:sizes[i]])
    
    return S


def get_tau(score, target, I, ordered, cumsum, penalty, randomized, allow_zero_sets):
    """
    score   : [1, C] scores for a single example (not used directly)
    target  : int label
    I       : [1, C] sorted indices
    ordered : [1, C] sorted probabilities
    cumsum  : [1, C] cumulative sums
    penalty : [C] penalty vector
    """
    idx = np.where(I == target)
    tau_nonrandom = cumsum[idx]

    if not randomized:
        return tau_nonrandom + penalty[0]

    U = np.random.random()

    if idx == (0, 0):
        if not allow_zero_sets:
            return tau_nonrandom + penalty[0]
        else:
            return U * tau_nonrandom + penalty[0]
    else:
        return (
            U * ordered[idx]
            + cumsum[(idx[0], idx[1] - 1)]
            + (penalty[0:(idx[1][0] + 1)]).sum()
        )


def giq(scores, targets, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
    """
    Generalized inverse quantile conformity score function.
    E from equation (7) in Romano, Sesia, Candès:
    minimum tau in [0, 1] such that the correct label enters the set.

    scores  : [B, C]
    targets : array of shape [B] (ints)
    """
    E = -np.ones((scores.shape[0],))
    for i in range(scores.shape[0]):
        E[i] = get_tau(
            scores[i:i + 1, :],
            targets[i].item(),
            I[i:i + 1, :],
            ordered[i:i + 1, :],
            cumsum[i:i + 1, :],
            penalties[0, :],
            randomized=randomized,
            allow_zero_sets=allow_zero_sets,
        )

    return E




# ---------------------------------------------------------------------
# Conformal calibration on logits
# ---------------------------------------------------------------------

def conformal_calibration_logits(cmodel, calib_loader):
    """
    Compute Qhat on calibration logits.

    cmodel.T        : scalar temperature tensor
    cmodel.alpha    : miscoverage level
    cmodel.penalties: [1, C] numpy array
    """
    E = []
    with torch.no_grad():
        for logits, targets in calib_loader:
            logits = logits.to(cmodel.device)
            targets = targets.to(cmodel.device)

            scores_t = torch.softmax(logits / cmodel.T.item(), dim=1)
            scores = scores_t.cpu().numpy()  # [B, C]

            I, ordered, cumsum = sort_sum(scores)
            targets_np = targets.cpu().numpy()

            E_batch = giq(
                scores,
                targets_np,
                I=I,
                ordered=ordered,
                cumsum=cumsum,
                penalties=cmodel.penalties,
                randomized=True,
                allow_zero_sets=True,
            )
            E.append(E_batch)

    E = np.concatenate(E, axis=0)
    Qhat = np.quantile(E, 1 - cmodel.alpha, interpolation="higher")
    return Qhat


# ---------------------------------------------------------------------
# Validate on logits (replacement for their validate, but with logits)
# ---------------------------------------------------------------------

def validate_logits(loader, cmodel):
    """
    Evaluate conformal model on a logits+targets loader.

    Returns:
        top1_avg, top5_avg, coverage_avg, set_size_avg
    """
    device = cmodel.device
    total = 0
    top1_correct = 0
    top5_correct = 0
    cover_correct = 0
    size_sum = 0.0

    with torch.no_grad():
        for logits, targets in loader:
            logits = logits.to(device)
            targets = targets.to(device)

            # base top-1 / top-5 accuracy from logits
            preds = logits.argmax(dim=1)
            top1_correct += (preds == targets).sum().item()

            if logits.shape[1] >= 5:
                top5 = logits.topk(5, dim=1).indices
                top5_correct += (
                    (top5 == targets.unsqueeze(1))
                    .any(dim=1)
                    .sum()
                    .item()
                )
            else:
                top5_correct += (preds == targets).sum().item()

            # conformal sets
            _, S = cmodel(logits)

            bs = targets.size(0)
            for i in range(bs):
                size_sum += len(S[i])
                cover_correct += int(targets[i].item() in S[i])

            total += bs

    top1_avg = top1_correct / total
    top5_avg = top5_correct / total
    coverage_avg = cover_correct / total
    set_size_avg = size_sum / total

    return top1_avg, top5_avg, coverage_avg, set_size_avg


# ---------------------------------------------------------------------
# Automatic parameter tuning (same logic as their pick_* functions)
# ---------------------------------------------------------------------

def pick_kreg(paramtune_dataset, alpha: float) -> int:
    """
    paramtune_dataset: dataset of (logits, target), logits shape [C].

    Returns k_reg (1-based).
    """
    gt_locs = []
    for logits, target in paramtune_dataset:
        logits_np = logits.detach().cpu().numpy().reshape(-1)
        sorted_idx = np.argsort(logits_np)[::-1]
        rank = int(np.where(sorted_idx == int(target))[0][0])
        gt_locs.append(rank)

    gt_locs = np.array(gt_locs)
    kstar = np.quantile(gt_locs, 1 - alpha, interpolation="higher") + 1
    return int(kstar)


def pick_lamda_size(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets):
    """
    Choose λ by minimizing average set size on param-tune set.
    """
    first_batch = next(iter(paramtune_loader))
    num_classes = first_batch[0].shape[-1]

    best_size = float(num_classes)
    lamda_star = 0.0

    for temp_lam in [0.001, 0.01, 0.1, 0.2, 0.5]:
        cmodel = ConformalModelLogits(
            model=model,
            calib_loader=paramtune_loader,
            alpha=alpha,
            kreg=kreg,
            lamda=temp_lam,
            randomized=randomized,
            allow_zero_sets=allow_zero_sets,
            naive=False,
            LAC=False,
        )
        _, _, _, sz_avg = validate_logits(paramtune_loader, cmodel)
        if sz_avg < best_size:
            best_size = sz_avg
            lamda_star = temp_lam

    return lamda_star


def get_violation(cmodel, loader_paramtune, strata, alpha):
    """
    Worst-case coverage violation over size strata.
    """
    sizes_list = []
    correct_list = []

    device = cmodel.device
    with torch.no_grad():
        for logits, targets in loader_paramtune:
            logits = logits.to(device)
            targets = targets.to(device)
            _, S = cmodel(logits)

            bs = targets.size(0)
            for i in range(bs):
                sizes_list.append(len(S[i]))
                correct_list.append(1 if targets[i].item() in S[i] else 0)

    df = pd.DataFrame({"size": sizes_list, "correct": correct_list})

    wc_violation = 0.0
    for low, high in strata:
        tmp = df[(df["size"] >= low) & (df["size"] <= high)]
        if len(tmp) == 0:
            continue
        stratum_violation = abs(tmp["correct"].mean() - (1 - alpha))
        wc_violation = max(wc_violation, stratum_violation)

    return wc_violation


def pick_lamda_adaptiveness(
    model,
    paramtune_loader,
    alpha,
    kreg,
    randomized,
    allow_zero_sets,
    strata=None,
):
    """
    Choose λ by minimizing worst-case coverage violation across size strata.
    """
    if strata is None:
        strata = [[0, 1], [2, 3], [4, 6], [7, 10], [11, 100], [101, 1000]]

    lamda_star = 0.0
    best_violation = 1.0
    grid = [0, 1e-5, 1e-4, 8e-4, 9e-4, 1e-3, 1.5e-3, 2e-3]

    for temp_lam in grid:
        cmodel = ConformalModelLogits(
            model=model,
            calib_loader=paramtune_loader,
            alpha=alpha,
            kreg=kreg,
            lamda=temp_lam,
            randomized=randomized,
            allow_zero_sets=allow_zero_sets,
            naive=False,
            LAC=False,
        )
        curr_violation = get_violation(cmodel, paramtune_loader, strata, alpha)
        if curr_violation < best_violation:
            best_violation = curr_violation
            lamda_star = temp_lam

    return lamda_star


def pick_parameters(
    model,
    calib_dataset,         # TensorDataset or Subset of (logits, target)
    alpha,
    kreg,
    lamda,
    randomized,
    allow_zero_sets,
    pct_paramtune,
    batch_size,
    lamda_criterion,
):
    """
    Split calibration dataset into param-tune and calibration subsets, then
    choose k_reg and λ using their procedures.
    """
    n_total = len(calib_dataset)
    n_paramtune = int(np.ceil(pct_paramtune * n_total))
    if n_paramtune <= 0 or n_paramtune >= n_total:
        raise ValueError(f"pct_paramtune={pct_paramtune} gives empty split.")

    indices = np.random.permutation(n_total)
    param_idx = indices[:n_paramtune]
    calib_idx = indices[n_paramtune:]

    paramtune_subset = tdata.Subset(calib_dataset, param_idx.tolist())
    calib_subset = tdata.Subset(calib_dataset, calib_idx.tolist())

    paramtune_loader = tdata.DataLoader(
        paramtune_subset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    if kreg is None:
        kreg = pick_kreg(paramtune_subset, alpha)

    if lamda is None:
        if lamda_criterion == "size":
            lamda = pick_lamda_size(
                model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets
            )
        elif lamda_criterion == "adaptiveness":
            lamda = pick_lamda_adaptiveness(
                model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets
            )
        else:
            raise ValueError(f"Unknown lamda_criterion: {lamda_criterion}")

    return kreg, lamda, calib_subset

