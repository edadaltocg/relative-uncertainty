import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------
# Temperature scaling on logits (adapted from platt_logits)
# ---------------------------------------------------------------------

def platt_logits(calib_loader, max_iters=10, lr=0.01, epsilon=0.01, device="cuda"):
    """
    calib_loader: DataLoader over (logits, targets).
    """
    nll_criterion = nn.CrossEntropyLoss().to(device)

    T = nn.Parameter(torch.tensor([1.3], device=device))

    optimizer = optim.SGD([T], lr=lr)
    for _ in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            x = x.to(device)
            targets = targets.long().to(device)

            optimizer.zero_grad()
            out = x / T
            loss = nll_criterion(out, targets)
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break

    return T.detach()