# soc_rnn_downstream.py
# Downstream test for Recurrent SOC-ICNN on convex trajectory smoothing.
#
# Output:
#   results_soc_rnn_downstream/
#     summary.csv
#     summary_table.tex
#     rnn_downstream_examples.pdf
#     rnn_downstream_examples.png
#     training_curves.png
#     training_curves.pdf
#     config.json

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    seed: int = 0
    device: str = "cuda"
    out_dir: str = "results_soc_rnn_downstream"

    # sequence data
    T: int = 40
    p: int = 4
    train_size: int = 1200
    val_size: int = 200
    test_size: int = 200
    candidates_per_sequence: int = 14

    # true convex trajectory energy
    rho: float = 1.0          # tracking weight
    beta_acc: float = 0.20    # quadratic acceleration smoothing
    lam_tv: float = 0.18      # conic switching / temporal TV
    gamma_u: float = 0.02     # weak control magnitude regularization
    tv_eps: float = 1e-5

    # model
    hidden_dim: int = 20
    branch_channels: int = 24
    norm_groups: int = 12

    # training
    epochs: int = 50
    batch_size: int = 1200
    lr: float = 2e-3
    weight_decay: float = 1e-6

    # downstream optimization
    true_opt_steps: int = 900
    true_opt_lr: float = 0.045
    surrogate_opt_steps: int = 400
    surrogate_opt_lr: float = 0.04
    restarts: int = 4
    eval_batch_size: int = 64

    # visualization
    num_plot_examples: int = 5
    plot_indices: Tuple[int, ...] = (0, 7, 19, 42, 88)
    plot_dim: int = 0

    num_workers: int = 0


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Reference sequence generation
# ============================================================

def generate_reference_sequences(
    n: int,
    cfg: Config,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generates nontrivial reference trajectories R in [-1,1]^{T x p}.
    Each sequence combines smooth sinusoidal components, low-frequency drift,
    and a few piecewise changes. This gives a controlled but non-toy temporal
    downstream optimization task.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    T_len, p_dim = cfg.T, cfg.p
    t = torch.linspace(0.0, 1.0, T_len, device=device)

    refs = torch.zeros(n, T_len, p_dim, device=device)

    for i in range(n):
        for j in range(p_dim):
            amp1 = 0.25 + 0.45 * torch.rand((), generator=gen, device=device)
            amp2 = 0.10 + 0.25 * torch.rand((), generator=gen, device=device)
            freq1 = torch.randint(1, 4, (1,), generator=gen, device=device).float().item()
            freq2 = torch.randint(2, 6, (1,), generator=gen, device=device).float().item()
            phase1 = 2.0 * np.pi * torch.rand((), generator=gen, device=device)
            phase2 = 2.0 * np.pi * torch.rand((), generator=gen, device=device)

            base = (
                amp1 * torch.sin(2.0 * np.pi * freq1 * t + phase1)
                + amp2 * torch.cos(2.0 * np.pi * freq2 * t + phase2)
            )

            # Piecewise offset.
            num_jumps = int(torch.randint(1, 4, (1,), generator=gen, device=device).item())
            offset = torch.zeros_like(t)
            for _ in range(num_jumps):
                jump_t = int(torch.randint(5, T_len - 5, (1,), generator=gen, device=device).item())
                jump_size = 0.15 * torch.randn((), generator=gen, device=device)
                offset[jump_t:] += jump_size

            # Mild AR-like local perturbation.
            noise = 0.04 * torch.randn(T_len, generator=gen, device=device)
            for k in range(1, T_len):
                noise[k] = 0.85 * noise[k - 1] + noise[k]

            refs[i, :, j] = base + offset + noise

    refs = torch.clamp(refs, -1.0, 1.0)
    return refs.detach()


def smooth_sequence(u: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Moving-average smoothing along time.
    u: [B, T, p]
    """
    b, T_len, p_dim = u.shape
    x = u.transpose(1, 2)  # [B, p, T]
    kernel = torch.ones(p_dim, 1, kernel_size, device=u.device, dtype=u.dtype) / kernel_size
    y = F.conv1d(x, kernel, padding=kernel_size // 2, groups=p_dim)
    return y.transpose(1, 2)


# ============================================================
# True convex sequence energy
# ============================================================

def true_energy(u: torch.Tensor, r: torch.Tensor, cfg: Config) -> torch.Tensor:
    """
    Convex trajectory energy:
        rho/2 * mean_t ||u_t - r_t||^2
      + beta/2 * mean_t ||u_{t+1} - 2u_t + u_{t-1}||^2
      + lambda * mean_t sqrt(||u_t - u_{t-1}||^2 + eps)
      + gamma/2 * mean_t ||u_t||^2

    Returns:
        Tensor of shape [B].
    """
    tracking = 0.5 * cfg.rho * (u - r).pow(2).mean(dim=(1, 2))

    acc = u[:, 2:, :] - 2.0 * u[:, 1:-1, :] + u[:, :-2, :]
    acc_energy = 0.5 * cfg.beta_acc * acc.pow(2).mean(dim=(1, 2))

    du = u[:, 1:, :] - u[:, :-1, :]
    tv = cfg.lam_tv * torch.sqrt(du.pow(2).sum(dim=2) + cfg.tv_eps).mean(dim=1)

    mag = 0.5 * cfg.gamma_u * u.pow(2).mean(dim=(1, 2))

    return tracking + acc_energy + tv + mag


# ============================================================
# Candidate supervised dataset
# ============================================================

def build_candidate_dataset(
    refs: torch.Tensor,
    cfg: Config,
    device: torch.device,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Builds supervised pairs (candidate decision U, context R, true energy value).
    """
    refs = refs.to(device)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    us = []
    rs = []
    vals = []

    n = refs.shape[0]
    smooth_refs = smooth_sequence(refs)

    for j in range(cfg.candidates_per_sequence):
        if j == 0:
            cand = refs.clone()

        elif j == 1:
            cand = smooth_refs.clone()

        elif j == 2:
            cand = torch.zeros_like(refs)

        elif j == 3:
            cand = torch.empty_like(refs).uniform_(-1.0, 1.0)

        elif j == 4:
            cand = torch.clamp(refs + 0.25 * torch.randn(refs.shape, generator=gen, device=device), -1.0, 1.0)

        elif j == 5:
            cand = torch.clamp(smooth_refs + 0.20 * torch.randn(refs.shape, generator=gen, device=device), -1.0, 1.0)

        elif j % 4 == 0:
            alpha = torch.rand((n, 1, 1), generator=gen, device=device)
            random_u = torch.empty_like(refs).uniform_(-1.0, 1.0)
            cand = torch.clamp(alpha * refs + (1.0 - alpha) * random_u, -1.0, 1.0)

        elif j % 4 == 1:
            alpha = torch.rand((n, 1, 1), generator=gen, device=device)
            cand = torch.clamp(alpha * refs + (1.0 - alpha) * smooth_refs, -1.0, 1.0)

        elif j % 4 == 2:
            random_u = torch.empty_like(refs).uniform_(-1.0, 1.0)
            cand = smooth_sequence(random_u)

        else:
            alpha = torch.rand((n, 1, 1), generator=gen, device=device)
            noise = 0.30 * torch.randn(refs.shape, generator=gen, device=device)
            cand = torch.clamp(alpha * smooth_refs + (1.0 - alpha) * (refs + noise), -1.0, 1.0)

        val = true_energy(cand, refs, cfg)

        us.append(cand.detach().cpu())
        rs.append(refs.detach().cpu())
        vals.append(val.detach().cpu())

    u_all = torch.cat(us, dim=0)
    r_all = torch.cat(rs, dim=0)
    v_all = torch.cat(vals, dim=0)

    return u_all, r_all, v_all


# ============================================================
# Recurrent convex surrogate models
# ============================================================

class PositiveLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.raw_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.normal_(self.raw_weight, mean=-3.0, std=0.15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.softplus(self.raw_weight)
        return F.linear(x, weight, self.bias)


class ContextAffineInU(nn.Module):
    """
    Context-dependent affine term in decision sequence U:
        <a(R), U> + b(R).

    For fixed R, this is affine in U, so convexity in U is preserved.
    """
    def __init__(self, T_len: int, p_dim: int, hidden: int = 64):
        super().__init__()
        self.T_len = T_len
        self.p_dim = p_dim
        in_dim = T_len * p_dim
        out_dim = T_len * p_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

        self.bias_net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, u: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        b = u.shape[0]
        r_flat = r.reshape(b, -1)
        coeff = self.net(r_flat).view(b, self.T_len, self.p_dim)
        bias = self.bias_net(r_flat).squeeze(-1)
        lin = (coeff * u).mean(dim=(1, 2))
        return lin + bias


class RNNICNNBackbone(nn.Module):
    """
    Recurrent ICNN backbone, convex in decision sequence U for fixed context R:

        z_t = act(W_u u_t + W_r r_t + U z_{t-1} + b),

    where U has nonnegative weights. Output weight c is also nonnegative.
    """
    def __init__(
        self,
        p_dim: int,
        T_len: int,
        hidden_dim: int,
        activation: str = "relu",
    ):
        super().__init__()
        self.p_dim = p_dim
        self.T_len = T_len
        self.hidden_dim = hidden_dim
        self.activation_name = activation

        self.u_linear = nn.Linear(p_dim, hidden_dim)
        self.r_linear = nn.Linear(p_dim, hidden_dim)
        self.z_linear = PositiveLinear(hidden_dim, hidden_dim)

        self.raw_out_weight = nn.Parameter(torch.full((hidden_dim,), -2.0))
        self.affine_u = ContextAffineInU(T_len=T_len, p_dim=p_dim, hidden=64)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.u_linear.weight, gain=0.7)
        nn.init.zeros_(self.u_linear.bias)
        nn.init.xavier_uniform_(self.r_linear.weight, gain=0.7)
        nn.init.zeros_(self.r_linear.bias)

    def activation(self, a: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "relu":
            return F.relu(a)
        if self.activation_name == "softplus":
            return F.softplus(a, beta=1.0)
        raise ValueError(f"Unknown activation: {self.activation_name}")

    def forward(self, u: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        b = u.shape[0]
        z = torch.zeros(b, self.hidden_dim, device=u.device, dtype=u.dtype)

        for t in range(self.T_len):
            a = self.u_linear(u[:, t, :]) + self.r_linear(r[:, t, :])
            if t > 0:
                a = a + self.z_linear(z)
            z = self.activation(a)

        out_weight = F.softplus(self.raw_out_weight)
        convex_out = (z * out_weight).mean(dim=1)
        affine_out = self.affine_u(u, r)

        return convex_out + affine_out


class TemporalQuadraticBranch(nn.Module):
    """
    Quadratic temporal branch:
        alpha/2 || B_u * U + B_r * R + e ||^2,
    implemented by 1D convolutions over time.
    """
    def __init__(self, p_dim: int, out_channels: int):
        super().__init__()
        self.u_conv = nn.Conv1d(p_dim, out_channels, kernel_size=5, padding=2)
        self.r_conv = nn.Conv1d(p_dim, out_channels, kernel_size=5, padding=2)
        self.raw_alpha = nn.Parameter(torch.tensor(-0.3))

        nn.init.xavier_uniform_(self.u_conv.weight, gain=0.45)
        nn.init.xavier_uniform_(self.r_conv.weight, gain=0.45)
        nn.init.zeros_(self.u_conv.bias)
        nn.init.zeros_(self.r_conv.bias)

    def forward(self, u: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        u_ch = u.transpose(1, 2)
        r_ch = r.transpose(1, 2)
        q = self.u_conv(u_ch) + self.r_conv(r_ch)
        alpha = F.softplus(self.raw_alpha)
        return 0.5 * alpha * q.pow(2).mean(dim=(1, 2))


class TemporalNormBranch(nn.Module):
    """
    Conic temporal branch:
        lambda * mean grouped 2D local norms.
    """
    def __init__(self, p_dim: int, groups: int):
        super().__init__()
        self.groups = groups
        out_channels = 2 * groups

        self.u_conv = nn.Conv1d(p_dim, out_channels, kernel_size=5, padding=2)
        self.r_conv = nn.Conv1d(p_dim, out_channels, kernel_size=5, padding=2)
        self.raw_lambda = nn.Parameter(torch.tensor(-0.3))

        nn.init.xavier_uniform_(self.u_conv.weight, gain=0.45)
        nn.init.xavier_uniform_(self.r_conv.weight, gain=0.45)
        nn.init.zeros_(self.u_conv.bias)
        nn.init.zeros_(self.r_conv.bias)

    def forward(self, u: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        u_ch = u.transpose(1, 2)
        r_ch = r.transpose(1, 2)
        z = self.u_conv(u_ch) + self.r_conv(r_ch)

        batch, channels, T_len = z.shape
        z = z.view(batch, self.groups, 2, T_len)

        norms = torch.sqrt(z.pow(2).sum(dim=2) + 1e-8)
        lam = F.softplus(self.raw_lambda)

        return lam * norms.mean(dim=(1, 2))


class RNN_SOC_ICNN(nn.Module):
    def __init__(
        self,
        p_dim: int,
        T_len: int,
        hidden_dim: int,
        branch_channels: int,
        norm_groups: int,
        activation: str = "relu",
        use_quad: bool = False,
        use_norm: bool = False,
    ):
        super().__init__()

        self.backbone = RNNICNNBackbone(
            p_dim=p_dim,
            T_len=T_len,
            hidden_dim=hidden_dim,
            activation=activation,
        )

        self.use_quad = use_quad
        self.use_norm = use_norm

        self.quad = (
            TemporalQuadraticBranch(p_dim=p_dim, out_channels=branch_channels)
            if use_quad else None
        )

        self.norm = (
            TemporalNormBranch(p_dim=p_dim, groups=norm_groups)
            if use_norm else None
        )

    def forward(self, u: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        out = self.backbone(u, r)

        if self.use_quad:
            out = out + self.quad(u, r)

        if self.use_norm:
            out = out + self.norm(u, r)

        return out


def build_models(cfg: Config, device: torch.device) -> Dict[str, nn.Module]:
    specs = {
        "RNN-ReLU": dict(activation="relu", use_quad=False, use_norm=False),
        "RNN-Softplus": dict(activation="softplus", use_quad=False, use_norm=False),
        "RNN-Quad": dict(activation="relu", use_quad=True, use_norm=False),
        "RNN-Norm": dict(activation="relu", use_quad=False, use_norm=True),
        "RNN-SOC": dict(activation="relu", use_quad=True, use_norm=True),
    }

    models = {}
    for name, kwargs in specs.items():
        model = RNN_SOC_ICNN(
            p_dim=cfg.p,
            T_len=cfg.T,
            hidden_dim=cfg.hidden_dim,
            branch_channels=cfg.branch_channels,
            norm_groups=cfg.norm_groups,
            **kwargs,
        ).to(device)

        models[name] = model

    return models


# ============================================================
# Training
# ============================================================

def standardize_values(train_v, val_v, test_v):
    mean = train_v.mean()
    std = train_v.std().clamp_min(1e-8)
    return (train_v - mean) / std, (val_v - mean) / std, (test_v - mean) / std, mean, std


def train_one_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"train": [], "val": []}
    best_state = None
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []

        for ub, rb, vb in train_loader:
            ub = ub.to(device)
            rb = rb.to(device)
            vb = vb.to(device)

            pred = model(ub, rb)
            loss = F.mse_loss(pred, vb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for ub, rb, vb in val_loader:
                ub = ub.to(device)
                rb = rb.to(device)
                vb = vb.to(device)

                pred = model(ub, rb)
                loss = F.mse_loss(pred, vb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(
                f"[{name:13s}] epoch {epoch:03d}/{cfg.epochs} | "
                f"train={train_loss:.4e} | val={val_loss:.4e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def evaluate_prediction_error(
    model: nn.Module,
    loader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()

    mean = mean.to(device)
    std = std.to(device)

    preds = []
    trues = []

    with torch.no_grad():
        for ub, rb, vb_std in loader:
            ub = ub.to(device)
            rb = rb.to(device)
            vb_std = vb_std.to(device)

            pred_std = model(ub, rb)
            pred = pred_std * std + mean
            true = vb_std * std + mean

            preds.append(pred.detach().cpu())
            trues.append(true.detach().cpu())

    pred_all = torch.cat(preds)
    true_all = torch.cat(trues)

    relerr = torch.norm(pred_all - true_all) / torch.norm(true_all)
    return float(relerr.item())


# ============================================================
# Downstream optimization
# ============================================================

def optimize_true_energy(
    r: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    r = r.to(device)
    batch_size = r.shape[0]

    smooth_r = smooth_sequence(r)

    starts = [
        torch.clamp(r, -1.0, 1.0),
        torch.clamp(smooth_r, -1.0, 1.0),
        torch.zeros_like(r),
    ]

    for _ in range(max(cfg.restarts - 3, 0)):
        starts.append(torch.empty_like(r).uniform_(-1.0, 1.0))

    u = torch.cat(starts, dim=0)
    r_rep = r.repeat(len(starts), 1, 1)

    u = u.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([u], lr=cfg.true_opt_lr)

    start_time = time.perf_counter()

    for _ in range(cfg.true_opt_steps):
        loss = true_energy(u, r_rep, cfg).sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            u.clamp_(-1.0, 1.0)

    elapsed = time.perf_counter() - start_time

    with torch.no_grad():
        values = true_energy(u, r_rep, cfg).view(len(starts), batch_size)
        us = u.view(len(starts), batch_size, cfg.T, cfg.p)

        best_idx = values.argmin(dim=0)
        u_best = us[best_idx, torch.arange(batch_size, device=device)]
        v_best = values[best_idx, torch.arange(batch_size, device=device)]

    return u_best.detach(), v_best.detach(), elapsed


def freeze_model(model: nn.Module, frozen: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(not frozen)


def optimize_surrogate(
    model: nn.Module,
    r: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    model.eval()
    freeze_model(model, True)

    r = r.to(device)
    batch_size = r.shape[0]
    smooth_r = smooth_sequence(r)

    starts = [
        torch.clamp(r, -1.0, 1.0),
        torch.clamp(smooth_r, -1.0, 1.0),
        torch.zeros_like(r),
    ]

    for _ in range(max(cfg.restarts - 3, 0)):
        starts.append(torch.empty_like(r).uniform_(-1.0, 1.0))

    u = torch.cat(starts, dim=0)
    r_rep = r.repeat(len(starts), 1, 1)

    u = u.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([u], lr=cfg.surrogate_opt_lr)

    start_time = time.perf_counter()

    for _ in range(cfg.surrogate_opt_steps):
        loss = model(u, r_rep).sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            u.clamp_(-1.0, 1.0)

    elapsed = time.perf_counter() - start_time

    with torch.no_grad():
        values = model(u, r_rep).view(len(starts), batch_size)
        us = u.view(len(starts), batch_size, cfg.T, cfg.p)

        best_idx = values.argmin(dim=0)
        u_best = us[best_idx, torch.arange(batch_size, device=device)]

    freeze_model(model, False)

    return u_best.detach(), elapsed


def downstream_eval(
    model: nn.Module,
    r_test: torch.Tensor,
    u_star: torch.Tensor,
    v_star: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> Dict[str, float]:
    regrets = []
    decision_errors = []
    inference_times = []

    n_test = r_test.shape[0]

    for start in range(0, n_test, cfg.eval_batch_size):
        end = min(start + cfg.eval_batch_size, n_test)

        rb = r_test[start:end].to(device)
        usb = u_star[start:end].to(device)
        vsb = v_star[start:end].to(device)

        u_hat, elapsed = optimize_surrogate(model, rb, cfg, device)
        v_hat = true_energy(u_hat, rb, cfg)

        regret = (v_hat - vsb).detach().cpu()
        decision_error = torch.sqrt((u_hat - usb).pow(2).mean(dim=(1, 2))).detach().cpu()

        regrets.append(regret)
        decision_errors.append(decision_error)
        inference_times.append(elapsed / rb.shape[0])

    regrets = torch.cat(regrets)
    decision_errors = torch.cat(decision_errors)

    return {
        "regret_mean": float(regrets.mean().item()),
        "regret_std": float(regrets.std().item()),
        "decision_error_mean": float(decision_errors.mean().item()),
        "decision_error_std": float(decision_errors.std().item()),
        "infer_ms_mean": float(1000.0 * np.mean(inference_times)),
    }


# ============================================================
# Plotting and output
# ============================================================

def plot_training_curves(histories: Dict[str, Dict[str, List[float]]], out_path: str) -> None:
    plt.figure(figsize=(6.4, 4.0))

    for name, hist in histories.items():
        plt.plot(hist["val"], label=name, linewidth=1.8)

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)

    base, ext = os.path.splitext(out_path)
    if ext.lower() != ".pdf":
        plt.savefig(base + ".pdf", bbox_inches="tight")

    plt.close()


def plot_trajectory_examples(
    models: Dict[str, nn.Module],
    r_test: torch.Tensor,
    u_star: torch.Tensor,
    cfg: Config,
    device: torch.device,
    out_path: str,
) -> None:
    model_names = ["RNN-ReLU", "RNN-Quad", "RNN-Norm", "RNN-SOC"]

    indices = [idx for idx in cfg.plot_indices if idx < r_test.shape[0]]

    if len(indices) < cfg.num_plot_examples:
        fallback = np.linspace(0, r_test.shape[0] - 1, cfg.num_plot_examples, dtype=int).tolist()
        indices = []
        for idx in fallback:
            if idx not in indices:
                indices.append(idx)

    indices = indices[:cfg.num_plot_examples]

    r = r_test[indices].to(device)

    restored_by_model = {}

    for name in model_names:
        if name not in models:
            continue
        u_hat, _ = optimize_surrogate(models[name], r, cfg, device)
        restored_by_model[name] = u_hat.detach().cpu()

    columns = ["Reference", "Ref. opt."] + [
        name for name in model_names if name in restored_by_model
    ]

    n_rows = len(indices)
    n_cols = len(columns)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.9 * n_cols, 1.35 * n_rows),
        squeeze=False,
    )

    time_grid = np.arange(cfg.T)
    d = cfg.plot_dim

    for row_id, idx in enumerate(indices):
        trajectory_map = {
            "Reference": r_test[idx, :, d].cpu().numpy(),
            "Ref. opt.": u_star[idx, :, d].cpu().numpy(),
        }

        for name in restored_by_model:
            trajectory_map[name] = restored_by_model[name][row_id, :, d].cpu().numpy()

        y_min = min(float(np.min(v)) for v in trajectory_map.values()) - 0.10
        y_max = max(float(np.max(v)) for v in trajectory_map.values()) + 0.10
        y_min = max(y_min, -1.15)
        y_max = min(y_max, 1.15)

        for col_id, title in enumerate(columns):
            ax = axes[row_id, col_id]
            ax.plot(time_grid, trajectory_map[title], linewidth=1.8)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([])
            ax.set_yticks([])

            if row_id == 0:
                ax.set_title(title, fontsize=8, pad=3)

            if col_id == 0:
                ax.set_ylabel(f"ex. {row_id + 1}", fontsize=8)

    plt.subplots_adjust(
        left=0.04,
        right=0.99,
        top=0.93,
        bottom=0.04,
        wspace=0.12,
        hspace=0.18,
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    base, ext = os.path.splitext(out_path)
    if ext.lower() != ".pdf":
        plt.savefig(base + ".pdf", bbox_inches="tight")
    if ext.lower() == ".pdf":
        plt.savefig(base + ".png", dpi=300, bbox_inches="tight")

    plt.close()


def write_latex_table(df: pd.DataFrame, out_path: str) -> None:
    order = ["RNN-ReLU", "RNN-Softplus", "RNN-Quad", "RNN-Norm", "RNN-SOC"]
    df = df.set_index("model").loc[order].reset_index()

    best_rel = df["test_relerr"].min()
    best_reg = df["regret_mean"].min()
    best_dec = df["decision_error_mean"].min()

    def bold_if_best(value: float, best: float, digits: int = 4) -> str:
        text = f"{value:.{digits}f}"
        if abs(value - best) <= 1e-12:
            return r"\mathbf{" + text + "}"
        return text

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Recurrent downstream trajectory-smoothing test. "
        r"Regret and decision error are evaluated using the true convex sequence energy "
        r"after optimizing each learned recurrent surrogate. Lower is better for RelErr, regret, and decision error.}"
    )
    lines.append(r"\label{tab:rnn_downstream}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Params & RelErr & Regret & Decision error & Infer. (ms) \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        model = row["model"]
        params = int(row["params"])

        rel = bold_if_best(row["test_relerr"], best_rel)
        reg = bold_if_best(row["regret_mean"], best_reg)
        dec = bold_if_best(row["decision_error_mean"], best_dec)

        line = (
            f"{model} & {params:,} & ${rel}$ & "
            f"${reg}\\pm{row['regret_std']:.4f}$ & "
            f"${dec}\\pm{row['decision_error_std']:.4f}$ & "
            f"${row['infer_ms_mean']:.2f}$ \\\\"
        )
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(out_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg = Config()

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    ensure_dir(cfg.out_dir)

    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(asdict(cfg), file, indent=2)

    print("=" * 80)
    print("SOC-RNN downstream experiment on convex trajectory smoothing")
    print(f"Device: {device}")
    print(f"Output dir: {cfg.out_dir}")
    print("=" * 80)

    # ----------------------------
    # Generate reference sequences
    # ----------------------------
    r_train = generate_reference_sequences(cfg.train_size, cfg, cfg.seed + 10, device).cpu()
    r_val = generate_reference_sequences(cfg.val_size, cfg, cfg.seed + 20, device).cpu()
    r_test = generate_reference_sequences(cfg.test_size, cfg, cfg.seed + 30, device).cpu()

    # ----------------------------
    # Build supervised candidate datasets
    # ----------------------------
    print("Building candidate datasets...")

    train_u, train_r, train_v = build_candidate_dataset(r_train, cfg, device, cfg.seed + 100)
    val_u, val_r, val_v = build_candidate_dataset(r_val, cfg, device, cfg.seed + 200)
    test_u, test_r, test_v = build_candidate_dataset(r_test, cfg, device, cfg.seed + 300)

    train_v_std, val_v_std, test_v_std, energy_mean, energy_std = standardize_values(
        train_v, val_v, test_v
    )

    train_ds = TensorDataset(train_u, train_r, train_v_std)
    val_ds = TensorDataset(val_u, val_r, val_v_std)
    test_ds = TensorDataset(test_u, test_r, test_v_std)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ----------------------------
    # Train models
    # ----------------------------
    models = build_models(cfg, device)
    histories = {}

    for name, model in models.items():
        print("\n" + "=" * 80)
        print(f"Training {name} | params={count_params(model):,}")
        print("=" * 80)

        history = train_one_model(
            name=name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
        )
        histories[name] = history

        relerr = evaluate_prediction_error(model, test_loader, energy_mean, energy_std, device)
        print(f"[{name}] Test RelErr = {relerr:.4f}")

    plot_training_curves(
        histories,
        os.path.join(cfg.out_dir, "training_curves.png"),
    )

    # ----------------------------
    # Compute approximate reference optima
    # ----------------------------
    print("\nComputing reference optima for downstream regret...")

    u_star_list = []
    v_star_list = []
    true_times = []

    for start in range(0, r_test.shape[0], cfg.eval_batch_size):
        end = min(start + cfg.eval_batch_size, r_test.shape[0])

        rb = r_test[start:end]
        u_star_batch, v_star_batch, elapsed = optimize_true_energy(rb, cfg, device)

        u_star_list.append(u_star_batch.detach().cpu())
        v_star_list.append(v_star_batch.detach().cpu())
        true_times.append(elapsed / rb.shape[0])

        print(f"  true opt batch {start:04d}-{end:04d}")

    u_star = torch.cat(u_star_list, dim=0)
    v_star = torch.cat(v_star_list, dim=0)

    print(f"Reference optimizer runtime: {1000.0 * np.mean(true_times):.2f} ms / instance")

    # ----------------------------
    # Downstream evaluation
    # ----------------------------
    print("\nDownstream evaluation...")

    rows = []

    for name, model in models.items():
        relerr = evaluate_prediction_error(model, test_loader, energy_mean, energy_std, device)

        metrics = downstream_eval(
            model=model,
            r_test=r_test,
            u_star=u_star,
            v_star=v_star,
            cfg=cfg,
            device=device,
        )

        row = {
            "model": name,
            "params": count_params(model),
            "test_relerr": relerr,
            **metrics,
        }
        rows.append(row)

        print(
            f"{name:13s} | "
            f"RelErr={relerr:.4f} | "
            f"Regret={metrics['regret_mean']:.4e}±{metrics['regret_std']:.2e} | "
            f"DecErr={metrics['decision_error_mean']:.4f}±{metrics['decision_error_std']:.4f} | "
            f"Infer={metrics['infer_ms_mean']:.2f} ms"
        )

    df = pd.DataFrame(rows)

    summary_csv = os.path.join(cfg.out_dir, "summary.csv")
    summary_tex = os.path.join(cfg.out_dir, "summary_table.tex")

    df.to_csv(summary_csv, index=False)
    write_latex_table(df, summary_tex)

    plot_trajectory_examples(
        models=models,
        r_test=r_test,
        u_star=u_star,
        cfg=cfg,
        device=device,
        out_path=os.path.join(cfg.out_dir, "rnn_downstream_examples.pdf"),
    )

    print("\nSaved:")
    print(f"  {summary_csv}")
    print(f"  {summary_tex}")
    print(f"  {os.path.join(cfg.out_dir, 'rnn_downstream_examples.pdf')}")
    print(f"  {os.path.join(cfg.out_dir, 'rnn_downstream_examples.png')}")
    print(f"  {os.path.join(cfg.out_dir, 'training_curves.png')}")
    print(f"  {os.path.join(cfg.out_dir, 'training_curves.pdf')}")


if __name__ == "__main__":
    main()