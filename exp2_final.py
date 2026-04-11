#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 2 (reference-rule fast GPU version)

Design rule
-----------
This script follows the user's requested parameter-selection rule:

1) For each dimension d in {5, 10, 20, 50}, first define a SMALL anchor SOC-ICNN:
      - ReLU backbone depth = 2
      - hidden width = soc_hidden_map[d]
      - one Quad block
      - one Norm block
      - quad_rank = d
      - norm_dim = d

2) Compute its total parameter count P_soc(d).

3) For the other baselines, keep the SAME hidden width as the SOC anchor and
   gradually increase the ReLU backbone depth until the total parameter count
   becomes >= P_soc(d). The smallest such depth is selected.
   This is done for:
      - ReLU-ICNN
      - Softplus-ICNN
      - Quad-ICNN
      - Norm-ICNN

4) Quad / Norm blocks are limited to at most 2, and default to 1 block in the
   main benchmark to avoid harmful over-parameterization.

5) Training is performed fully on GPU (if available), with no DataLoader-based
   CPU↔GPU toggling in the training loop.

Optional supplement
-------------------
A lightweight P1-ICKAN-adapt-style baseline is kept as an optional low-dimensional
supplementary experiment (default dims 5/10/20 only).

Reported metrics
----------------
For each (dimension, function, model, seed), we report:
    - test MSE
    - test relative L2 error
    - parameter count
    - training time (seconds)
    - best validation loss
    - best epoch
    - status (OK / FAILED_NAN)

This file is designed to be fast, fair, and easy to edit.
"""

import copy
import math
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("high")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Utilities
# =========================
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def sample_inputs(n: int, d: int, low: float, high: float, device: torch.device) -> torch.Tensor:
    return (high - low) * torch.rand(n, d, device=device) + low


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float, float]:
    mse = torch.mean((y_true - y_pred) ** 2).item()
    rel = (torch.norm(y_true - y_pred) / (torch.norm(y_true) + 1e-12)).item()
    return float(mse), float(rel)


# =========================
# Benchmark functions
# =========================
class BenchmarkFunctions:
    @staticmethod
    def quadratic_iso(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(x ** 2, dim=1, keepdim=True)

    @staticmethod
    def quadratic_aniso(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[1]
        w = torch.linspace(0.5, 2.5, d, device=x.device)
        return 0.5 * torch.sum(w * x ** 2, dim=1, keepdim=True)

    @staticmethod
    def norm_euclid(x: torch.Tensor) -> torch.Tensor:
        return torch.norm(x, p=2, dim=1, keepdim=True)

    @staticmethod
    def norm_aniso(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[1]
        w = torch.linspace(1.0, 10.0, d, device=x.device)
        return torch.sqrt(torch.sum(w * x ** 2, dim=1, keepdim=True) + 1e-12)

    @staticmethod
    def mixed_convex(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[1]
        w1 = torch.linspace(0.5, 2.0, d, device=x.device)
        w2 = torch.linspace(2.0, 0.5, d, device=x.device)
        quad = 0.25 * torch.sum(w1 * x ** 2, dim=1, keepdim=True)
        norm = 0.7 * torch.sqrt(torch.sum(w2 * x ** 2, dim=1, keepdim=True) + 1e-12)
        a1 = torch.linspace(-1.0, 1.0, d, device=x.device)
        a2 = torch.cos(torch.linspace(0.0, math.pi, d, device=x.device))
        a3 = torch.sign(torch.linspace(-1.0, 1.0, d, device=x.device) + 1e-8)
        z1 = x @ a1.view(-1, 1) + 0.2
        z2 = x @ a2.view(-1, 1) - 0.1
        z3 = x @ a3.view(-1, 1) + 0.4
        cpwl = torch.maximum(torch.maximum(z1, z2), z3)
        return quad + norm + cpwl

    @staticmethod
    def softplus_sum(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.nn.functional.softplus(x), dim=1, keepdim=True)

    @staticmethod
    def logsumexp_quad(x: torch.Tensor) -> torch.Tensor:
        lse = torch.logsumexp(x, dim=1, keepdim=True)
        quad = 0.1 * torch.sum(x ** 2, dim=1, keepdim=True)
        return lse + quad

    @staticmethod
    def huber_like(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.huber_loss(
            x, torch.zeros_like(x), reduction="none", delta=1.0
        ).sum(dim=1, keepdim=True)

    @staticmethod
    def l1_norm(x: torch.Tensor) -> torch.Tensor:
        return torch.norm(x, p=1, dim=1, keepdim=True)

    @staticmethod
    def ickan_paper_target(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[1]
        w = torch.linspace(0.5, 2.0, d, device=x.device)
        quad = 0.25 * torch.sum(w * x ** 2, dim=1, keepdim=True)
        return torch.sum(torch.abs(x) + torch.abs(1.0 - x), dim=1, keepdim=True) + quad


FUNC_MAP = {
    "QuadraticIso": BenchmarkFunctions.quadratic_iso,
    "QuadraticAniso": BenchmarkFunctions.quadratic_aniso,
    "NormEuclid": BenchmarkFunctions.norm_euclid,
    "NormAniso": BenchmarkFunctions.norm_aniso,
    "Mixed": BenchmarkFunctions.mixed_convex,
    "SoftplusSum": BenchmarkFunctions.softplus_sum,
    "LogSumExpQuad": BenchmarkFunctions.logsumexp_quad,
    "Huber": BenchmarkFunctions.huber_like,
    "L1Norm": BenchmarkFunctions.l1_norm,
    "ICKANPaperTarget": BenchmarkFunctions.ickan_paper_target,
}


# =========================
# Backbone blocks
# =========================
class ConvexBackbone(nn.Module):
    """
    Generic convex backbone with configurable activation:
        z1 = act(W1 x)
        z_l = act(W_l x + U_{l-1} z_{l-1}), U>=0
        out = W_out x + U_out z_last, U_out>=0
    """
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, activation: str = "relu"):
        super().__init__()
        assert depth >= 1
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.activation = activation

        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.W_layers = nn.ModuleList()
        self.U_layers_raw = nn.ParameterList()
        for _ in range(depth - 1):
            self.W_layers.append(nn.Linear(input_dim, hidden_dim))
            self.U_layers_raw.append(nn.Parameter(torch.empty(hidden_dim, hidden_dim)))

        self.W_out = nn.Linear(input_dim, 1)
        self.U_out_raw = nn.Parameter(torch.empty(1, hidden_dim))
        self._reset()

    def _reset(self):
        nn.init.kaiming_uniform_(self.W_in.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W_in.bias)

        for W in self.W_layers:
            nn.init.kaiming_uniform_(W.weight, a=math.sqrt(5))
            nn.init.zeros_(W.bias)

        for U in self.U_layers_raw:
            nn.init.normal_(U, mean=-3.5, std=0.2)

        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.zeros_(self.W_out.bias)
        nn.init.normal_(self.U_out_raw, mean=-3.5, std=0.2)

    def act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return torch.relu(x)
        if self.activation == "softplus":
            return torch.nn.functional.softplus(x)
        raise ValueError(self.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.act(self.W_in(x))
        for W, U_raw in zip(self.W_layers, self.U_layers_raw):
            U = torch.nn.functional.softplus(U_raw)
            z = self.act(W(x) + z @ U.T)
        U_out = torch.nn.functional.softplus(self.U_out_raw)
        return self.W_out(x) + z @ U_out.T


class ReLUICNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        self.backbone = ConvexBackbone(input_dim, hidden_dim, depth, activation="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SoftplusICNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        self.backbone = ConvexBackbone(input_dim, hidden_dim, depth, activation="softplus")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class QuadICNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, quad_rank: int, num_quad_blocks: int = 1):
        super().__init__()
        self.backbone = ConvexBackbone(input_dim, hidden_dim, depth, activation="relu")
        self.num_quad_blocks = num_quad_blocks
        self.Ls = nn.Parameter(torch.randn(num_quad_blocks, input_dim, quad_rank) * 0.01)
        self.alpha_raw = nn.Parameter(torch.zeros(num_quad_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        quad_terms = []
        alpha = torch.nn.functional.softplus(self.alpha_raw) + 1e-8
        for h in range(self.num_quad_blocks):
            q = x @ self.Ls[h]
            quad_terms.append(0.5 * alpha[h] * torch.sum(q * q, dim=1, keepdim=True))
        return out + torch.stack(quad_terms, dim=0).sum(dim=0)


class NormICNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, norm_dim: int, num_norm_blocks: int = 1):
        super().__init__()
        self.backbone = ConvexBackbone(input_dim, hidden_dim, depth, activation="relu")
        self.num_norm_blocks = num_norm_blocks
        self.A = nn.Parameter(torch.randn(num_norm_blocks, norm_dim, input_dim) * 0.01)
        self.d = nn.Parameter(torch.zeros(num_norm_blocks, norm_dim))
        self.lam_raw = nn.Parameter(torch.zeros(num_norm_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        lam = torch.nn.functional.softplus(self.lam_raw) + 1e-8
        norm_terms = []
        for g in range(self.num_norm_blocks):
            u = x @ self.A[g].T + self.d[g]
            norm_terms.append(lam[g] * torch.norm(u, p=2, dim=1, keepdim=True))
        return out + torch.stack(norm_terms, dim=0).sum(dim=0)


class SOCICNN(nn.Module):
    """
    Anchor structured model:
        2-layer ReLU backbone + one Quad block + one Norm block
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        quad_rank: int,
        norm_dim: int,
        num_quad_blocks: int = 1,
        num_norm_blocks: int = 1,
    ):
        super().__init__()
        self.backbone = ConvexBackbone(input_dim, hidden_dim, depth, activation="relu")

        self.num_quad_blocks = num_quad_blocks
        self.Ls = nn.Parameter(torch.randn(num_quad_blocks, input_dim, quad_rank) * 0.01)
        self.alpha_raw = nn.Parameter(torch.zeros(num_quad_blocks))

        self.num_norm_blocks = num_norm_blocks
        self.A = nn.Parameter(torch.randn(num_norm_blocks, norm_dim, input_dim) * 0.01)
        self.d = nn.Parameter(torch.zeros(num_norm_blocks, norm_dim))
        self.lam_raw = nn.Parameter(torch.zeros(num_norm_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)

        alpha = torch.nn.functional.softplus(self.alpha_raw) + 1e-8
        quad_terms = []
        for h in range(self.num_quad_blocks):
            q = x @ self.Ls[h]
            quad_terms.append(0.5 * alpha[h] * torch.sum(q * q, dim=1, keepdim=True))
        out_quad = torch.stack(quad_terms, dim=0).sum(dim=0)

        lam = torch.nn.functional.softplus(self.lam_raw) + 1e-8
        norm_terms = []
        for g in range(self.num_norm_blocks):
            u = x @ self.A[g].T + self.d[g]
            norm_terms.append(lam[g] * torch.norm(u, p=2, dim=1, keepdim=True))
        out_norm = torch.stack(norm_terms, dim=0).sum(dim=0)

        return out + out_quad + out_norm


# =========================
# Optional ICKAN-style supplement
# =========================
class P1ICKANAdaptLite(nn.Module):
    """
    Lightweight adaptive convex P1-ICKAN-inspired model.
    Kept for low-dimensional supplementary comparison only.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_segments: int = 8, domain_low: float = -2.0, domain_high: float = 2.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_segments = num_segments
        self.domain_low = float(domain_low)
        self.domain_high = float(domain_high)

        K = num_segments
        self.base = nn.Parameter(torch.zeros(hidden_dim, input_dim))
        self.slope0 = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.delta_raw = nn.Parameter(torch.randn(hidden_dim, input_dim, K - 1) * 0.1)
        self.length_raw = nn.Parameter(torch.randn(hidden_dim, input_dim, K) * 0.1)

        self.out_weight = nn.Parameter(torch.ones(hidden_dim))
        self.lin = nn.Linear(input_dim, 1)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def _interior_knots(self) -> torch.Tensor:
        lengths = torch.nn.functional.softplus(self.length_raw) + 1e-4
        lengths = lengths / lengths.sum(dim=-1, keepdim=True)
        total = self.domain_high - self.domain_low
        lengths = total * lengths
        knots = self.domain_low + torch.cumsum(lengths, dim=-1)
        return knots[..., :-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_exp = x[:, None, :, None]
        t_interior = self._interior_knots()[None, :, :, :]
        delta = torch.nn.functional.softplus(self.delta_raw)[None, :, :, :]
        slope0 = self.slope0[None, :, :, None]
        base = self.base[None, :, :, None]

        pwl = base + slope0 * (x_exp - self.domain_low) + torch.sum(
            delta * torch.relu(x_exp - t_interior), dim=-1, keepdim=True
        )
        hidden = pwl.squeeze(-1).sum(dim=2)
        out = self.lin(x) + hidden @ torch.abs(self.out_weight).unsqueeze(1)
        return out


# =========================
# Config
# =========================
@dataclass
class Exp2Config:
    # dimensions / data
    dims: Tuple[int, ...] = (5, 10, 20, 50)
    seeds: Tuple[int, ...] = (2026, 2027, 2028)
    n_train: int = 6000
    n_val: int = 1000
    n_test: int = 2000
    x_low: float = -3.0
    x_high: float = 3.0

    # training
    batch_size: int = 4096
    max_epochs: int = 120
    patience: int = 8
    eval_every: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-6
    clip_grad_norm: float = 1.0
    use_amp: bool = True

    # anchor SOC settings
    soc_hidden_map: Dict[int, int] = None
    soc_depth: int = 2
    soc_num_quad_blocks: int = 1
    soc_num_norm_blocks: int = 1

    # one-sided search for baselines:
    # keep hidden same as SOC anchor, gradually add ReLU layers until params >= SOC
    baseline_depth_search: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)

    # optional ickan supplement
    run_ickan_supplement: bool = True
    ickan_dims: Tuple[int, ...] = (5, 10, 20)
    ickan_hidden_search: Tuple[int, ...] = (8, 12, 16, 24, 32)
    ickan_segment_search: Tuple[int, ...] = (8, 12)
    ickan_batch_size: int = 2048
    ickan_max_epochs: int = 60
    ickan_patience: int = 6
    ickan_eval_every: int = 5
    ickan_lr: float = 1e-3
    ickan_weight_decay: float = 1e-6

    # outputs
    detail_csv: str = "exp2_ref_rule_detail.csv"
    summary_csv: str = "exp2_ref_rule_summary.csv"
    ickan_detail_csv: str = "exp2_ref_rule_ickan_detail.csv"
    ickan_summary_csv: str = "exp2_ref_rule_ickan_summary.csv"


# =========================
# Model builders / matching
# =========================
def get_soc_hidden(d: int, cfg: Exp2Config) -> int:
    if cfg.soc_hidden_map is None:
        default_map = {5: 16, 10: 20, 20: 24, 50: 32}
        return default_map[d]
    return cfg.soc_hidden_map[d]


def build_model(model_name: str, d: int, hidden: int, depth: int, cfg: Exp2Config, *, num_segments: Optional[int] = None) -> nn.Module:
    if model_name == "ReLU-ICNN":
        return ReLUICNN(d, hidden, depth=depth)
    if model_name == "Softplus-ICNN":
        return SoftplusICNN(d, hidden, depth=depth)
    if model_name == "Quad-ICNN":
        return QuadICNN(d, hidden, depth=depth, quad_rank=d, num_quad_blocks=1)
    if model_name == "Norm-ICNN":
        return NormICNN(d, hidden, depth=depth, norm_dim=d, num_norm_blocks=1)
    if model_name == "SOC-ICNN":
        return SOCICNN(
            d, hidden, depth=depth,
            quad_rank=d, norm_dim=d,
            num_quad_blocks=min(cfg.soc_num_quad_blocks, 2),
            num_norm_blocks=min(cfg.soc_num_norm_blocks, 2),
        )
    if model_name == "P1-ICKAN-adapt":
        assert num_segments is not None
        return P1ICKANAdaptLite(d, hidden, num_segments=num_segments, domain_low=cfg.x_low, domain_high=cfg.x_high)
    raise ValueError(model_name)


def build_anchor_soc_info(d: int, cfg: Exp2Config) -> Dict[str, int]:
    h = get_soc_hidden(d, cfg)
    model = build_model("SOC-ICNN", d=d, hidden=h, depth=cfg.soc_depth, cfg=cfg)
    return {"hidden": h, "depth": cfg.soc_depth, "params": count_params(model)}


def find_min_depth_ge_budget(model_name: str, d: int, target_params: int, cfg: Exp2Config) -> Dict[str, int]:
    """
    Use SAME hidden as SOC anchor, gradually increase depth until params >= target_params.
    If no candidate exceeds target, return the largest depth candidate.
    """
    h = get_soc_hidden(d, cfg)
    feasible = []
    all_candidates = []
    for depth in cfg.baseline_depth_search:
        model = build_model(model_name, d=d, hidden=h, depth=depth, cfg=cfg)
        p = count_params(model)
        all_candidates.append((p, depth))
        if p >= target_params:
            feasible.append((p, depth))

    if len(feasible) > 0:
        p, depth = min(feasible, key=lambda t: (t[0], t[1]))
    else:
        p, depth = max(all_candidates, key=lambda t: (t[0], t[1]))

    return {"hidden": h, "depth": depth, "params": p}


def find_ickan_nearest_budget(d: int, target_params: int, cfg: Exp2Config) -> Dict[str, int]:
    best = None
    for h in cfg.ickan_hidden_search:
        for seg in cfg.ickan_segment_search:
            model = build_model("P1-ICKAN-adapt", d=d, hidden=h, depth=1, cfg=cfg, num_segments=seg)
            p = count_params(model)
            cand = (abs(p - target_params), p, h, seg)
            if best is None or cand < best:
                best = cand
    _, p, h, seg = best
    return {"hidden": h, "segments": seg, "params": p}


# =========================
# Training core (GPU-only batching)
# =========================
def train_one_model_fast(
    model: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    *,
    batch_size: int,
    max_epochs: int,
    patience: int,
    eval_every: int,
    lr: float,
    weight_decay: float,
    clip_grad_norm: float,
    use_amp: bool,
) -> Dict[str, float]:
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and DEVICE.type == "cuda"))

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    bad_checks = 0

    n = X_train.size(0)
    bs = min(batch_size, n)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, device=X_train.device)

        bad_train = False
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            bx = X_train[idx]
            by = Y_train[idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and DEVICE.type == "cuda")):
                pred = model(bx)
                loss = criterion(pred, by)

            if not torch.isfinite(loss):
                bad_train = True
                break

            scaler.scale(loss).backward()
            if clip_grad_norm is not None and clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

        if bad_train:
            break

        if ((epoch % eval_every) != 0) and (epoch != max_epochs):
            continue

        model.eval()
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=(use_amp and DEVICE.type == "cuda")):
                val_pred = model(X_val)
                val_loss = criterion(val_pred, Y_val).item()

        if not np.isfinite(val_loss):
            break

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_checks = 0
        else:
            bad_checks += 1
            if bad_checks >= patience:
                break

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    if best_state is None:
        return {
            "status": "FAILED_NAN",
            "best_val_loss": float("nan"),
            "best_epoch": -1,
            "train_time_sec": t1 - t0,
            "test_mse": float("nan"),
            "test_relerr": float("nan"),
        }

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=(use_amp and DEVICE.type == "cuda")):
            pred_test = model(X_val.new_tensor(X_test_global_ref)) if False else None  # dummy line kept invalid path unused

    return {
        "status": "OK",
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "train_time_sec": t1 - t0,
    }


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    with torch.cuda.amp.autocast(enabled=(use_amp and DEVICE.type == "cuda")):
        pred = model(X_test)
    mse, rel = compute_metrics(Y_test, pred)
    return {"test_mse": mse, "test_relerr": rel}


# =========================
# Dataset cache
# =========================
def make_dataset_cache(cfg: Exp2Config, d: int, seed: int) -> Dict[str, Dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train = sample_inputs(cfg.n_train, d, cfg.x_low, cfg.x_high, DEVICE)
    X_val = sample_inputs(cfg.n_val, d, cfg.x_low, cfg.x_high, DEVICE)
    X_test = sample_inputs(cfg.n_test, d, cfg.x_low, cfg.x_high, DEVICE)

    cache = {"X": {"train": X_train, "val": X_val, "test": X_test}, "Y": {}}
    for func_name, func_fn in FUNC_MAP.items():
        cache["Y"][func_name] = {
            "train": func_fn(X_train),
            "val": func_fn(X_val),
            "test": func_fn(X_test),
        }
    return cache


# =========================
# Trial runners
# =========================
def run_single_trial_main(
    cfg: Exp2Config,
    d: int,
    seed: int,
    func_name: str,
    dataset_cache: Dict[str, Dict[str, torch.Tensor]],
    arch_table: Dict[int, Dict[str, Dict[str, int]]],
) -> List[Dict[str, object]]:
    X_train = dataset_cache["X"]["train"]
    X_val = dataset_cache["X"]["val"]
    X_test = dataset_cache["X"]["test"]
    Y_train = dataset_cache["Y"][func_name]["train"]
    Y_val = dataset_cache["Y"][func_name]["val"]
    Y_test = dataset_cache["Y"][func_name]["test"]

    rows = []
    model_order = ("ReLU-ICNN", "Softplus-ICNN", "Quad-ICNN", "Norm-ICNN", "SOC-ICNN")
    for model_name in model_order:
        info = arch_table[d][model_name]
        model = build_model(model_name, d=d, hidden=info["hidden"], depth=info["depth"], cfg=cfg)
        pcount = count_params(model)

        stats = train_one_model_fast(
            model=model,
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            batch_size=cfg.batch_size,
            max_epochs=cfg.max_epochs,
            patience=cfg.patience,
            eval_every=cfg.eval_every,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            clip_grad_norm=cfg.clip_grad_norm,
            use_amp=cfg.use_amp,
        )

        if stats["status"] == "OK":
            eval_stats = evaluate_model(model, X_test, Y_test, cfg.use_amp)
        else:
            eval_stats = {"test_mse": float("nan"), "test_relerr": float("nan")}

        rows.append({
            "benchmark": "main",
            "dim": d,
            "seed": seed,
            "function": func_name,
            "model": model_name,
            "status": stats["status"],
            "num_params": pcount,
            "hidden_or_width": info["hidden"],
            "depth": info["depth"],
            "num_segments": float("nan"),
            "best_val_loss": stats["best_val_loss"],
            "best_epoch": stats["best_epoch"],
            "train_time_sec": stats["train_time_sec"],
            "test_mse": eval_stats["test_mse"],
            "test_relerr": eval_stats["test_relerr"],
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rows


def run_single_trial_ickan(
    cfg: Exp2Config,
    d: int,
    seed: int,
    func_name: str,
    dataset_cache: Dict[str, Dict[str, torch.Tensor]],
    arch_table_ickan: Dict[int, Dict[str, Dict[str, int]]],
) -> List[Dict[str, object]]:
    X_train = dataset_cache["X"]["train"]
    X_val = dataset_cache["X"]["val"]
    X_test = dataset_cache["X"]["test"]
    Y_train = dataset_cache["Y"][func_name]["train"]
    Y_val = dataset_cache["Y"][func_name]["val"]
    Y_test = dataset_cache["Y"][func_name]["test"]

    rows = []
    model_order = ("ReLU-ICNN", "SOC-ICNN", "P1-ICKAN-adapt")
    for model_name in model_order:
        info = arch_table_ickan[d][model_name]
        if model_name == "P1-ICKAN-adapt":
            model = build_model(model_name, d=d, hidden=info["hidden"], depth=1, cfg=cfg, num_segments=info["segments"])
            pcount = count_params(model)
            stats = train_one_model_fast(
                model=model,
                X_train=X_train,
                Y_train=Y_train,
                X_val=X_val,
                Y_val=Y_val,
                batch_size=cfg.ickan_batch_size,
                max_epochs=cfg.ickan_max_epochs,
                patience=cfg.ickan_patience,
                eval_every=cfg.ickan_eval_every,
                lr=cfg.ickan_lr,
                weight_decay=cfg.ickan_weight_decay,
                clip_grad_norm=cfg.clip_grad_norm,
                use_amp=cfg.use_amp,
            )
        else:
            model = build_model(model_name, d=d, hidden=info["hidden"], depth=info["depth"], cfg=cfg)
            pcount = count_params(model)
            stats = train_one_model_fast(
                model=model,
                X_train=X_train,
                Y_train=Y_train,
                X_val=X_val,
                Y_val=Y_val,
                batch_size=cfg.batch_size,
                max_epochs=cfg.max_epochs,
                patience=cfg.patience,
                eval_every=cfg.eval_every,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                clip_grad_norm=cfg.clip_grad_norm,
                use_amp=cfg.use_amp,
            )

        if stats["status"] == "OK":
            eval_stats = evaluate_model(model, X_test, Y_test, cfg.use_amp)
        else:
            eval_stats = {"test_mse": float("nan"), "test_relerr": float("nan")}

        rows.append({
            "benchmark": "ickan_supplement",
            "dim": d,
            "seed": seed,
            "function": func_name,
            "model": model_name,
            "status": stats["status"],
            "num_params": pcount,
            "hidden_or_width": info["hidden"],
            "depth": info.get("depth", float("nan")),
            "num_segments": info.get("segments", float("nan")),
            "best_val_loss": stats["best_val_loss"],
            "best_epoch": stats["best_epoch"],
            "train_time_sec": stats["train_time_sec"],
            "test_mse": eval_stats["test_mse"],
            "test_relerr": eval_stats["test_relerr"],
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rows


# =========================
# Pretty printing
# =========================
def print_budget_table_main(cfg: Exp2Config, arch_table: Dict[int, Dict[str, Dict[str, int]]]):
    print("\n" + "=" * 140)
    print("Main benchmark | reference-rule parameter selection")
    print("=" * 140)
    print("Rule: first fix small SOC, then increase other models' ReLU backbone depth until params >= SOC")
    for d in cfg.dims:
        print(f"d = {d}")
        for model_name in ("SOC-ICNN", "ReLU-ICNN", "Softplus-ICNN", "Quad-ICNN", "Norm-ICNN"):
            info = arch_table[d][model_name]
            print(f"  {model_name:<14s}: hidden={info['hidden']:<3d}, depth={info['depth']:<2d}, params={info['params']}")
        print("-" * 140)


def print_budget_table_ickan(cfg: Exp2Config, arch_table_ickan: Dict[int, Dict[str, Dict[str, int]]]):
    print("\n" + "=" * 140)
    print("ICKAN supplement | low-dimensional comparison")
    print("=" * 140)
    for d in cfg.ickan_dims:
        print(f"d = {d}")
        for model_name in ("ReLU-ICNN", "SOC-ICNN", "P1-ICKAN-adapt"):
            info = arch_table_ickan[d][model_name]
            if model_name == "P1-ICKAN-adapt":
                print(f"  {model_name:<14s}: hidden={info['hidden']:<3d}, segments={info['segments']:<2d}, params={info['params']}")
            else:
                print(f"  {model_name:<14s}: hidden={info['hidden']:<3d}, depth={info['depth']:<2d}, params={info['params']}")
        print("-" * 140)


def print_group_summary(df_group: pd.DataFrame, title: str, model_order: Tuple[str, ...]):
    print("\n" + "=" * 140)
    print(title)
    print("=" * 140)
    header = f"{'Model':<16} | {'RelErr(mean±std)':>24} | {'MSE(mean±std)':>24} | {'Time(s)':>12} | {'Params':>10} | {'OKRate':>8}"
    print(header)
    print("-" * len(header))

    for model_name in model_order:
        sub = df_group[df_group["model"] == model_name]
        if len(sub) == 0:
            continue
        rel = pd.to_numeric(sub["test_relerr"], errors="coerce")
        mse = pd.to_numeric(sub["test_mse"], errors="coerce")
        tsec = pd.to_numeric(sub["train_time_sec"], errors="coerce")
        pnum = pd.to_numeric(sub["num_params"], errors="coerce")
        ok_rate = (sub["status"] == "OK").mean()

        print(
            f"{model_name:<16} | "
            f"{rel.mean():10.3e} ± {rel.std():8.3e} | "
            f"{mse.mean():10.3e} ± {mse.std():8.3e} | "
            f"{tsec.mean():12.3f} | "
            f"{pnum.mean():10.1f} | "
            f"{ok_rate:8.3f}"
        )


# =========================
# Main
# =========================
def main():
    # ============================================================
    # Editable experiment parameters
    # ============================================================
    cfg = Exp2Config(
        dims=(5, 10, 20, 50),
        seeds=(2026, 2027, 2028),

        n_train=6000,
        n_val=1000,
        n_test=2000,
        x_low=-3.0,
        x_high=3.0,

        batch_size=4096,
        max_epochs=120,
        patience=8,
        eval_every=5,
        lr=1e-3,
        weight_decay=1e-6,
        clip_grad_norm=1.0,
        use_amp=True,

        soc_hidden_map={5: 16, 10: 20, 20: 24, 50: 32},
        soc_depth=2,
        soc_num_quad_blocks=1,
        soc_num_norm_blocks=1,

        baseline_depth_search=(2, 3, 4, 5, 6, 7, 8),

        run_ickan_supplement=True,
        ickan_dims=(5, 10, 20),
        ickan_hidden_search=(8, 12, 16, 24, 32),
        ickan_segment_search=(8, 12),
        ickan_batch_size=2048,
        ickan_max_epochs=60,
        ickan_patience=6,
        ickan_eval_every=5,
        ickan_lr=1e-3,
        ickan_weight_decay=1e-6,

        detail_csv="exp2_ref_rule_detail.csv",
        summary_csv="exp2_ref_rule_summary.csv",
        ickan_detail_csv="exp2_ref_rule_ickan_detail.csv",
        ickan_summary_csv="exp2_ref_rule_ickan_summary.csv",
    )
    # ============================================================

    print("=" * 140)
    print("Experiment 2 | reference-rule fair and fast GPU benchmark")
    print("=" * 140)
    print(f"Device: {DEVICE}")
    print(f"Dims: {cfg.dims}")
    print(f"Seeds: {cfg.seeds}")
    print(f"Functions: {list(FUNC_MAP.keys())}")
    print(f"Train/Val/Test: {cfg.n_train}/{cfg.n_val}/{cfg.n_test}")
    print(f"SOC hidden map: {cfg.soc_hidden_map}")

    # ----- build main arch table -----
    arch_table_main: Dict[int, Dict[str, Dict[str, int]]] = {}
    for d in cfg.dims:
        arch_table_main[d] = {}
        soc_info = build_anchor_soc_info(d, cfg)
        arch_table_main[d]["SOC-ICNN"] = soc_info

        target_params = soc_info["params"]
        arch_table_main[d]["ReLU-ICNN"] = find_min_depth_ge_budget("ReLU-ICNN", d, target_params, cfg)
        arch_table_main[d]["Softplus-ICNN"] = find_min_depth_ge_budget("Softplus-ICNN", d, target_params, cfg)
        arch_table_main[d]["Quad-ICNN"] = find_min_depth_ge_budget("Quad-ICNN", d, target_params, cfg)
        arch_table_main[d]["Norm-ICNN"] = find_min_depth_ge_budget("Norm-ICNN", d, target_params, cfg)

    print_budget_table_main(cfg, arch_table_main)

    # ----- cache datasets once per (dim, seed) -----
    dataset_cache_all: Dict[Tuple[int, int], Dict[str, Dict[str, torch.Tensor]]] = {}
    for d in cfg.dims:
        for seed in cfg.seeds:
            dataset_cache_all[(d, seed)] = make_dataset_cache(cfg, d, seed)

    # ----- main benchmark -----
    all_rows_main: List[Dict[str, object]] = []
    for d in cfg.dims:
        for func_name in FUNC_MAP.keys():
            print(f"\n[Main] dim={d}, function={func_name}")
            for seed in cfg.seeds:
                rows = run_single_trial_main(
                    cfg=cfg,
                    d=d,
                    seed=seed,
                    func_name=func_name,
                    dataset_cache=dataset_cache_all[(d, seed)],
                    arch_table=arch_table_main,
                )
                all_rows_main.extend(rows)

            df_tmp = pd.DataFrame([r for r in all_rows_main if r["dim"] == d and r["function"] == func_name])
            print_group_summary(
                df_tmp,
                title=f"Main summary | dim={d}, function={func_name}",
                model_order=("ReLU-ICNN", "Softplus-ICNN", "Quad-ICNN", "Norm-ICNN", "SOC-ICNN"),
            )

    df_main = pd.DataFrame(all_rows_main)
    df_main.to_csv(cfg.detail_csv, index=False)

    summary_main = (
        df_main.groupby(["dim", "function", "model"])
        .agg(
            OKRate=("status", lambda s: float(np.mean(np.array(s) == "OK"))),
            RelErr_Mean=("test_relerr", "mean"),
            RelErr_Std=("test_relerr", "std"),
            MSE_Mean=("test_mse", "mean"),
            MSE_Std=("test_mse", "std"),
            TrainTime_Mean=("train_time_sec", "mean"),
            TrainTime_Std=("train_time_sec", "std"),
            Params_Mean=("num_params", "mean"),
            Params_Std=("num_params", "std"),
            BestVal_Mean=("best_val_loss", "mean"),
            BestVal_Std=("best_val_loss", "std"),
            BestEpoch_Mean=("best_epoch", "mean"),
            BestEpoch_Std=("best_epoch", "std"),
        )
        .reset_index()
    )
    summary_main.to_csv(cfg.summary_csv, index=False)

    print("\n" + "=" * 140)
    print("Saved main benchmark files")
    print("=" * 140)
    print(cfg.detail_csv)
    print(cfg.summary_csv)

    # ----- ICKAN supplement -----
    if cfg.run_ickan_supplement:
        arch_table_ickan: Dict[int, Dict[str, Dict[str, int]]] = {}
        for d in cfg.ickan_dims:
            arch_table_ickan[d] = {}
            target_params = arch_table_main[d]["SOC-ICNN"]["params"]
            arch_table_ickan[d]["ReLU-ICNN"] = arch_table_main[d]["ReLU-ICNN"]
            arch_table_ickan[d]["SOC-ICNN"] = arch_table_main[d]["SOC-ICNN"]
            arch_table_ickan[d]["P1-ICKAN-adapt"] = find_ickan_nearest_budget(d, target_params, cfg)

        print_budget_table_ickan(cfg, arch_table_ickan)

        all_rows_ickan: List[Dict[str, object]] = []
        for d in cfg.ickan_dims:
            for func_name in FUNC_MAP.keys():
                print(f"\n[ICKAN supplement] dim={d}, function={func_name}")
                for seed in cfg.seeds:
                    rows = run_single_trial_ickan(
                        cfg=cfg,
                        d=d,
                        seed=seed,
                        func_name=func_name,
                        dataset_cache=dataset_cache_all[(d, seed)],
                        arch_table_ickan=arch_table_ickan,
                    )
                    all_rows_ickan.extend(rows)

                df_tmp = pd.DataFrame([r for r in all_rows_ickan if r["dim"] == d and r["function"] == func_name])
                print_group_summary(
                    df_tmp,
                    title=f"ICKAN supplement summary | dim={d}, function={func_name}",
                    model_order=("ReLU-ICNN", "SOC-ICNN", "P1-ICKAN-adapt"),
                )

        df_ickan = pd.DataFrame(all_rows_ickan)
        df_ickan.to_csv(cfg.ickan_detail_csv, index=False)

        summary_ickan = (
            df_ickan.groupby(["dim", "function", "model"])
            .agg(
                OKRate=("status", lambda s: float(np.mean(np.array(s) == "OK"))),
                RelErr_Mean=("test_relerr", "mean"),
                RelErr_Std=("test_relerr", "std"),
                MSE_Mean=("test_mse", "mean"),
                MSE_Std=("test_mse", "std"),
                TrainTime_Mean=("train_time_sec", "mean"),
                TrainTime_Std=("train_time_sec", "std"),
                Params_Mean=("num_params", "mean"),
                Params_Std=("num_params", "std"),
                BestVal_Mean=("best_val_loss", "mean"),
                BestVal_Std=("best_val_loss", "std"),
                BestEpoch_Mean=("best_epoch", "mean"),
                BestEpoch_Std=("best_epoch", "std"),
            )
            .reset_index()
        )
        summary_ickan.to_csv(cfg.ickan_summary_csv, index=False)

        print("\n" + "=" * 140)
        print("Saved ICKAN supplement files")
        print("=" * 140)
        print(cfg.ickan_detail_csv)
        print(cfg.ickan_summary_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()
