# -*- coding: utf-8 -*-
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Console / encoding
# ============================================================


def setup_console_encoding():
    try:
        if os.name == "nt":
            try:
                import ctypes
                ctypes.windll.kernel32.SetConsoleOutputCP(65001)
                ctypes.windll.kernel32.SetConsoleCP(65001)
            except Exception:
                pass
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


setup_console_encoding()


# ============================================================
# Basic utils
# ============================================================


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_npz_dict(path: str) -> Dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def normalize_scalar(x):
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return x.item()
        if x.size == 1:
            return x.reshape(-1)[0].item() if np.issubdtype(x.dtype, np.number) else str(x.reshape(-1)[0])
        return x
    return x


def get_task_name(data: Dict) -> str:
    return str(normalize_scalar(data["task_name"]))


def get_task_family(data: Dict) -> str:
    return str(normalize_scalar(data["task_family"]))


def get_feasible_type(data: Dict) -> str:
    return str(normalize_scalar(data["feasible_type"]))


def get_budget(data: Dict) -> Optional[float]:
    val = float(normalize_scalar(data["budget"]))
    return None if val < 0 else val


def stable_softplus_np(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def stable_logsumexp_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    xmax = np.max(x, axis=axis, keepdims=True)
    out = xmax + np.log(np.sum(np.exp(x - xmax), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def huber_np(x: np.ndarray, delta: float) -> np.ndarray:
    abs_x = np.abs(x)
    quad = abs_x <= delta
    return np.where(quad, x * x, 2.0 * delta * abs_x - delta * delta)


# ============================================================
# Data split / flatten
# ============================================================

INSTANCE_KEYS_BASE = {
    "contexts",
    "x_star",
    "f_star",
    "cand_X",
    "cand_y",
    "m_arr",
    "c_arr",
    "statuses",
    "solvers",
    "lambda_arr",
    "beta_arr",
    "shift_arr",
}

SPECIAL_PREFIXES = ["d_arr_", "shift_arr_"]


def is_instance_key(key: str) -> bool:
    return key in INSTANCE_KEYS_BASE or any(key.startswith(prefix) for prefix in SPECIAL_PREFIXES)


def split_instance_dict(data: Dict, val_ratio: float = 0.15, seed: int = 0) -> Tuple[Dict, Dict]:
    n = data["contexts"].shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_val = max(1, int(round(n * val_ratio)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    def slice_by_index(d: Dict, indices: np.ndarray) -> Dict:
        out = {}
        for k, v in d.items():
            out[k] = v[indices] if is_instance_key(k) else v
        return out

    return slice_by_index(data, train_idx), slice_by_index(data, val_idx)


def subset_instance_dict(data: Dict, max_instances: Optional[int], seed: int = 0) -> Dict:
    if max_instances is None:
        return data
    n = data["contexts"].shape[0]
    if max_instances >= n:
        return data

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    idx = idx[:max_instances]

    out = {}
    for k, v in data.items():
        out[k] = v[idx] if is_instance_key(k) else v
    return out


def flatten_supervised_arrays(data: Dict) -> Dict[str, np.ndarray]:
    contexts = data["contexts"]
    cand_X = data["cand_X"]
    cand_y = data["cand_y"]
    N, M, d = cand_X.shape
    theta = np.repeat(contexts, M, axis=0)
    x = cand_X.reshape(N * M, d)
    y = cand_y.reshape(N * M, 1)
    return {"theta": theta, "x": x, "y": y}


def normalize_theta_arrays(
    arr_train: Dict[str, np.ndarray],
    arr_val: Dict[str, np.ndarray],
    arr_test: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    theta_mean = arr_train["theta"].mean(axis=0, keepdims=True)
    theta_std = arr_train["theta"].std(axis=0, keepdims=True)
    theta_std = np.maximum(theta_std, 1e-8)

    def _apply(arr: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = {
            "theta": (arr["theta"] - theta_mean) / theta_std,
            "x": arr["x"].copy(),
            "y": arr["y"].copy(),
        }
        return out

    return _apply(arr_train), _apply(arr_val), _apply(arr_test), theta_mean, theta_std


class Exp3FlatDataset(Dataset):
    def __init__(self, arrays: Dict[str, np.ndarray], y_mean=None, y_std=None):
        self.theta = torch.tensor(arrays["theta"], dtype=torch.float32)
        self.x = torch.tensor(arrays["x"], dtype=torch.float32)
        self.y_raw = torch.tensor(arrays["y"], dtype=torch.float32).view(-1, 1)

        if y_mean is None:
            y_mean = float(self.y_raw.mean().item())
        if y_std is None:
            y_std = float(self.y_raw.std(unbiased=False).item())

        self.y_mean = y_mean
        self.y_std = max(y_std, 1e-8)
        self.y = (self.y_raw - self.y_mean) / self.y_std

    def __len__(self):
        return self.theta.shape[0]

    def __getitem__(self, idx):
        return {
            "theta": self.theta[idx],
            "x": self.x[idx],
            "y": self.y[idx],
            "y_raw": self.y_raw[idx],
        }


# ============================================================
# Models
# ============================================================


class PositiveLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight_raw = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight_raw)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        weight = F.softplus(self.weight_raw) + 1e-8
        return F.linear(x, weight, self.bias)


class ICNNBackbone(nn.Module):
    def __init__(
        self,
        x_dim: int,
        theta_dim: int,
        hidden_dims: List[int],
        activation_mode: str = "relu_all",
    ):
        super().__init__()
        assert activation_mode in ["relu_all", "relu_last_softplus"]
        self.hidden_dims = hidden_dims
        self.activation_mode = activation_mode

        self.wx_layers = nn.ModuleList()
        self.wt_layers = nn.ModuleList()
        self.uz_layers = nn.ModuleList()

        prev_h = None
        for h in hidden_dims:
            self.wx_layers.append(nn.Linear(x_dim, h, bias=False))
            self.wt_layers.append(nn.Linear(theta_dim, h, bias=True))
            if prev_h is not None:
                self.uz_layers.append(PositiveLinear(prev_h, h, bias=False))
            prev_h = h

        self.out_x = nn.Linear(x_dim, 1, bias=False)
        self.out_t = nn.Linear(theta_dim, 1, bias=True)
        self.out_z = PositiveLinear(hidden_dims[-1], 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.wx_layers:
            nn.init.xavier_uniform_(m.weight)
        for m in self.wt_layers:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.out_x.weight)
        nn.init.xavier_uniform_(self.out_t.weight)
        nn.init.zeros_(self.out_t.bias)

    def act(self, z, layer_idx: int):
        is_last = layer_idx == len(self.hidden_dims) - 1
        if self.activation_mode == "relu_last_softplus" and is_last:
            return F.softplus(z)
        return F.relu(z)

    def forward(self, theta, x):
        z = None
        for i in range(len(self.hidden_dims)):
            pre = self.wx_layers[i](x) + self.wt_layers[i](theta)
            if i > 0:
                pre = pre + self.uz_layers[i - 1](z)
            z = self.act(pre, i)
        y = self.out_x(x) + self.out_t(theta) + self.out_z(z)
        return y.squeeze(-1)


class QuadBranch(nn.Module):
    def __init__(self, x_dim: int, theta_dim: int, quad_dims: List[int]):
        super().__init__()
        self.x_layers = nn.ModuleList()
        self.t_layers = nn.ModuleList()
        self.alpha_raw = nn.ParameterList()
        for qdim in quad_dims:
            lx = nn.Linear(x_dim, qdim, bias=False)
            lt = nn.Linear(theta_dim, qdim, bias=True)
            nn.init.xavier_uniform_(lx.weight)
            nn.init.xavier_uniform_(lt.weight)
            nn.init.zeros_(lt.bias)
            self.x_layers.append(lx)
            self.t_layers.append(lt)
            self.alpha_raw.append(nn.Parameter(torch.tensor(0.0)))

    def forward(self, theta, x):
        out = 0.0
        for lx, lt, a_raw in zip(self.x_layers, self.t_layers, self.alpha_raw):
            q = lx(x) + lt(theta)
            alpha = F.softplus(a_raw) + 1e-8
            out = out + 0.5 * alpha * torch.sum(q * q, dim=-1)
        return out


class NormBranch(nn.Module):
    def __init__(self, x_dim: int, theta_dim: int, norm_dims: List[int]):
        super().__init__()
        self.x_layers = nn.ModuleList()
        self.t_layers = nn.ModuleList()
        self.lambda_raw = nn.ParameterList()
        for nd in norm_dims:
            lx = nn.Linear(x_dim, nd, bias=False)
            lt = nn.Linear(theta_dim, nd, bias=True)
            nn.init.xavier_uniform_(lx.weight)
            nn.init.xavier_uniform_(lt.weight)
            nn.init.zeros_(lt.bias)
            self.x_layers.append(lx)
            self.t_layers.append(lt)
            self.lambda_raw.append(nn.Parameter(torch.tensor(0.0)))

    def forward(self, theta, x):
        out = 0.0
        for lx, lt, lam_raw in zip(self.x_layers, self.t_layers, self.lambda_raw):
            u = lx(x) + lt(theta)
            lam = F.softplus(lam_raw) + 1e-8
            out = out + lam * torch.norm(u, dim=-1, p=2)
        return out


class Exp3Model(nn.Module):
    def __init__(
        self,
        model_name: str,
        x_dim: int,
        theta_dim: int,
        hidden_dims: List[int],
        quad_dims: Optional[List[int]] = None,
        norm_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.model_name = model_name.lower()
        assert self.model_name in ["relu", "softplus", "quad", "norm", "soc"]

        activation_mode = "relu_last_softplus" if self.model_name == "softplus" else "relu_all"
        self.backbone = ICNNBackbone(
            x_dim=x_dim,
            theta_dim=theta_dim,
            hidden_dims=hidden_dims,
            activation_mode=activation_mode,
        )

        self.quad_branch = None
        self.norm_branch = None
        if self.model_name in ["quad", "soc"]:
            self.quad_branch = QuadBranch(x_dim, theta_dim, quad_dims or [x_dim])
        if self.model_name in ["norm", "soc"]:
            self.norm_branch = NormBranch(x_dim, theta_dim, norm_dims or [max(4, x_dim // 2)])

    def forward(self, theta, x):
        y = self.backbone(theta, x)
        if self.quad_branch is not None:
            y = y + self.quad_branch(theta, x)
        if self.norm_branch is not None:
            y = y + self.norm_branch(theta, x)
        return y


# ============================================================
# Regression metrics
# ============================================================


@torch.no_grad()
def evaluate_regression(model, loader, device, y_mean: float, y_std: float) -> Dict[str, float]:
    model.eval()
    ys, yhs = [], []
    for batch in loader:
        theta = batch["theta"].to(device, non_blocking=True)
        x = batch["x"].to(device, non_blocking=True)
        y_raw = batch["y_raw"].to(device, non_blocking=True).view(-1)
        pred_norm = model(theta, x)
        pred_raw = pred_norm * y_std + y_mean
        ys.append(y_raw.cpu())
        yhs.append(pred_raw.cpu())
    y = torch.cat(ys, dim=0).numpy()
    yh = torch.cat(yhs, dim=0).numpy()
    mse = float(np.mean((yh - y) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(yh - y)))
    rel_l2 = float(np.linalg.norm(yh - y) / (np.linalg.norm(y) + 1e-12))
    return {"mse": mse, "rmse": rmse, "mae": mae, "rel_l2": rel_l2}


# ============================================================
# Projection to feasible set
# ============================================================


def project_capped_simplex_torch(y: torch.Tensor, total_sum: float, upper: float = None, num_bisect: int = 50) -> torch.Tensor:
    assert y.ndim == 2
    device = y.device

    if upper is None:
        lower_tau = torch.min(y, dim=1, keepdim=True).values - total_sum
        upper_tau = torch.max(y, dim=1, keepdim=True).values
        for _ in range(num_bisect):
            tau = 0.5 * (lower_tau + upper_tau)
            x = torch.clamp(y - tau, min=0.0)
            s = x.sum(dim=1, keepdim=True)
            mask = s > total_sum
            lower_tau = torch.where(mask, tau, lower_tau)
            upper_tau = torch.where(mask, upper_tau, tau)
        tau = 0.5 * (lower_tau + upper_tau)
        x = torch.clamp(y - tau, min=0.0)
        s = x.sum(dim=1, keepdim=True)
        return x * (total_sum / (s + 1e-12))

    lower_tau = torch.min(y, dim=1, keepdim=True).values - upper
    upper_tau = torch.max(y, dim=1, keepdim=True).values
    target = torch.tensor(total_sum, device=device, dtype=y.dtype).view(1, 1)

    for _ in range(num_bisect):
        tau = 0.5 * (lower_tau + upper_tau)
        x = torch.clamp(y - tau, min=0.0, max=upper)
        s = x.sum(dim=1, keepdim=True)
        mask = s > target
        lower_tau = torch.where(mask, tau, lower_tau)
        upper_tau = torch.where(mask, upper_tau, tau)

    tau = 0.5 * (lower_tau + upper_tau)
    x = torch.clamp(y - tau, min=0.0, max=upper)
    s = x.sum(dim=1, keepdim=True)
    diff = target - s
    free_mask = (x > 1e-10) & (x < upper - 1e-10)
    free_count = free_mask.sum(dim=1, keepdim=True).clamp(min=1)
    x = x + free_mask.float() * (diff / free_count.float())
    return torch.clamp(x, min=0.0, max=upper)


def project_feasible_torch(x: torch.Tensor, feasible_type: str, budget: float = None) -> torch.Tensor:
    if feasible_type == "box":
        return torch.clamp(x, min=0.0, max=1.0)
    if feasible_type == "simplex":
        return project_capped_simplex_torch(x, total_sum=1.0, upper=None)
    if feasible_type == "budget":
        assert budget is not None
        return project_capped_simplex_torch(x, total_sum=budget, upper=1.0)
    raise ValueError(f"Unknown feasible type: {feasible_type}")


def initial_feasible_points(n_restart: int, dim: int, feasible_type: str, device: torch.device, budget: float = None) -> torch.Tensor:
    if feasible_type == "box":
        x0 = torch.rand(n_restart, dim, device=device)
        x0[0].fill_(0.5)
        return x0
    if feasible_type == "simplex":
        x0 = torch.rand(n_restart, dim, device=device)
        x0 = project_feasible_torch(x0, feasible_type)
        x0[0].fill_(1.0 / dim)
        return x0
    if feasible_type == "budget":
        x0 = torch.rand(n_restart, dim, device=device)
        x0 = project_feasible_torch(x0, feasible_type, budget=budget)
        x0[0].fill_(budget / dim)
        return x0
    raise ValueError(f"Unknown feasible type: {feasible_type}")


# ============================================================
# Decision optimization on learned objective
# ============================================================


def solve_learned_decision(
    model: nn.Module,
    theta_np: np.ndarray,
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
    feasible_type: str,
    dim: int,
    device: torch.device,
    budget: float = None,
    n_restart: int = 8,
    n_steps: int = 600,
    lr: float = 3e-3,
) -> Tuple[np.ndarray, float]:
    model.eval()
    theta_norm = (theta_np.reshape(1, -1) - theta_mean) / theta_std
    theta = torch.tensor(theta_norm, dtype=torch.float32, device=device)
    theta_batch = theta.repeat(n_restart, 1)

    x = initial_feasible_points(n_restart, dim, feasible_type, device, budget).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)

    best_x = None
    best_val = None
    for _ in range(n_steps):
        optimizer.zero_grad()
        vals = model(theta_batch, x)
        pred = vals.sum()
        pred.backward()
        optimizer.step()
        with torch.no_grad():
            x.copy_(project_feasible_torch(x, feasible_type, budget))
            vals_now = model(theta_batch, x)
            idx = torch.argmin(vals_now).item()
            val = float(vals_now[idx].item())
            if (best_val is None) or (val < best_val):
                best_val = val
                best_x = x[idx].detach().cpu().numpy().copy()

    return best_x, best_val


# ============================================================
# True objective / feasibility
# ============================================================


def true_objective_single(data: Dict, i: int, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    family = get_task_family(data)
    quad_scale = float(normalize_scalar(data["quad_scale"]))
    quad_w = data["quad_w"]
    m = data["m_arr"][i]
    c = data["c_arr"][i]

    quad = 0.5 * quad_scale * np.sum((quad_w * (x - m)) ** 2)
    lin = float(np.dot(c, x))
    out = quad + lin

    if family in {"socp", "socp_two"}:
        lambdas = data["lambda_arr"][i]
        cone_dims = data["cone_dims"]
        for j in range(len(cone_dims)):
            A = data[f"A_{j}"]
            d = data[f"d_arr_{j}"][i]
            out += float(lambdas[j] * np.linalg.norm(A @ x - d, ord=2))

    elif family == "logistic":
        A = data["LOGI_A"]
        shift = data["shift_arr"][i]
        beta = data["beta_arr"][i]
        z = A @ x - shift
        out += float(np.sum(beta * stable_softplus_np(z)))

    elif family == "logsumexp":
        n_blocks = int(normalize_scalar(data["n_blocks"]))
        beta = data["beta_arr"][i]
        for j in range(n_blocks):
            A = data[f"LSE_A_{j}"]
            shift = data[f"shift_arr_{j}"][i]
            out += float(beta[j] * stable_logsumexp_np(A @ x - shift, axis=0))

    elif family == "huber":
        A = data["HUB_A"]
        shift = data["shift_arr"][i]
        beta = data["beta_arr"][i]
        delta = float(normalize_scalar(data["huber_delta"]))
        out += float(np.sum(beta * huber_np(A @ x - shift, delta)))

    else:
        raise ValueError(f"Unknown task family: {family}")

    return out


def feasibility_violation_single(data: Dict, x: np.ndarray) -> float:
    feasible_type = get_feasible_type(data)
    budget = get_budget(data)
    x = np.asarray(x)
    neg_violation = np.maximum(-x, 0.0).sum()

    if feasible_type == "box":
        upper_violation = np.maximum(x - 1.0, 0.0).sum()
        eq_violation = 0.0
    elif feasible_type == "simplex":
        upper_violation = 0.0
        eq_violation = abs(x.sum() - 1.0)
    elif feasible_type == "budget":
        upper_violation = np.maximum(x - 1.0, 0.0).sum()
        eq_violation = abs(x.sum() - float(budget))
    else:
        raise ValueError(f"Unknown feasible type: {feasible_type}")

    return float(neg_violation + upper_violation + eq_violation)


def evaluate_decision_metrics(
    model: nn.Module,
    data: Dict,
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
    device: torch.device,
    n_restart: int = 8,
    n_steps: int = 600,
    lr: float = 3e-3,
) -> Dict[str, float]:
    feasible_type = get_feasible_type(data)
    dim = int(normalize_scalar(data["dim"]))
    budget = get_budget(data)
    contexts = data["contexts"]
    x_star = data["x_star"]
    f_star = data["f_star"]

    regrets, rel_regrets, x_errs, feas_viol, solve_times = [], [], [], [], []
    for i in range(contexts.shape[0]):
        t0 = time.perf_counter()
        x_hat, _ = solve_learned_decision(
            model=model,
            theta_np=contexts[i],
            theta_mean=theta_mean,
            theta_std=theta_std,
            feasible_type=feasible_type,
            dim=dim,
            device=device,
            budget=budget,
            n_restart=n_restart,
            n_steps=n_steps,
            lr=lr,
        )
        dt = time.perf_counter() - t0
        f_hat_true = true_objective_single(data, i, x_hat)
        reg = float(f_hat_true - f_star[i])
        rel_reg = float(reg / (abs(float(f_star[i])) + 1e-8))
        x_err = float(np.linalg.norm(x_hat - x_star[i]))
        feas = feasibility_violation_single(data, x_hat)
        regrets.append(reg)
        rel_regrets.append(rel_reg)
        x_errs.append(x_err)
        feas_viol.append(feas)
        solve_times.append(dt)

    regrets = np.asarray(regrets)
    rel_regrets = np.asarray(rel_regrets)
    x_errs = np.asarray(x_errs)
    feas_viol = np.asarray(feas_viol)
    solve_times = np.asarray(solve_times)

    return {
        "regret_mean": float(regrets.mean()),
        "regret_median": float(np.median(regrets)),
        "regret_max": float(regrets.max()),
        "rel_regret_mean": float(rel_regrets.mean()),
        "rel_regret_median": float(np.median(rel_regrets)),
        "x_error_mean": float(x_errs.mean()),
        "x_error_median": float(np.median(x_errs)),
        "feas_violation_mean": float(feas_viol.mean()),
        "feas_violation_max": float(feas_viol.max()),
        "decision_time_mean_sec": float(solve_times.mean()),
    }


# ============================================================
# Train / validate
# ============================================================


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_n = 0.0, 0
    for batch in loader:
        theta = batch["theta"].to(device, non_blocking=True)
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).view(-1)
        optimizer.zero_grad()
        pred = model(theta, x)
        loss = F.smooth_l1_loss(pred, y, beta=1.0)
        loss.backward()
        optimizer.step()
        bs = theta.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs
    return total_loss / max(total_n, 1)


@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.eval()
    total_loss, total_n = 0.0, 0
    for batch in loader:
        theta = batch["theta"].to(device, non_blocking=True)
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).view(-1)
        pred = model(theta, x)
        loss = F.smooth_l1_loss(pred, y, beta=1.0)
        bs = theta.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs
    return total_loss / max(total_n, 1)


# ============================================================
# Fair model configuration
# ============================================================

SOC_ANCHOR_SPECS = {
    10: {"hidden_dims": [128, 128], "quad_dims": [10], "norm_dims": [10, 10]},
    20: {"hidden_dims": [128, 128], "quad_dims": [20], "norm_dims": [20, 20]},
    50: {"hidden_dims": [256, 256], "quad_dims": [50], "norm_dims": [50, 50]},
    "default": {"hidden_dims": [128, 128], "quad_dims": None, "norm_dims": [15, 15]},
}


def get_anchor_spec(dim: int) -> Dict:
    spec = SOC_ANCHOR_SPECS.get(dim, SOC_ANCHOR_SPECS["default"])
    quad_dims = spec["quad_dims"] if spec["quad_dims"] is not None else [dim]
    return {
        "hidden_dims": list(spec["hidden_dims"]),
        "quad_dims": list(quad_dims),
        "norm_dims": list(spec["norm_dims"]),
    }


def search_baseline_hidden_dims(
    model_name: str,
    x_dim: int,
    theta_dim: int,
    anchor_spec: Dict,
    target_params: int,
    overshoot_ratio: float = 1.02,
    width_step: int = 16,
    max_extra_width_steps: int = 24,
) -> List[int]:
    target = int(np.ceil(target_params * overshoot_ratio))
    anchor_hidden = anchor_spec["hidden_dims"]
    anchor_width = anchor_hidden[0]
    anchor_depth = len(anchor_hidden)

    candidates: List[List[int]] = []

    # 先同深度加宽
    for k in range(max_extra_width_steps + 1):
        width = anchor_width + k * width_step
        candidates.append([width] * anchor_depth)

    # 再最多到 3 层，并继续优先控制宽度
    if anchor_depth < 3:
        for k in range(max_extra_width_steps + 1):
            width = anchor_width + k * width_step
            candidates.append([width] * 3)

    best_hidden = None
    best_params = None
    for hidden_dims in candidates:
        model = Exp3Model(
            model_name=model_name,
            x_dim=x_dim,
            theta_dim=theta_dim,
            hidden_dims=hidden_dims,
            quad_dims=[],
            norm_dims=[],
        )
        nparams = count_parameters(model)
        if nparams >= target:
            if best_params is None or nparams < best_params:
                best_hidden = hidden_dims
                best_params = nparams

    if best_hidden is None:
        raise RuntimeError(
            f"Cannot find baseline width/depth for {model_name} at dim={x_dim}. "
            f"target_params={target_params}."
        )

    return best_hidden


def get_model_structure_config(dim: int, theta_dim: int) -> Dict[str, Dict]:
    anchor = get_anchor_spec(dim)
    soc_model = Exp3Model(
        model_name="soc",
        x_dim=dim,
        theta_dim=theta_dim,
        hidden_dims=anchor["hidden_dims"],
        quad_dims=anchor["quad_dims"],
        norm_dims=anchor["norm_dims"],
    )
    soc_params = count_parameters(soc_model)

    relu_hidden = search_baseline_hidden_dims("relu", dim, theta_dim, anchor, soc_params)
    softplus_hidden = search_baseline_hidden_dims("softplus", dim, theta_dim, anchor, soc_params)

    return {
        "soc_anchor_params": soc_params,
        "relu": {"hidden_dims": relu_hidden, "quad_dims": [], "norm_dims": []},
        "softplus": {"hidden_dims": softplus_hidden, "quad_dims": [], "norm_dims": []},
        "quad": {"hidden_dims": anchor["hidden_dims"], "quad_dims": anchor["quad_dims"], "norm_dims": []},
        "norm": {"hidden_dims": anchor["hidden_dims"], "quad_dims": [], "norm_dims": anchor["norm_dims"]},
        "soc": {"hidden_dims": anchor["hidden_dims"], "quad_dims": anchor["quad_dims"], "norm_dims": anchor["norm_dims"]},
    }


# ============================================================
# Single experiment
# ============================================================


def run_single_experiment(global_cfg: Dict, task: str, dim: int, model_name: str, seed: int) -> Dict:
    device = torch.device(global_cfg["device"])
    data_root = Path(global_cfg["data_root"])
    exp_dir = Path(global_cfg["save_root"]) / task / f"d{dim}" / model_name / f"seed{seed}"
    ensure_dir(exp_dir)

    train_path = data_root / task / f"d{dim}" / "train.npz"
    test_path = data_root / task / f"d{dim}" / "test.npz"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing file(s): {train_path}, {test_path}")

    raw_train = load_npz_dict(str(train_path))
    raw_test = load_npz_dict(str(test_path))
    raw_subtrain, raw_val = split_instance_dict(raw_train, val_ratio=global_cfg["train"]["val_ratio"], seed=seed)
    raw_test_for_decision = subset_instance_dict(raw_test, global_cfg["decision_eval"]["max_test_instances"], seed=seed)

    arr_subtrain = flatten_supervised_arrays(raw_subtrain)
    arr_val = flatten_supervised_arrays(raw_val)
    arr_test = flatten_supervised_arrays(raw_test)

    arr_subtrain, arr_val, arr_test, theta_mean, theta_std = normalize_theta_arrays(arr_subtrain, arr_val, arr_test)

    y_mean = float(arr_subtrain["y"].mean())
    y_std = max(float(arr_subtrain["y"].std()), 1e-8)

    ds_subtrain = Exp3FlatDataset(arr_subtrain, y_mean=y_mean, y_std=y_std)
    ds_val = Exp3FlatDataset(arr_val, y_mean=y_mean, y_std=y_std)
    ds_test = Exp3FlatDataset(arr_test, y_mean=y_mean, y_std=y_std)

    pin_mem = torch.cuda.is_available() and device.type == "cuda"
    dl_subtrain = DataLoader(ds_subtrain, batch_size=global_cfg["train"]["batch_size"], shuffle=True,
                             num_workers=global_cfg["train"]["num_workers"], pin_memory=pin_mem)
    dl_val = DataLoader(ds_val, batch_size=global_cfg["train"]["batch_size"], shuffle=False,
                        num_workers=global_cfg["train"]["num_workers"], pin_memory=pin_mem)
    dl_test = DataLoader(ds_test, batch_size=global_cfg["train"]["batch_size"], shuffle=False,
                         num_workers=global_cfg["train"]["num_workers"], pin_memory=pin_mem)

    theta_dim = raw_train["contexts"].shape[1]
    structure_cfg = get_model_structure_config(dim, theta_dim)
    model_cfg = structure_cfg[model_name]

    model = Exp3Model(
        model_name=model_name,
        x_dim=dim,
        theta_dim=theta_dim,
        hidden_dims=model_cfg["hidden_dims"],
        quad_dims=model_cfg["quad_dims"],
        norm_dims=model_cfg["norm_dims"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=global_cfg["train"]["lr"],
        weight_decay=global_cfg["train"]["weight_decay"],
    )

    train_y = arr_subtrain["y"].reshape(-1)
    val_y = arr_val["y"].reshape(-1)
    test_y = arr_test["y"].reshape(-1)

    print("=" * 120)
    print(f"Start | task={task:>20s} | dim={dim:<3d} | model={model_name:<8s} | seed={seed}")
    print(
        f"Device={device} | params={count_parameters(model)} | "
        f"target_soc_params={structure_cfg['soc_anchor_params']} | hidden={model_cfg['hidden_dims']} | "
        f"quad={model_cfg['quad_dims']} | norm={model_cfg['norm_dims']}"
    )
    print(
        f"theta_mean_norm={float(np.linalg.norm(theta_mean)):.4f} | theta_std_min={float(theta_std.min()):.4f} | "
        f"y_train[min,max,std]=({float(train_y.min()):.4f},{float(train_y.max()):.4f},{float(train_y.std()):.4f}) | "
        f"y_val[min,max,std]=({float(val_y.min()):.4f},{float(val_y.max()):.4f},{float(val_y.std()):.4f}) | "
        f"y_test[min,max,std]=({float(test_y.min()):.4f},{float(test_y.max()):.4f},{float(test_y.std()):.4f})"
    )

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    patience_count = 0
    log_rows = []
    total_t0 = time.perf_counter()

    for epoch in range(1, global_cfg["train"]["epochs"] + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, dl_subtrain, optimizer, device)
        val_loss = validate_one_epoch(model, dl_val, device)
        val_reg = evaluate_regression(model, dl_val, device, y_mean=y_mean, y_std=y_std)
        dt = time.perf_counter() - t0

        row = {
            "epoch": epoch,
            "train_smoothl1_norm": train_loss,
            "val_smoothl1_norm": val_loss,
            "val_mse_raw": val_reg["mse"],
            "val_rmse_raw": val_reg["rmse"],
            "val_mae_raw": val_reg["mae"],
            "val_rel_l2": val_reg["rel_l2"],
            "epoch_time_sec": dt,
        }
        log_rows.append(row)

        print(
            f"[Epoch {epoch:03d}] train={train_loss:.6f} | val={val_loss:.6f} | "
            f"val_rmse={val_reg['rmse']:.6f} | val_rel_l2={val_reg['rel_l2']:.6f} | time={dt:.2f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= global_cfg["train"]["patience"]:
            print(f"Early stop at epoch {epoch}.")
            break

    total_train_time = time.perf_counter() - total_t0
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, exp_dir / "best_model.pt")

    with open(exp_dir / "train_log.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    train_reg = evaluate_regression(model, dl_subtrain, device, y_mean=y_mean, y_std=y_std)
    val_reg = evaluate_regression(model, dl_val, device, y_mean=y_mean, y_std=y_std)
    test_reg = evaluate_regression(model, dl_test, device, y_mean=y_mean, y_std=y_std)

    print("Running decision evaluation ...")
    test_decision = evaluate_decision_metrics(
        model=model,
        data=raw_test_for_decision,
        theta_mean=theta_mean,
        theta_std=theta_std,
        device=device,
        n_restart=global_cfg["decision_eval"]["n_restart"],
        n_steps=global_cfg["decision_eval"]["n_steps"],
        lr=global_cfg["decision_eval"]["lr"],
    )

    metrics = {
        "config": {
            "task": task,
            "task_family": get_task_family(raw_train),
            "feasible_type": get_feasible_type(raw_train),
            "dim": dim,
            "model": model_name,
            "seed": seed,
            "device": str(device),
            "hidden_dims": model_cfg["hidden_dims"],
            "quad_dims": model_cfg["quad_dims"],
            "norm_dims": model_cfg["norm_dims"],
            "target_soc_params": structure_cfg["soc_anchor_params"],
            "epochs": global_cfg["train"]["epochs"],
            "batch_size": global_cfg["train"]["batch_size"],
            "lr": global_cfg["train"]["lr"],
            "weight_decay": global_cfg["train"]["weight_decay"],
            "patience": global_cfg["train"]["patience"],
            "val_ratio": global_cfg["train"]["val_ratio"],
            "decision_steps": global_cfg["decision_eval"]["n_steps"],
            "decision_lr": global_cfg["decision_eval"]["lr"],
            "decision_restarts": global_cfg["decision_eval"]["n_restart"],
            "decision_eval_instances": int(raw_test_for_decision["contexts"].shape[0]),
            "num_parameters": count_parameters(model),
            "best_epoch": best_epoch,
            "y_mean": y_mean,
            "y_std": y_std,
            "theta_mean": theta_mean.tolist(),
            "theta_std": theta_std.tolist(),
            "total_train_time_sec": total_train_time,
        },
        "train_regression": train_reg,
        "val_regression": val_reg,
        "test_regression": test_reg,
        "test_decision": test_decision,
    }

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(
        f"Done | {task} | d={dim} | {model_name} | test_rel_l2={test_reg['rel_l2']:.6f} | "
        f"regret_mean={test_decision['regret_mean']:.6f} | feas_mean={test_decision['feas_violation_mean']:.3e}"
    )

    return {
        "task": task,
        "task_family": get_task_family(raw_train),
        "feasible_type": get_feasible_type(raw_train),
        "dim": dim,
        "model": model_name,
        "seed": seed,
        "num_parameters": metrics["config"]["num_parameters"],
        "target_soc_params": structure_cfg["soc_anchor_params"],
        "best_epoch": best_epoch,
        "train_time_sec": total_train_time,
        "train_rel_l2": train_reg["rel_l2"],
        "val_rel_l2": val_reg["rel_l2"],
        "test_rel_l2": test_reg["rel_l2"],
        "test_rmse": test_reg["rmse"],
        "test_mae": test_reg["mae"],
        "regret_mean": test_decision["regret_mean"],
        "regret_median": test_decision["regret_median"],
        "rel_regret_mean": test_decision["rel_regret_mean"],
        "x_error_mean": test_decision["x_error_mean"],
        "feas_violation_mean": test_decision["feas_violation_mean"],
        "decision_time_mean_sec": test_decision["decision_time_mean_sec"],
    }


# ============================================================
# Summary output
# ============================================================


def save_summary_csv(rows: List[Dict], path: Path):
    if len(rows) == 0:
        return
    ensure_dir(path.parent)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_all_results_json(rows: List[Dict], path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def aggregate_rows(rows: List[Dict], metric_keys: List[str]) -> List[Dict]:
    groups = {}
    for r in rows:
        key = (r["task"], r["dim"], r["model"])
        groups.setdefault(key, []).append(r)

    agg_rows = []
    for (task, dim, model), group in sorted(groups.items()):
        base = {
            "task": task,
            "task_family": group[0]["task_family"],
            "feasible_type": group[0]["feasible_type"],
            "dim": dim,
            "model": model,
            "n_seeds": len(group),
        }
        for mk in metric_keys:
            vals = np.array([float(g[mk]) for g in group], dtype=np.float64)
            base[f"{mk}_mean"] = float(vals.mean())
            base[f"{mk}_std"] = float(vals.std(ddof=0))
        agg_rows.append(base)
    return agg_rows


def print_group_comparison(rows: List[Dict], sort_key: str = "regret_mean_mean"):
    if len(rows) == 0:
        return
    grouped = {}
    for r in rows:
        grouped.setdefault((r["task"], r["dim"]), []).append(r)

    print("\n" + "#" * 130)
    print("AGGREGATED COMPARISON SUMMARY")
    print("#" * 130)

    for (task, dim), group in grouped.items():
        group = sorted(group, key=lambda z: z[sort_key])
        print(f"\n[Task={task}, Dim={dim}] sorted by {sort_key}")
        print("-" * 130)
        print(
            f"{'Model':<10s} {'Seeds':>5s} {'Params':>10s} {'TargetSOC':>10s} {'TestRelL2':>16s} "
            f"{'RegretMean':>18s} {'RelRegret':>18s} {'XErr':>16s} {'Feas':>14s}"
        )
        for r in group:
            print(
                f"{r['model']:<10s} {int(r['n_seeds']):>5d} "
                f"{int(r['num_parameters_mean']):>10d} {int(r['target_soc_params_mean']):>10d} "
                f"{r['test_rel_l2_mean']:>8.6f}±{r['test_rel_l2_std']:<7.6f} "
                f"{r['regret_mean_mean']:>9.6f}±{r['regret_mean_std']:<8.6f} "
                f"{r['rel_regret_mean_mean']:>9.6f}±{r['rel_regret_mean_std']:<8.6f} "
                f"{r['x_error_mean_mean']:>8.6f}±{r['x_error_mean_std']:<7.6f} "
                f"{r['feas_violation_mean_mean']:>10.3e}"
            )
        print("-" * 130)


# ============================================================
# Config
# ============================================================


def build_config() -> Dict:
    use_cuda = torch.cuda.is_available()
    return {
        "device": "cuda" if use_cuda else "cpu",
        "data_root": "exp3_convex_fair_data",
        "save_root": "exp3_compare_runs_fair_fixed",
        "tasks": [
            "simplex_socp",
            "box_socp",
            "budget_twocone_socp",
            "simplex_logistic",
            "box_logsumexp",
            "budget_huber",
        ],
        "dims": [10, 20, 50],  # debug first; paper version can use [10, 20, 50]
        "models": ["relu", "softplus", "quad", "norm", "soc"],
        "seeds": [2026, 2025, 2024],
        "train": {
            "epochs": 120,
            "batch_size": 1024,
            "lr": 2e-3,
            "weight_decay": 1e-6,
            "patience": 5,
            "val_ratio": 0.1,
            "num_workers": 0,
        },
        "decision_eval": {
            "n_restart": 5,
            "n_steps": 200,
            "lr": 2e-3,
            "max_test_instances": 20,
        },
    }


# ============================================================
# Main
# ============================================================


def main():
    cfg = build_config()
    set_seed(cfg["seeds"][0])

    save_root = Path(cfg["save_root"])
    ensure_dir(save_root)
    all_rows = []

    for seed in cfg["seeds"]:
        for task in cfg["tasks"]:
            for dim in cfg["dims"]:
                for model_name in cfg["models"]:
                    row = run_single_experiment(cfg, task, dim, model_name, seed)
                    all_rows.append(row)
                    save_summary_csv(all_rows, save_root / "summary_per_seed.csv")
                    save_all_results_json(all_rows, save_root / "all_results_per_seed.json")

                    metric_keys = [
                        "num_parameters",
                        "target_soc_params",
                        "train_time_sec",
                        "train_rel_l2",
                        "val_rel_l2",
                        "test_rel_l2",
                        "test_rmse",
                        "test_mae",
                        "regret_mean",
                        "regret_median",
                        "rel_regret_mean",
                        "x_error_mean",
                        "feas_violation_mean",
                        "decision_time_mean_sec",
                    ]
                    agg_rows = aggregate_rows(all_rows, metric_keys)
                    save_summary_csv(agg_rows, save_root / "summary_aggregated.csv")
                    save_all_results_json(agg_rows, save_root / "all_results_aggregated.json")

    metric_keys = [
        "num_parameters",
        "target_soc_params",
        "train_time_sec",
        "train_rel_l2",
        "val_rel_l2",
        "test_rel_l2",
        "test_rmse",
        "test_mae",
        "regret_mean",
        "regret_median",
        "rel_regret_mean",
        "x_error_mean",
        "feas_violation_mean",
        "decision_time_mean_sec",
    ]
    agg_rows = aggregate_rows(all_rows, metric_keys)
    print_group_comparison(agg_rows, sort_key="regret_mean_mean")

    print("\nAll experiments finished.")
    print(f"Per-seed CSV     : {(save_root / 'summary_per_seed.csv').resolve()}")
    print(f"Aggregated CSV   : {(save_root / 'summary_aggregated.csv').resolve()}")
    print(f"Per-seed JSON    : {(save_root / 'all_results_per_seed.json').resolve()}")
    print(f"Aggregated JSON  : {(save_root / 'all_results_aggregated.json').resolve()}")


if __name__ == "__main__":
    main()
