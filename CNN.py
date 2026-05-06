# soc_cnn_olivetti_downstream.py
# Downstream test for Convolutional SOC-ICNN on real 64x64 Olivetti face images.
#
# Output:
#   results_soc_cnn_olivetti/
#     summary.csv
#     summary_table.tex
#     cnn_downstream_real_examples.pdf
#     cnn_downstream_real_examples.png
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
    out_dir: str = "results_soc_cnn_olivetti"
    data_root: str = "./data"
    download: bool = True

    # data
    image_size: int = 64
    train_size: int = 280
    val_size: int = 60
    test_size: int = 60
    candidates_per_image: int = 16

    # corruption
    base_keep_prob: float = 0.80
    num_holes_min: int = 2
    num_holes_max: int = 4
    hole_size_min: int = 8
    hole_size_max: int = 12
    noise_std: float = 0.1

    # true convex restoration energy
    rho: float = 1.0
    beta_lap: float = 0.035
    lam_tv: float = 0.055
    tv_eps: float = 1e-4
    blur_sigma: float = 0.7

    # model
    hidden_channels: int = 10
    layers: int = 3
    branch_channels: int = 16
    norm_groups: int = 8

    # training
    epochs: int = 50
    batch_size: int = 280
    lr: float = 2e-3
    weight_decay: float = 1e-6

    # downstream optimization
    true_opt_steps: int = 700
    true_opt_lr: float = 0.045
    surrogate_opt_steps: int = 350
    surrogate_opt_lr: float = 0.035
    restarts: int = 3
    eval_batch_size: int = 20

    # visualization
    num_plot_examples: int = 5
    plot_indices: Tuple[int, ...] = (0, 5, 12, 21, 34)

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
# Olivetti Faces data
# ============================================================

def load_real_image_tensors(cfg: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads sklearn Olivetti Faces:
        - real grayscale face images
        - native resolution 64 x 64
        - values in [0, 1]
    """
    from sklearn.datasets import fetch_olivetti_faces

    data = fetch_olivetti_faces(
        data_home=cfg.data_root,
        shuffle=True,
        random_state=cfg.seed,
        download_if_missing=cfg.download,
    )

    images = data.images.astype(np.float32)       # [400, 64, 64]
    images = images[:, None, :, :]                # [400, 1, 64, 64]
    all_imgs = torch.tensor(images, dtype=torch.float32)

    n_total = all_imgs.shape[0]
    n_needed = cfg.train_size + cfg.val_size + cfg.test_size

    if n_needed > n_total:
        raise ValueError(
            f"Olivetti Faces has only {n_total} images, but requested "
            f"train_size + val_size + test_size = {n_needed}. "
            f"Use smaller split sizes, e.g., train=280, val=60, test=60."
        )

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n_total)

    train_idx = perm[:cfg.train_size]
    val_idx = perm[cfg.train_size:cfg.train_size + cfg.val_size]
    test_idx = perm[
        cfg.train_size + cfg.val_size:
        cfg.train_size + cfg.val_size + cfg.test_size
    ]

    clean_train = all_imgs[train_idx]
    clean_val = all_imgs[val_idx]
    clean_test = all_imgs[test_idx]

    return clean_train, clean_val, clean_test


# ============================================================
# True convex image energy
# ============================================================

def gaussian_kernel_2d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def blur_conv(x: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = gaussian_kernel_2d(5, sigma, x.device, x.dtype)
    return F.conv2d(x, kernel, padding=2)


def laplacian_conv(x: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)
    return F.conv2d(x, kernel, padding=1)


def finite_differences(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dx = F.pad(dx, (0, 1, 0, 0), mode="constant", value=0.0)

    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = F.pad(dy, (0, 0, 0, 1), mode="constant", value=0.0)

    return dx, dy


def make_corrupted_observation(
    clean: torch.Tensor,
    cfg: Config,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates a corrupted observation using mild blur, Gaussian noise,
    light random pixel masking, and several rectangular missing blocks.
    """
    clean = clean.to(device)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    batch, _, height, width = clean.shape

    blurred = blur_conv(clean, cfg.blur_sigma)

    # Start with a light random keep mask.
    mask = (
        torch.rand(clean.shape, generator=gen, device=device)
        < cfg.base_keep_prob
    ).float()

    # Add rectangular holes.
    for b in range(batch):
        num_holes = int(
            torch.randint(
                low=cfg.num_holes_min,
                high=cfg.num_holes_max + 1,
                size=(1,),
                generator=gen,
                device=device,
            ).item()
        )

        for _ in range(num_holes):
            hole_h = int(
                torch.randint(
                    low=cfg.hole_size_min,
                    high=cfg.hole_size_max + 1,
                    size=(1,),
                    generator=gen,
                    device=device,
                ).item()
            )
            hole_w = int(
                torch.randint(
                    low=cfg.hole_size_min,
                    high=cfg.hole_size_max + 1,
                    size=(1,),
                    generator=gen,
                    device=device,
                ).item()
            )

            top = int(
                torch.randint(
                    low=0,
                    high=max(1, height - hole_h + 1),
                    size=(1,),
                    generator=gen,
                    device=device,
                ).item()
            )
            left = int(
                torch.randint(
                    low=0,
                    high=max(1, width - hole_w + 1),
                    size=(1,),
                    generator=gen,
                    device=device,
                ).item()
            )

            mask[b, :, top:top + hole_h, left:left + hole_w] = 0.0

    noise = cfg.noise_std * torch.randn(clean.shape, generator=gen, device=device)
    y_obs = torch.clamp(mask * (blurred + noise), 0.0, 1.0)

    # Context has two channels:
    #   channel 1: observed image
    #   channel 2: observation mask
    ctx = torch.cat([y_obs, mask], dim=1)

    return y_obs.detach(), mask.detach(), ctx.detach()


def true_energy(
    x: torch.Tensor,
    y_obs: torch.Tensor,
    mask: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    """
    Convex restoration energy:
        rho/2 || M ⊙ (H*X - Y) ||_F^2
      + beta/2 || L*X ||_F^2
      + lambda * TV_epsilon(X).

    Returns:
        Tensor of shape [B].
    """
    hx = blur_conv(x, cfg.blur_sigma)

    fidelity = 0.5 * cfg.rho * (mask * (hx - y_obs)).pow(2).mean(dim=(1, 2, 3))

    lap = laplacian_conv(x)
    smooth = 0.5 * cfg.beta_lap * lap.pow(2).mean(dim=(1, 2, 3))

    dx, dy = finite_differences(x)
    tv = cfg.lam_tv * torch.sqrt(dx.pow(2) + dy.pow(2) + cfg.tv_eps).mean(dim=(1, 2, 3))

    return fidelity + smooth + tv


# ============================================================
# Candidate supervised dataset
# ============================================================

def build_candidate_dataset(
    clean: torch.Tensor,
    y_obs: torch.Tensor,
    mask: torch.Tensor,
    ctx: torch.Tensor,
    cfg: Config,
    device: torch.device,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clean = clean.to(device)
    y_obs = y_obs.to(device)
    mask = mask.to(device)
    ctx = ctx.to(device)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    xs = []
    ctxs = []
    vals = []

    n = clean.shape[0]

    for j in range(cfg.candidates_per_image):
        if j == 0:
            cand = torch.clamp(y_obs, 0.0, 1.0)

        elif j == 1:
            cand = clean.clone()

        elif j == 2:
            cand = blur_conv(clean, cfg.blur_sigma).clamp(0.0, 1.0)

        elif j == 3:
            cand = torch.rand(clean.shape, generator=gen, device=device)

        elif j % 4 == 0:
            cand = torch.clamp(
                clean + 0.25 * torch.randn(clean.shape, generator=gen, device=device),
                0.0,
                1.0,
            )

        elif j % 4 == 1:
            cand = torch.clamp(
                y_obs + 0.25 * torch.randn(clean.shape, generator=gen, device=device),
                0.0,
                1.0,
            )

        elif j % 4 == 2:
            alpha = torch.rand((n, 1, 1, 1), generator=gen, device=device)
            rand_img = torch.rand(clean.shape, generator=gen, device=device)
            cand = torch.clamp(alpha * clean + (1.0 - alpha) * rand_img, 0.0, 1.0)

        else:
            alpha = torch.rand((n, 1, 1, 1), generator=gen, device=device)
            cand = torch.clamp(alpha * clean + (1.0 - alpha) * y_obs, 0.0, 1.0)

        val = true_energy(cand, y_obs, mask, cfg)

        xs.append(cand.detach().cpu())
        ctxs.append(ctx.detach().cpu())
        vals.append(val.detach().cpu())

    x_all = torch.cat(xs, dim=0)
    ctx_all = torch.cat(ctxs, dim=0)
    v_all = torch.cat(vals, dim=0)

    return x_all, ctx_all, v_all


# ============================================================
# Convex CNN surrogate models
# ============================================================

class PositiveConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.raw_weight = nn.Parameter(torch.empty(out_ch, in_ch, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.padding = padding

        nn.init.normal_(self.raw_weight, mean=-3.0, std=0.15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.softplus(self.raw_weight)
        return F.conv2d(x, weight, self.bias, padding=self.padding)


class ContextAffineInX(nn.Module):
    """
    Context-dependent affine term in decision image X:
        <a(C), X> + b(C).

    For fixed context C, this is affine in X, so convexity is preserved.
    """
    def __init__(self, ctx_channels: int = 2, hidden: int = 16):
        super().__init__()

        self.coeff_net = nn.Sequential(
            nn.Conv2d(ctx_channels, hidden, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )

        self.bias_net = nn.Sequential(
            nn.Conv2d(ctx_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        coeff = self.coeff_net(ctx)
        lin = (coeff * x).mean(dim=(1, 2, 3))
        bias = self.bias_net(ctx).squeeze(-1)
        return lin + bias


class ConvICNNBackbone(nn.Module):
    """
    Conv-ICNN backbone, convex in decision image X for fixed context C:

        z_l = act(W_l * X + U_l * z_{l-1} + C_l * context + b_l),

    where U_l has nonnegative convolution weights.
    """
    def __init__(
        self,
        hidden_channels: int,
        layers: int,
        activation: str = "relu",
        ctx_channels: int = 2,
    ):
        super().__init__()

        self.layers = layers
        self.activation_name = activation

        self.x_convs = nn.ModuleList()
        self.ctx_convs = nn.ModuleList()
        self.z_convs = nn.ModuleList()

        for layer in range(layers):
            self.x_convs.append(
                nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1)
            )

            self.ctx_convs.append(
                nn.Conv2d(ctx_channels, hidden_channels, kernel_size=3, padding=1)
            )

            if layer > 0:
                self.z_convs.append(
                    PositiveConv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
                )

        self.raw_out_weight = nn.Parameter(torch.full((hidden_channels,), -2.0))
        self.affine_x = ContextAffineInX(ctx_channels=ctx_channels, hidden=hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in list(self.x_convs) + list(self.ctx_convs):
            nn.init.xavier_uniform_(module.weight, gain=0.6)
            nn.init.zeros_(module.bias)

    def activation(self, a: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "relu":
            return F.relu(a)
        if self.activation_name == "softplus":
            return F.softplus(a, beta=1.0)
        raise ValueError(f"Unknown activation: {self.activation_name}")

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        z = None

        for layer in range(self.layers):
            a = self.x_convs[layer](x) + self.ctx_convs[layer](ctx)

            if layer > 0:
                a = a + self.z_convs[layer - 1](z)

            z = self.activation(a)

        out_weight = F.softplus(self.raw_out_weight).view(1, -1, 1, 1)
        convex_out = (out_weight * z).mean(dim=(1, 2, 3))
        affine_out = self.affine_x(x, ctx)

        return convex_out + affine_out


class QuadraticBranch(nn.Module):
    """
    Quadratic branch:
        alpha/2 || B_x * X + B_c * C + e ||^2, alpha >= 0.
    """
    def __init__(self, out_channels: int, ctx_channels: int = 2):
        super().__init__()

        self.x_conv = nn.Conv2d(1, out_channels, kernel_size=5, padding=2)
        self.ctx_conv = nn.Conv2d(ctx_channels, out_channels, kernel_size=5, padding=2)
        self.raw_alpha = nn.Parameter(torch.tensor(-0.3))

        nn.init.xavier_uniform_(self.x_conv.weight, gain=0.45)
        nn.init.xavier_uniform_(self.ctx_conv.weight, gain=0.45)
        nn.init.zeros_(self.x_conv.bias)
        nn.init.zeros_(self.ctx_conv.bias)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        q = self.x_conv(x) + self.ctx_conv(ctx)
        alpha = F.softplus(self.raw_alpha)
        return 0.5 * alpha * q.pow(2).mean(dim=(1, 2, 3))


class NormBranch(nn.Module):
    """
    Conic branch:
        lambda * mean grouped local channel norms, lambda >= 0.
    """
    def __init__(self, groups: int = 8, ctx_channels: int = 2):
        super().__init__()

        self.groups = groups
        out_channels = 2 * groups

        self.x_conv = nn.Conv2d(1, out_channels, kernel_size=5, padding=2)
        self.ctx_conv = nn.Conv2d(ctx_channels, out_channels, kernel_size=5, padding=2)
        self.raw_lambda = nn.Parameter(torch.tensor(-0.3))

        nn.init.xavier_uniform_(self.x_conv.weight, gain=0.45)
        nn.init.xavier_uniform_(self.ctx_conv.weight, gain=0.45)
        nn.init.zeros_(self.x_conv.bias)
        nn.init.zeros_(self.ctx_conv.bias)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        u = self.x_conv(x) + self.ctx_conv(ctx)

        batch, channels, height, width = u.shape
        u = u.view(batch, self.groups, 2, height, width)

        norms = torch.sqrt(u.pow(2).sum(dim=2) + 1e-8)
        lam = F.softplus(self.raw_lambda)

        return lam * norms.mean(dim=(1, 2, 3))


class ConvSOCICNN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        layers: int,
        branch_channels: int,
        norm_groups: int,
        activation: str = "relu",
        use_quad: bool = False,
        use_norm: bool = False,
        ctx_channels: int = 2,
    ):
        super().__init__()

        self.backbone = ConvICNNBackbone(
            hidden_channels=hidden_channels,
            layers=layers,
            activation=activation,
            ctx_channels=ctx_channels,
        )

        self.use_quad = use_quad
        self.use_norm = use_norm

        self.quad = (
            QuadraticBranch(branch_channels, ctx_channels=ctx_channels)
            if use_quad
            else None
        )

        self.norm = (
            NormBranch(norm_groups, ctx_channels=ctx_channels)
            if use_norm
            else None
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x, ctx)

        if self.use_quad:
            out = out + self.quad(x, ctx)

        if self.use_norm:
            out = out + self.norm(x, ctx)

        return out


def build_models(cfg: Config, device: torch.device) -> Dict[str, nn.Module]:
    specs = {
        "Conv-ReLU": dict(activation="relu", use_quad=False, use_norm=False),
        "Conv-Softplus": dict(activation="softplus", use_quad=False, use_norm=False),
        "Conv-Quad": dict(activation="relu", use_quad=True, use_norm=False),
        "Conv-Norm": dict(activation="relu", use_quad=False, use_norm=True),
        "Conv-SOC": dict(activation="relu", use_quad=True, use_norm=True),
    }

    models = {}

    for name, kwargs in specs.items():
        model = ConvSOCICNN(
            hidden_channels=cfg.hidden_channels,
            layers=cfg.layers,
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

    train_v_std = (train_v - mean) / std
    val_v_std = (val_v - mean) / std
    test_v_std = (test_v - mean) / std

    return train_v_std, val_v_std, test_v_std, mean, std


def train_one_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    history = {"train": [], "val": []}

    best_state = None
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []

        for xb, ctxb, vb in train_loader:
            xb = xb.to(device)
            ctxb = ctxb.to(device)
            vb = vb.to(device)

            pred = model(xb, ctxb)
            loss = F.mse_loss(pred, vb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, ctxb, vb in val_loader:
                xb = xb.to(device)
                ctxb = ctxb.to(device)
                vb = vb.to(device)

                pred = model(xb, ctxb)
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
        for xb, ctxb, vb_std in loader:
            xb = xb.to(device)
            ctxb = ctxb.to(device)
            vb_std = vb_std.to(device)

            pred_std = model(xb, ctxb)

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
    y_obs: torch.Tensor,
    mask: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    y_obs = y_obs.to(device)
    mask = mask.to(device)

    batch_size = y_obs.shape[0]

    starts = [
        torch.clamp(y_obs, 0.0, 1.0),
        torch.full_like(y_obs, 0.5),
    ]

    for _ in range(max(cfg.restarts - 2, 0)):
        starts.append(torch.rand_like(y_obs))

    x = torch.cat(starts, dim=0)
    y_rep = y_obs.repeat(len(starts), 1, 1, 1)
    mask_rep = mask.repeat(len(starts), 1, 1, 1)

    x = x.detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([x], lr=cfg.true_opt_lr)

    start_time = time.perf_counter()

    for _ in range(cfg.true_opt_steps):
        loss = true_energy(x, y_rep, mask_rep, cfg).sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x.clamp_(0.0, 1.0)

    elapsed = time.perf_counter() - start_time

    with torch.no_grad():
        values = true_energy(x, y_rep, mask_rep, cfg).view(len(starts), batch_size)
        xs = x.view(len(starts), batch_size, 1, cfg.image_size, cfg.image_size)

        best_idx = values.argmin(dim=0)

        x_best = xs[best_idx, torch.arange(batch_size, device=device)]
        v_best = values[best_idx, torch.arange(batch_size, device=device)]

    return x_best.detach(), v_best.detach(), elapsed


def freeze_model(model: nn.Module, frozen: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(not frozen)


def optimize_surrogate(
    model: nn.Module,
    ctx: torch.Tensor,
    y_obs: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    model.eval()
    freeze_model(model, True)

    ctx = ctx.to(device)
    y_obs = y_obs.to(device)

    batch_size = y_obs.shape[0]

    starts = [
        torch.clamp(y_obs, 0.0, 1.0),
        torch.full_like(y_obs, 0.5),
    ]

    for _ in range(max(cfg.restarts - 2, 0)):
        starts.append(torch.rand_like(y_obs))

    x = torch.cat(starts, dim=0)
    ctx_rep = ctx.repeat(len(starts), 1, 1, 1)

    x = x.detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([x], lr=cfg.surrogate_opt_lr)

    start_time = time.perf_counter()

    for _ in range(cfg.surrogate_opt_steps):
        loss = model(x, ctx_rep).sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x.clamp_(0.0, 1.0)

    elapsed = time.perf_counter() - start_time

    with torch.no_grad():
        values = model(x, ctx_rep).view(len(starts), batch_size)
        xs = x.view(len(starts), batch_size, 1, cfg.image_size, cfg.image_size)

        best_idx = values.argmin(dim=0)

        x_best = xs[best_idx, torch.arange(batch_size, device=device)]

    freeze_model(model, False)

    return x_best.detach(), elapsed


def downstream_eval(
    model: nn.Module,
    ctx_test: torch.Tensor,
    y_test: torch.Tensor,
    mask_test: torch.Tensor,
    x_star: torch.Tensor,
    v_star: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> Dict[str, float]:
    regrets = []
    decision_errors = []
    inference_times = []

    n_test = y_test.shape[0]

    for start in range(0, n_test, cfg.eval_batch_size):
        end = min(start + cfg.eval_batch_size, n_test)

        ctxb = ctx_test[start:end].to(device)
        yb = y_test[start:end].to(device)
        mb = mask_test[start:end].to(device)
        xsb = x_star[start:end].to(device)
        vsb = v_star[start:end].to(device)

        xhat, elapsed = optimize_surrogate(model, ctxb, yb, cfg, device)
        vhat = true_energy(xhat, yb, mb, cfg)

        regret = (vhat - vsb).detach().cpu()
        decision_error = torch.sqrt(
            (xhat - xsb).pow(2).mean(dim=(1, 2, 3))
        ).detach().cpu()

        regrets.append(regret)
        decision_errors.append(decision_error)
        inference_times.append(elapsed / yb.shape[0])

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

def plot_training_curves(
    histories: Dict[str, Dict[str, List[float]]],
    out_path: str,
) -> None:
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


def plot_restoration_examples(
    models: Dict[str, nn.Module],
    clean_test: torch.Tensor,
    ctx_test: torch.Tensor,
    y_test: torch.Tensor,
    mask_test: torch.Tensor,
    x_star: torch.Tensor,
    cfg: Config,
    device: torch.device,
    out_path: str,
) -> None:
    """
    Plot multiple downstream restoration examples in a grid.

    Rows: different test instances.
    Columns: Clean / Observed / True opt. / Conv-ReLU / Conv-Quad / Conv-Norm / Conv-SOC.

    The mask is omitted from the final paper figure because it visually dominates
    the restoration comparison. The mask construction is described in the text.
    """
    model_names = ["Conv-ReLU", "Conv-Quad", "Conv-Norm", "Conv-SOC"]

    indices = [idx for idx in cfg.plot_indices if idx < clean_test.shape[0]]

    if len(indices) < cfg.num_plot_examples:
        fallback = np.linspace(
            0,
            clean_test.shape[0] - 1,
            cfg.num_plot_examples,
            dtype=int,
        ).tolist()

        indices = []
        for idx in fallback:
            if idx not in indices:
                indices.append(idx)

    indices = indices[:cfg.num_plot_examples]

    ctx = ctx_test[indices].to(device)
    y = y_test[indices].to(device)

    restored_by_model = {}

    for name in model_names:
        if name not in models:
            continue

        xhat, _ = optimize_surrogate(models[name], ctx, y, cfg, device)
        restored_by_model[name] = xhat.detach().cpu()

    columns = ["Clean", "Observed", "True opt."] + [
        name for name in model_names if name in restored_by_model
    ]

    n_rows = len(indices)
    n_cols = len(columns)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.45 * n_cols, 1.45 * n_rows),
        squeeze=False,
    )

    for row_id, idx in enumerate(indices):
        image_map = {
            "Clean": clean_test[idx, 0].cpu().numpy(),
            "Observed": y_test[idx, 0].cpu().numpy(),
            "True opt.": x_star[idx, 0].cpu().numpy(),
        }

        for name in restored_by_model:
            image_map[name] = restored_by_model[name][row_id, 0].cpu().numpy()

        for col_id, title in enumerate(columns):
            ax = axes[row_id, col_id]
            ax.imshow(image_map[title], cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")

            if row_id == 0:
                ax.set_title(title, fontsize=8, pad=3)

    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.94,
        bottom=0.02,
        wspace=0.04,
        hspace=0.08,
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    base, ext = os.path.splitext(out_path)

    if ext.lower() != ".pdf":
        plt.savefig(base + ".pdf", bbox_inches="tight")

    if ext.lower() == ".pdf":
        plt.savefig(base + ".png", dpi=300, bbox_inches="tight")

    plt.close()


def write_latex_table(df: pd.DataFrame, out_path: str) -> None:
    order = ["Conv-ReLU", "Conv-Softplus", "Conv-Quad", "Conv-Norm", "Conv-SOC"]
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
        r"\caption{Convolutional downstream image restoration test on Olivetti Faces "
        r"\(64\times64\) grayscale images. Regret and decision error are evaluated "
        r"using the true convex restoration energy after optimizing each learned "
        r"convolutional surrogate. Lower is better for RelErr, regret, and decision error.}"
    )
    lines.append(r"\label{tab:cnn_downstream_real}")
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
    print("SOC-CNN downstream experiment on Olivetti Faces 64x64")
    print(f"Device: {device}")
    print(f"Output dir: {cfg.out_dir}")
    print("=" * 80)

    # ----------------------------
    # Load real images
    # ----------------------------
    clean_train, clean_val, clean_test = load_real_image_tensors(cfg)

    y_train, mask_train, ctx_train = make_corrupted_observation(
        clean_train, cfg, cfg.seed + 10, device
    )

    y_val, mask_val, ctx_val = make_corrupted_observation(
        clean_val, cfg, cfg.seed + 20, device
    )

    y_test, mask_test, ctx_test = make_corrupted_observation(
        clean_test, cfg, cfg.seed + 30, device
    )

    y_train = y_train.cpu()
    mask_train = mask_train.cpu()
    ctx_train = ctx_train.cpu()

    y_val = y_val.cpu()
    mask_val = mask_val.cpu()
    ctx_val = ctx_val.cpu()

    y_test = y_test.cpu()
    mask_test = mask_test.cpu()
    ctx_test = ctx_test.cpu()

    # ----------------------------
    # Build supervised candidate datasets
    # ----------------------------
    print("Building candidate datasets...")

    train_x, train_ctx, train_v = build_candidate_dataset(
        clean_train,
        y_train,
        mask_train,
        ctx_train,
        cfg,
        device,
        cfg.seed + 100,
    )

    val_x, val_ctx, val_v = build_candidate_dataset(
        clean_val,
        y_val,
        mask_val,
        ctx_val,
        cfg,
        device,
        cfg.seed + 200,
    )

    test_x, test_ctx, test_v = build_candidate_dataset(
        clean_test,
        y_test,
        mask_test,
        ctx_test,
        cfg,
        device,
        cfg.seed + 300,
    )

    train_v_std, val_v_std, test_v_std, energy_mean, energy_std = standardize_values(
        train_v,
        val_v,
        test_v,
    )

    train_ds = TensorDataset(train_x, train_ctx, train_v_std)
    val_ds = TensorDataset(val_x, val_ctx, val_v_std)
    test_ds = TensorDataset(test_x, test_ctx, test_v_std)

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

        relerr = evaluate_prediction_error(
            model,
            test_loader,
            energy_mean,
            energy_std,
            device,
        )

        print(f"[{name}] Test RelErr = {relerr:.4f}")

    plot_training_curves(
        histories,
        os.path.join(cfg.out_dir, "training_curves.png"),
    )

    # ----------------------------
    # Compute approximate true optima
    # ----------------------------
    print("\nComputing true optima for downstream regret...")

    x_star_list = []
    v_star_list = []
    true_times = []

    for start in range(0, clean_test.shape[0], cfg.eval_batch_size):
        end = min(start + cfg.eval_batch_size, clean_test.shape[0])

        yb = y_test[start:end]
        mb = mask_test[start:end]

        x_star_batch, v_star_batch, elapsed = optimize_true_energy(
            yb,
            mb,
            cfg,
            device,
        )

        x_star_list.append(x_star_batch.detach().cpu())
        v_star_list.append(v_star_batch.detach().cpu())
        true_times.append(elapsed / yb.shape[0])

        print(f"  true opt batch {start:04d}-{end:04d}")

    x_star = torch.cat(x_star_list, dim=0)
    v_star = torch.cat(v_star_list, dim=0)

    print(f"True optimizer runtime: {1000.0 * np.mean(true_times):.2f} ms / instance")

    # ----------------------------
    # Downstream evaluation
    # ----------------------------
    print("\nDownstream evaluation...")

    rows = []

    for name, model in models.items():
        relerr = evaluate_prediction_error(
            model,
            test_loader,
            energy_mean,
            energy_std,
            device,
        )

        metrics = downstream_eval(
            model=model,
            ctx_test=ctx_test,
            y_test=y_test,
            mask_test=mask_test,
            x_star=x_star,
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

    plot_restoration_examples(
        models=models,
        clean_test=clean_test,
        ctx_test=ctx_test,
        y_test=y_test,
        mask_test=mask_test,
        x_star=x_star,
        cfg=cfg,
        device=device,
        out_path=os.path.join(cfg.out_dir, "cnn_downstream_real_examples.pdf"),
    )

    print("\nSaved:")
    print(f"  {summary_csv}")
    print(f"  {summary_tex}")
    print(f"  {os.path.join(cfg.out_dir, 'cnn_downstream_real_examples.pdf')}")
    print(f"  {os.path.join(cfg.out_dir, 'cnn_downstream_real_examples.png')}")
    print(f"  {os.path.join(cfg.out_dir, 'training_curves.png')}")
    print(f"  {os.path.join(cfg.out_dir, 'training_curves.pdf')}")


if __name__ == "__main__":
    main()