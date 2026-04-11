# -*- coding: utf-8 -*-
"""
Supplementary convex baselines on the fair Experiment-3 dataset.
Uses the same train/test splits and the same decision protocol as exp3_train_fair.py.
"""

import copy
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from exp3_train_fair import (
    setup_console_encoding,
    set_seed,
    ensure_dir,
    count_parameters,
    load_npz_dict,
    split_instance_dict,
    subset_instance_dict,
    flatten_supervised_arrays,
    Exp3FlatDataset,
    evaluate_regression,
    get_task_family,
    get_feasible_type,
    get_budget,
    normalize_scalar,
    initial_feasible_points,
    project_feasible_torch,
    true_objective_single,
    feasibility_violation_single,
    save_summary_csv,
    save_all_results_json,
    aggregate_rows,
    print_group_comparison,
)

setup_console_encoding()


class PCFModel(nn.Module):
    def __init__(self, x_dim: int, theta_dim: int, n_norm_atoms: int = 2, norm_atom_dim: int = 8):
        super().__init__()
        self.c_head = nn.Linear(theta_dim, x_dim)
        self.b_head = nn.Linear(theta_dim, 1)
        self.m_head = nn.Linear(theta_dim, x_dim)
        self.q_head = nn.Linear(theta_dim, x_dim)

        self.norm_x = nn.ModuleList()
        self.norm_t = nn.ModuleList()
        self.lam_head = nn.ModuleList()
        for _ in range(n_norm_atoms):
            lx = nn.Linear(x_dim, norm_atom_dim, bias=False)
            lt = nn.Linear(theta_dim, norm_atom_dim, bias=True)
            ll = nn.Linear(theta_dim, 1, bias=True)
            nn.init.xavier_uniform_(lx.weight)
            nn.init.xavier_uniform_(lt.weight)
            nn.init.zeros_(lt.bias)
            nn.init.xavier_uniform_(ll.weight)
            nn.init.zeros_(ll.bias)
            self.norm_x.append(lx)
            self.norm_t.append(lt)
            self.lam_head.append(ll)

        for mod in [self.c_head, self.b_head, self.m_head, self.q_head]:
            nn.init.xavier_uniform_(mod.weight)
            nn.init.zeros_(mod.bias)

    def forward(self, theta, x):
        c = self.c_head(theta)
        b = self.b_head(theta).squeeze(-1)
        m = self.m_head(theta)
        q = F.softplus(self.q_head(theta)) + 1e-8
        y = torch.sum(c * x, dim=-1) + b + 0.5 * torch.sum(q * (x - m) ** 2, dim=-1)
        for lx, lt, ll in zip(self.norm_x, self.norm_t, self.lam_head):
            u = lx(x) + lt(theta)
            lam = F.softplus(ll(theta)).squeeze(-1) + 1e-8
            y = y + lam * torch.norm(u, dim=-1, p=2)
        return y


class DCPModel(nn.Module):
    def __init__(self, x_dim: int, theta_dim: int, n_sp_atoms: int = 8, n_sq_atoms: int = 8,
                 n_norm_atoms: int = 2, norm_atom_dim: int = 8):
        super().__init__()
        self.c_head = nn.Linear(theta_dim, x_dim)
        self.b_head = nn.Linear(theta_dim, 1)
        nn.init.xavier_uniform_(self.c_head.weight)
        nn.init.zeros_(self.c_head.bias)
        nn.init.xavier_uniform_(self.b_head.weight)
        nn.init.zeros_(self.b_head.bias)

        self.sp_x, self.sp_t, self.sp_w = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(n_sp_atoms):
            lx = nn.Linear(x_dim, 1, bias=False)
            lt = nn.Linear(theta_dim, 1, bias=True)
            lw = nn.Linear(theta_dim, 1, bias=True)
            nn.init.xavier_uniform_(lx.weight)
            nn.init.xavier_uniform_(lt.weight)
            nn.init.zeros_(lt.bias)
            nn.init.xavier_uniform_(lw.weight)
            nn.init.zeros_(lw.bias)
            self.sp_x.append(lx)
            self.sp_t.append(lt)
            self.sp_w.append(lw)

        self.sq_x, self.sq_t, self.sq_w = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(n_sq_atoms):
            lx = nn.Linear(x_dim, 1, bias=False)
            lt = nn.Linear(theta_dim, 1, bias=True)
            lw = nn.Linear(theta_dim, 1, bias=True)
            nn.init.xavier_uniform_(lx.weight)
            nn.init.xavier_uniform_(lt.weight)
            nn.init.zeros_(lt.bias)
            nn.init.xavier_uniform_(lw.weight)
            nn.init.zeros_(lw.bias)
            self.sq_x.append(lx)
            self.sq_t.append(lt)
            self.sq_w.append(lw)

        self.norm_x, self.norm_t, self.norm_w = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(n_norm_atoms):
            lx = nn.Linear(x_dim, norm_atom_dim, bias=False)
            lt = nn.Linear(theta_dim, norm_atom_dim, bias=True)
            lw = nn.Linear(theta_dim, 1, bias=True)
            nn.init.xavier_uniform_(lx.weight)
            nn.init.xavier_uniform_(lt.weight)
            nn.init.zeros_(lt.bias)
            nn.init.xavier_uniform_(lw.weight)
            nn.init.zeros_(lw.bias)
            self.norm_x.append(lx)
            self.norm_t.append(lt)
            self.norm_w.append(lw)

    def forward(self, theta, x):
        y = torch.sum(self.c_head(theta) * x, dim=-1) + self.b_head(theta).squeeze(-1)
        for lx, lt, lw in zip(self.sp_x, self.sp_t, self.sp_w):
            rho = F.softplus(lw(theta)).squeeze(-1) + 1e-8
            atom = F.softplus((lx(x) + lt(theta)).squeeze(-1))
            y = y + rho * atom
        for lx, lt, lw in zip(self.sq_x, self.sq_t, self.sq_w):
            alpha = F.softplus(lw(theta)).squeeze(-1) + 1e-8
            affine = (lx(x) + lt(theta)).squeeze(-1)
            y = y + 0.5 * alpha * affine ** 2
        for lx, lt, lw in zip(self.norm_x, self.norm_t, self.norm_w):
            lam = F.softplus(lw(theta)).squeeze(-1) + 1e-8
            y = y + lam * torch.norm(lx(x) + lt(theta), dim=-1, p=2)
        return y


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        theta = batch["theta"].to(device)
        x = batch["x"].to(device)
        y = batch["y"].to(device).view(-1)
        optimizer.zero_grad()
        pred = model(theta, x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        bs = theta.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs
    return total_loss / max(total_n, 1)


@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        theta = batch["theta"].to(device)
        x = batch["x"].to(device)
        y = batch["y"].to(device).view(-1)
        pred = model(theta, x)
        loss = F.mse_loss(pred, y)
        bs = theta.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs
    return total_loss / max(total_n, 1)



def solve_learned_decision(model, theta_np, feasible_type, dim, device, budget=None, n_restart=8, n_steps=600, lr=3e-3):
    model.eval()
    theta = torch.tensor(theta_np, dtype=torch.float32, device=device).view(1, -1)
    theta_batch = theta.repeat(n_restart, 1)
    x = initial_feasible_points(n_restart, dim, feasible_type, device, budget).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)

    best_x = None
    best_val = None
    for _ in range(n_steps):
        optimizer.zero_grad()
        vals = model(theta_batch, x)
        vals.sum().backward()
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



def evaluate_decision_metrics(model, data, device, n_restart=8, n_steps=600, lr=3e-3):
    feasible_type = get_feasible_type(data)
    dim = int(normalize_scalar(data["dim"]))
    budget = get_budget(data)
    contexts = data["contexts"]
    x_star = data["x_star"]
    f_star = data["f_star"]

    regrets, rel_regrets, x_errs, feas_viol, solve_times = [], [], [], [], []
    for i in range(contexts.shape[0]):
        t0 = time.perf_counter()
        x_hat, _ = solve_learned_decision(model, contexts[i], feasible_type, dim, device, budget, n_restart, n_steps, lr)
        dt = time.perf_counter() - t0
        f_hat_true = true_objective_single(data, i, x_hat)
        reg = float(f_hat_true - f_star[i])
        regrets.append(reg)
        rel_regrets.append(float(reg / (abs(float(f_star[i])) + 1e-8)))
        x_errs.append(float(((x_hat - x_star[i]) ** 2).sum() ** 0.5))
        feas_viol.append(feasibility_violation_single(data, x_hat))
        solve_times.append(dt)

    import numpy as np
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



def make_suppl_model(model_name: str, x_dim: int, theta_dim: int, cfg: Dict) -> nn.Module:
    if model_name == "pcf":
        mc = cfg["pcf"]
        return PCFModel(x_dim, theta_dim, mc["n_norm_atoms"], mc["norm_atom_dim"])
    if model_name == "dcp":
        mc = cfg["dcp"]
        return DCPModel(x_dim, theta_dim, mc["n_sp_atoms"], mc["n_sq_atoms"], mc["n_norm_atoms"], mc["norm_atom_dim"])
    raise ValueError(model_name)



def run_single_experiment(global_cfg: Dict, task: str, dim: int, model_name: str, seed: int) -> Dict:
    device = torch.device(global_cfg["device"])
    data_root = Path(global_cfg["data_root"])
    exp_dir = Path(global_cfg["save_root"]) / task / f"d{dim}" / model_name / f"seed{seed}"
    ensure_dir(exp_dir)

    raw_train = load_npz_dict(str(data_root / task / f"d{dim}" / "train.npz"))
    raw_test = load_npz_dict(str(data_root / task / f"d{dim}" / "test.npz"))
    raw_subtrain, raw_val = split_instance_dict(raw_train, val_ratio=global_cfg["train"]["val_ratio"], seed=seed)
    raw_test_for_decision = subset_instance_dict(raw_test, global_cfg["decision_eval"]["max_test_instances"], seed=seed)

    arr_subtrain = flatten_supervised_arrays(raw_subtrain)
    arr_val = flatten_supervised_arrays(raw_val)
    arr_test = flatten_supervised_arrays(raw_test)

    y_mean = float(arr_subtrain["y"].mean())
    y_std = max(float(arr_subtrain["y"].std()), 1e-8)

    ds_subtrain = Exp3FlatDataset(arr_subtrain, y_mean=y_mean, y_std=y_std)
    ds_val = Exp3FlatDataset(arr_val, y_mean=y_mean, y_std=y_std)
    ds_test = Exp3FlatDataset(arr_test, y_mean=y_mean, y_std=y_std)

    dl_subtrain = DataLoader(ds_subtrain, batch_size=global_cfg["train"]["batch_size"], shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=global_cfg["train"]["batch_size"], shuffle=False, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=global_cfg["train"]["batch_size"], shuffle=False, num_workers=0)

    theta_dim = raw_train["contexts"].shape[1]
    model = make_suppl_model(model_name, dim, theta_dim, global_cfg["model_structure"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=global_cfg["train"]["lr"], weight_decay=global_cfg["train"]["weight_decay"])

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    patience_count = 0
    log_rows = []
    total_t0 = time.perf_counter()

    print("=" * 100)
    print(f"Start | task={task:>20s} | dim={dim:<3d} | model={model_name:<8s} | seed={seed} | params={count_parameters(model)}")

    for epoch in range(1, global_cfg["train"]["epochs"] + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, dl_subtrain, optimizer, device)
        val_loss = validate_one_epoch(model, dl_val, device)
        val_reg = evaluate_regression(model, dl_val, device, y_mean=y_mean, y_std=y_std)
        dt = time.perf_counter() - t0
        log_rows.append({
            "epoch": epoch,
            "train_mse_norm": train_loss,
            "val_mse_norm": val_loss,
            "val_mse_raw": val_reg["mse"],
            "val_rmse_raw": val_reg["rmse"],
            "val_mae_raw": val_reg["mae"],
            "val_rel_l2": val_reg["rel_l2"],
            "epoch_time_sec": dt,
        })
        print(f"[Epoch {epoch:03d}] train={train_loss:.6f} | val={val_loss:.6f} | val_rel_l2={val_reg['rel_l2']:.6f} | time={dt:.2f}s")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= global_cfg["train"]["patience"]:
            break

    total_train_time = time.perf_counter() - total_t0
    model.load_state_dict(best_state)
    torch.save(best_state, exp_dir / "best_model.pt")

    with open(exp_dir / "train_log.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    train_reg = evaluate_regression(model, dl_subtrain, device, y_mean=y_mean, y_std=y_std)
    val_reg = evaluate_regression(model, dl_val, device, y_mean=y_mean, y_std=y_std)
    test_reg = evaluate_regression(model, dl_test, device, y_mean=y_mean, y_std=y_std)
    test_decision = evaluate_decision_metrics(model, raw_test_for_decision, device,
                                              global_cfg["decision_eval"]["n_restart"],
                                              global_cfg["decision_eval"]["n_steps"],
                                              global_cfg["decision_eval"]["lr"])

    metrics = {
        "config": {
            "task": task,
            "task_family": get_task_family(raw_train),
            "dim": dim,
            "model": model_name,
            "seed": seed,
            "num_parameters": count_parameters(model),
            "best_epoch": best_epoch,
            "total_train_time_sec": total_train_time,
        },
        "train_regression": train_reg,
        "val_regression": val_reg,
        "test_regression": test_reg,
        "test_decision": test_decision,
    }
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return {
        "task": task,
        "task_family": get_task_family(raw_train),
        "feasible_type": get_feasible_type(raw_train),
        "dim": dim,
        "model": model_name,
        "seed": seed,
        "num_parameters": count_parameters(model),
        "target_soc_params": count_parameters(model),
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



def build_config() -> Dict:
    use_cuda = torch.cuda.is_available()
    return {
        "device": "cuda" if use_cuda else "cpu",
        "data_root": "exp3_convex_fair_data",
        "save_root": "exp3_suppl_runs_fair",
        "tasks": [
            "simplex_socp",
            "box_socp",
            "budget_twocone_socp",
            "simplex_logistic",
            "box_logsumexp",
            "budget_huber",
        ],
        "dims": [10, 20, 50],
        "models": ["pcf", "dcp"],
        "seeds": [2024, 2025, 2026],
        "train": {
            "epochs": 120,
            "batch_size": 1024,
            "lr": 1e-3,
            "weight_decay": 1e-6,
            "patience": 20,
            "val_ratio": 0.1,
        },
        "decision_eval": {
            "n_restart": 5,
            "n_steps": 200,
            "lr": 3e-3,
            "max_test_instances": 20,
        },
        "model_structure": {
            "pcf": {"n_norm_atoms": 2, "norm_atom_dim": 12},
            "dcp": {"n_sp_atoms": 12, "n_sq_atoms": 12, "n_norm_atoms": 2, "norm_atom_dim": 12},
        },
    }



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
                        "num_parameters", "train_time_sec", "train_rel_l2", "val_rel_l2", "test_rel_l2",
                        "test_rmse", "test_mae", "regret_mean", "regret_median", "rel_regret_mean",
                        "x_error_mean", "feas_violation_mean", "decision_time_mean_sec",
                    ]
                    agg_rows = aggregate_rows(all_rows, metric_keys)
                    save_summary_csv(agg_rows, save_root / "summary_aggregated.csv")
                    save_all_results_json(agg_rows, save_root / "all_results_aggregated.json")

    metric_keys = [
        "num_parameters", "train_time_sec", "train_rel_l2", "val_rel_l2", "test_rel_l2",
        "test_rmse", "test_mae", "regret_mean", "regret_median", "rel_regret_mean",
        "x_error_mean", "feas_violation_mean", "decision_time_mean_sec",
    ]
    agg_rows = aggregate_rows(all_rows, metric_keys)
    print_group_comparison(agg_rows, sort_key="regret_mean_mean")

    print("\nAll supplementary experiments finished.")
    print(f"Aggregated CSV   : {(save_root / 'summary_aggregated.csv').resolve()}")


if __name__ == "__main__":
    main()
