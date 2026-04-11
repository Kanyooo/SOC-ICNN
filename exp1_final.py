#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 1: Empirical verification of full SOC-ICNN value-function equivalence.

What this experiment checks
---------------------------
We test the FULL SOC-ICNN with three parallel branches:
    1) ReLU backbone branch
    2) Quad branch   : alpha/2 * ||B x + e||_2^2
    3) Norm branch   : lambda * ||A x + d||_2

For a fixed input x, we report:
    - Closed-form primal/dual gap:
          | f_forward(x) - f_closed_dual(x) |
      This checks internal value-function consistency.

    - External solver absolute error:
          | f_solver_primal(x) - f_forward(x) |
      This is the main Max Abs Error against the lifted SOCP/QCP primal model.

    - ReLU branch residuals:
          * primal feasibility violation
          * dual box feasibility violation
          * complementarity residual

    - Quad branch residuals:
          * epigraph feasibility violation
          * tightness residual

    - Norm branch residuals:
          * epigraph feasibility violation
          * tightness residual

We run only ONE large-scale configuration, with a passthrough on/off switch.
Results are:
    (i) printed to console
    (ii) saved to CSV (detailed trial table + grouped summary table)

Notes
-----
- "Passthrough = True" means every hidden layer receives a direct affine term W_l x.
- "Passthrough = False" means only the first hidden layer sees x directly; deeper layers use U_l z_{l-1} + b_l.
- If Gurobi is not available, solver-based columns are saved as NaN, but all closed-form checks still run.
"""

import math
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", r"All-NaN slice encountered")
warnings.filterwarnings("ignore", r"Mean of empty slice")

torch.set_default_dtype(torch.float64)

# =========================
# Optional external solver
# =========================
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except Exception as e:
    GUROBI_AVAILABLE = False
    print("[WARN] gurobipy import failed. Solver-based Max Abs Error will be skipped.")
    print(e)


# =========================
# Config
# =========================
@dataclass
class ExpConfig:
    input_dim: int = 100
    hidden_dim: int = 256
    depth: int = 6

    n_quad: int = 2
    quad_dim: int = 24
    n_norm: int = 2
    norm_dim: int = 24

    n_seeds: int = 5
    trials_per_seed: int = 30
    base_seed: int = 2026

    quad_alpha_init: float = 1.5
    norm_lambda_init: float = 2.0

    passthrough_settings: Tuple[bool, ...] = (True, False)

    run_solver_check: bool = True
    solver_time_limit_sec: float = 20.0

    detail_csv: str = "exp1_soc_icnn_equivalence_detail.csv"
    summary_csv: str = "exp1_soc_icnn_equivalence_summary.csv"


# =========================
# Full SOC-ICNN
# =========================
class FullSOCICNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        passthrough: bool,
        n_quad: int,
        quad_dim: int,
        n_norm: int,
        norm_dim: int,
        quad_alpha_init: float,
        norm_lambda_init: float,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.passthrough = passthrough
        self.n_quad = n_quad
        self.quad_dim = quad_dim
        self.n_norm = n_norm
        self.norm_dim = norm_dim

        scale_w = 1.0 / math.sqrt(max(1, input_dim))
        scale_u = 2.0 / max(1, hidden_dim)

        self.W1 = nn.Parameter(torch.randn(hidden_dim, input_dim) * scale_w)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))

        self.layers_W = nn.ParameterList()
        self.layers_U = nn.ParameterList()
        self.layers_b = nn.ParameterList()
        for _ in range(depth - 1):
            W = nn.Parameter(torch.randn(hidden_dim, input_dim) * scale_w)
            U = nn.Parameter(torch.empty(hidden_dim, hidden_dim).uniform_(0.0, scale_u))
            b = nn.Parameter(torch.zeros(hidden_dim))
            self.layers_W.append(W)
            self.layers_U.append(U)
            self.layers_b.append(b)

        self.c = nn.Parameter(torch.randn(hidden_dim))
        self.v = nn.Parameter(torch.randn(input_dim) * scale_w)
        self.b0 = nn.Parameter(torch.zeros(1))

        self.B_quad = nn.ParameterList()
        self.e_quad = nn.ParameterList()
        self.alpha_quad_raw = nn.ParameterList()
        for _ in range(n_quad):
            self.B_quad.append(nn.Parameter(torch.randn(quad_dim, input_dim) * scale_w))
            self.e_quad.append(nn.Parameter(torch.zeros(quad_dim)))
            self.alpha_quad_raw.append(nn.Parameter(torch.tensor(float(quad_alpha_init))))

        self.A_norm = nn.ParameterList()
        self.d_norm = nn.ParameterList()
        self.lambda_norm_raw = nn.ParameterList()
        for _ in range(n_norm):
            self.A_norm.append(nn.Parameter(torch.randn(norm_dim, input_dim) * scale_w))
            self.d_norm.append(nn.Parameter(torch.zeros(norm_dim)))
            self.lambda_norm_raw.append(nn.Parameter(torch.tensor(float(norm_lambda_init))))

    def c_nonneg(self) -> torch.Tensor:
        return torch.abs(self.c)

    def alpha_quad(self) -> List[torch.Tensor]:
        return [torch.abs(a) for a in self.alpha_quad_raw]

    def lambda_norm(self) -> List[torch.Tensor]:
        return [torch.abs(l) for l in self.lambda_norm_raw]

    def forward_closed_form(self, x: torch.Tensor) -> Dict[str, object]:
        zs: List[torch.Tensor] = []
        preacts: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []

        a0 = x @ self.W1.T + self.b1
        z0 = torch.relu(a0)
        m0 = (a0 > 0).double()
        zs.append(z0)
        preacts.append(a0)
        masks.append(m0)

        for i in range(self.depth - 1):
            U = self.layers_U[i]
            b = self.layers_b[i]
            if self.passthrough:
                a = x @ self.layers_W[i].T + zs[-1] @ U.T + b
            else:
                a = zs[-1] @ U.T + b
            z = torch.relu(a)
            m = (a > 0).double()
            zs.append(z)
            preacts.append(a)
            masks.append(m)

        relu_val = zs[-1] @ self.c_nonneg() + x @ self.v + self.b0

        quad_qs, quad_ss, quad_dual_y, quad_vals = [], [], [], []
        for B, e, alpha in zip(self.B_quad, self.e_quad, self.alpha_quad()):
            q = x @ B.T + e
            s = 0.5 * torch.sum(q * q)
            y = alpha * q
            val = alpha * s
            quad_qs.append(q)
            quad_ss.append(s)
            quad_dual_y.append(y)
            quad_vals.append(val)

        norm_us, norm_ts, norm_dual_r, norm_vals = [], [], [], []
        for A, d, lam in zip(self.A_norm, self.d_norm, self.lambda_norm()):
            u = x @ A.T + d
            t = torch.norm(u, p=2)
            if t.item() > 1e-12:
                r = lam * (u / t)
            else:
                r = torch.zeros_like(u)
            val = lam * t
            norm_us.append(u)
            norm_ts.append(t)
            norm_dual_r.append(r)
            norm_vals.append(val)

        total_val = relu_val
        if quad_vals:
            total_val = total_val + torch.stack(quad_vals).sum()
        if norm_vals:
            total_val = total_val + torch.stack(norm_vals).sum()

        return {
            "forward_val": total_val.squeeze(),
            "zs": zs,
            "preacts": preacts,
            "masks": masks,
            "quad_qs": quad_qs,
            "quad_ss": quad_ss,
            "quad_dual_y": quad_dual_y,
            "norm_us": norm_us,
            "norm_ts": norm_ts,
            "norm_dual_r": norm_dual_r,
        }

    def closed_form_dual(self, x: torch.Tensor, cache: Dict[str, object]) -> Dict[str, object]:
        zs = cache["zs"]
        masks = cache["masks"]
        depth = self.depth
        c = self.c_nonneg()

        nus: List[Optional[torch.Tensor]] = [None] * depth
        mus: List[Optional[torch.Tensor]] = [None] * depth
        nus[-1] = c * masks[-1]
        box_viol = 0.0

        for i in range(depth - 2, -1, -1):
            upper = nus[i + 1] @ self.layers_U[i]
            nus[i] = upper * masks[i]
            box_viol = max(box_viol, torch.clamp(nus[i] - upper, min=0.0).max().item())
            mus[i] = torch.clamp(upper - nus[i], min=0.0)
        mus[-1] = torch.clamp(c - nus[-1], min=0.0)

        dual_val = x @ self.v + self.b0
        dual_val = dual_val + torch.sum(nus[0] * (x @ self.W1.T + self.b1))
        for i in range(depth - 1):
            const_part = self.layers_b[i].clone()
            if self.passthrough:
                const_part = const_part + x @ self.layers_W[i].T
            dual_val = dual_val + torch.sum(nus[i + 1] * const_part)

        quad_dual_consistency = 0.0
        for q, y, alpha in zip(cache["quad_qs"], cache["quad_dual_y"], self.alpha_quad()):
            if alpha.item() > 1e-12:
                dual_val = dual_val + torch.dot(y, q) - 0.5 / alpha * torch.sum(y * y)
            quad_dual_consistency = max(quad_dual_consistency, torch.norm(y - alpha * q, p=2).item())

        norm_ball_viol = 0.0
        norm_align_viol = 0.0
        for u, r, lam in zip(cache["norm_us"], cache["norm_dual_r"], self.lambda_norm()):
            dual_val = dual_val + torch.dot(r, u)
            norm_ball_viol = max(norm_ball_viol, max(torch.norm(r, p=2).item() - lam.item(), 0.0))
            u_norm = torch.norm(u, p=2).item()
            if u_norm > 1e-12:
                target = lam * (u / torch.norm(u, p=2))
                norm_align_viol = max(norm_align_viol, torch.norm(r - target, p=2).item())
            else:
                norm_align_viol = max(norm_align_viol, torch.norm(r, p=2).item())

        return {
            "dual_val": dual_val.squeeze(),
            "nus": nus,
            "mus": mus,
            "relu_dual_box_viol": box_viol,
            "quad_dual_consistency": quad_dual_consistency,
            "norm_dual_ball_viol": norm_ball_viol,
            "norm_dual_align_viol": norm_align_viol,
        }

    def branch_residuals_from_closed_form(self, x: torch.Tensor, cache: Dict[str, object], dual_cache: Dict[str, object]) -> Dict[str, float]:
        zs = cache["zs"]
        nus = dual_cache["nus"]
        mus = dual_cache["mus"]

        relu_primal_viol = 0.0
        relu_comp = 0.0
        a0 = x @ self.W1.T + self.b1
        z0 = zs[0]
        relu_primal_viol = max(relu_primal_viol, torch.clamp(a0 - z0, min=0.0).max().item(), torch.clamp(-z0, min=0.0).max().item())
        relu_comp = max(relu_comp, torch.abs(nus[0] * (z0 - a0)).max().item(), torch.abs(mus[0] * z0).max().item())

        for i in range(1, self.depth):
            b = self.layers_b[i - 1]
            if self.passthrough:
                a = x @ self.layers_W[i - 1].T + zs[i - 1] @ self.layers_U[i - 1].T + b
            else:
                a = zs[i - 1] @ self.layers_U[i - 1].T + b
            z = zs[i]
            relu_primal_viol = max(relu_primal_viol, torch.clamp(a - z, min=0.0).max().item(), torch.clamp(-z, min=0.0).max().item())
            relu_comp = max(relu_comp, torch.abs(nus[i] * (z - a)).max().item(), torch.abs(mus[i] * z).max().item())

        quad_epi_viol = 0.0
        quad_tight = 0.0
        for q, s in zip(cache["quad_qs"], cache["quad_ss"]):
            rhs = 0.5 * torch.sum(q * q).item()
            lhs = s.item()
            quad_epi_viol = max(quad_epi_viol, max(rhs - lhs, 0.0))
            quad_tight = max(quad_tight, abs(lhs - rhs))

        norm_epi_viol = 0.0
        norm_tight = 0.0
        for u, t in zip(cache["norm_us"], cache["norm_ts"]):
            rhs = torch.norm(u, p=2).item()
            lhs = t.item()
            norm_epi_viol = max(norm_epi_viol, max(rhs - lhs, 0.0))
            norm_tight = max(norm_tight, abs(lhs - rhs))

        return {
            "ReLUPrimalViol": relu_primal_viol,
            "ReLUDualBoxViol": float(dual_cache["relu_dual_box_viol"]),
            "ReLUCompSlack": relu_comp,
            "QuadEpiViol": quad_epi_viol,
            "QuadTight": quad_tight,
            "NormEpiViol": norm_epi_viol,
            "NormTight": norm_tight,
            "NormDualBallViol": float(dual_cache["norm_dual_ball_viol"]),
            "NormDualAlignViol": float(dual_cache["norm_dual_align_viol"]),
            "QuadDualConsistency": float(dual_cache["quad_dual_consistency"]),
        }


def solve_primal_qcp_with_gurobi(model: FullSOCICNN, x: torch.Tensor, time_limit: Optional[float] = None) -> Dict[str, object]:
    if not GUROBI_AVAILABLE:
        return {"status": "NO_GUROBI", "obj_val": float("nan"), "runtime_sec": float("nan"), "z_vals": None, "s_vals": None, "q_vals": None, "t_vals": None, "u_vals": None}

    xnp = x.detach().cpu().numpy()
    hidden_dim = model.hidden_dim
    depth = model.depth

    W1 = model.W1.detach().cpu().numpy()
    b1 = model.b1.detach().cpu().numpy()
    Ws = [W.detach().cpu().numpy() for W in model.layers_W]
    Us = [U.detach().cpu().numpy() for U in model.layers_U]
    bs = [b.detach().cpu().numpy() for b in model.layers_b]
    c = np.abs(model.c.detach().cpu().numpy())
    v = model.v.detach().cpu().numpy()
    b0 = float(model.b0.detach().cpu().numpy().reshape(-1)[0])

    B_list = [B.detach().cpu().numpy() for B in model.B_quad]
    e_list = [e.detach().cpu().numpy() for e in model.e_quad]
    alpha_list = [float(torch.abs(a).item()) for a in model.alpha_quad_raw]

    A_list = [A.detach().cpu().numpy() for A in model.A_norm]
    d_list = [d.detach().cpu().numpy() for d in model.d_norm]
    lambda_list = [float(torch.abs(l).item()) for l in model.lambda_norm_raw]

    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.NumericFocus = 2
    if time_limit is not None:
        m.Params.TimeLimit = time_limit

    z_vars = [m.addVars(hidden_dim, lb=0.0, name=f"z_{l}") for l in range(depth)]
    for i in range(hidden_dim):
        m.addConstr(z_vars[0][i] >= float(W1[i, :].dot(xnp) + b1[i]))

    for l in range(1, depth):
        for i in range(hidden_dim):
            expr = gp.LinExpr(float(bs[l - 1][i]))
            if model.passthrough:
                expr += float(Ws[l - 1][i, :].dot(xnp))
            expr += gp.quicksum(float(Us[l - 1][i, k]) * z_vars[l - 1][k] for k in range(hidden_dim))
            m.addConstr(z_vars[l][i] >= expr)

    obj = gp.LinExpr(0.0)
    obj += gp.quicksum(float(c[i]) * z_vars[-1][i] for i in range(hidden_dim))
    const_term = float(v.dot(xnp) + b0)

    q_vars_all, s_vars_all = [], []
    for h, (B, e, alpha) in enumerate(zip(B_list, e_list, alpha_list)):
        q_const = B.dot(xnp) + e
        q_dim = q_const.shape[0]
        q_vars = m.addVars(q_dim, lb=-GRB.INFINITY, name=f"q_{h}")
        s_var = m.addVar(lb=0.0, name=f"s_{h}")
        for j in range(q_dim):
            m.addConstr(q_vars[j] == float(q_const[j]))
        qlhs = gp.QuadExpr()
        for j in range(q_dim):
            qlhs += q_vars[j] * q_vars[j]
        m.addQConstr(qlhs <= 2.0 * s_var)
        obj += alpha * s_var
        q_vars_all.append(q_vars)
        s_vars_all.append(s_var)

    u_vars_all, t_vars_all = [], []
    for g, (A, d, lam) in enumerate(zip(A_list, d_list, lambda_list)):
        u_const = A.dot(xnp) + d
        u_dim = u_const.shape[0]
        u_vars = m.addVars(u_dim, lb=-GRB.INFINITY, name=f"u_{g}")
        t_var = m.addVar(lb=0.0, name=f"t_{g}")
        for j in range(u_dim):
            m.addConstr(u_vars[j] == float(u_const[j]))
        qlhs = gp.QuadExpr()
        for j in range(u_dim):
            qlhs += u_vars[j] * u_vars[j]
        m.addQConstr(qlhs <= t_var * t_var)
        obj += lam * t_var
        u_vars_all.append(u_vars)
        t_vars_all.append(t_var)

    m.setObjective(obj, GRB.MINIMIZE)
    t0 = time.perf_counter()
    m.optimize()
    t1 = time.perf_counter()

    if m.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        return {"status": f"STATUS_{m.Status}", "obj_val": float("nan"), "runtime_sec": t1 - t0, "z_vals": None, "s_vals": None, "q_vals": None, "t_vals": None, "u_vals": None}

    z_vals = [np.array([z_vars[l][i].X for i in range(hidden_dim)], dtype=float) for l in range(depth)]
    s_vals = [float(s_vars_all[h].X) for h in range(len(s_vars_all))]
    q_vals = [np.array([q_vars_all[h][j].X for j in range(model.quad_dim)], dtype=float) for h in range(len(q_vars_all))]
    t_vals = [float(t_vars_all[g].X) for g in range(len(t_vars_all))]
    u_vals = [np.array([u_vars_all[g][j].X for j in range(model.norm_dim)], dtype=float) for g in range(len(u_vars_all))]

    return {"status": "OK", "obj_val": float(m.ObjVal + const_term), "runtime_sec": t1 - t0, "z_vals": z_vals, "s_vals": s_vals, "q_vals": q_vals, "t_vals": t_vals, "u_vals": u_vals}


def branch_residuals_from_solver_solution(model: FullSOCICNN, x: torch.Tensor, solver_out: Dict[str, object]) -> Dict[str, float]:
    if solver_out["z_vals"] is None:
        return {"Solver_ReLUPrimalViol": float("nan"), "Solver_QuadEpiViol": float("nan"), "Solver_QuadTight": float("nan"), "Solver_NormEpiViol": float("nan"), "Solver_NormTight": float("nan")}

    xnp = x.detach().cpu().numpy()
    z_vals = solver_out["z_vals"]
    s_vals = solver_out["s_vals"]
    q_vals = solver_out["q_vals"]
    t_vals = solver_out["t_vals"]
    u_vals = solver_out["u_vals"]

    relu_viol = 0.0
    a0 = (x @ model.W1.T + model.b1).detach().cpu().numpy()
    relu_viol = max(relu_viol, np.max(np.maximum(a0 - z_vals[0], 0.0)), np.max(np.maximum(-z_vals[0], 0.0)))
    for l in range(1, model.depth):
        rhs = z_vals[l - 1].dot(model.layers_U[l - 1].detach().cpu().numpy().T) + model.layers_b[l - 1].detach().cpu().numpy()
        if model.passthrough:
            rhs = rhs + xnp.dot(model.layers_W[l - 1].detach().cpu().numpy().T)
        relu_viol = max(relu_viol, np.max(np.maximum(rhs - z_vals[l], 0.0)), np.max(np.maximum(-z_vals[l], 0.0)))

    quad_epi_viol = 0.0
    quad_tight = 0.0
    for q, s in zip(q_vals, s_vals):
        rhs = 0.5 * float(np.dot(q, q))
        quad_epi_viol = max(quad_epi_viol, max(rhs - s, 0.0))
        quad_tight = max(quad_tight, abs(s - rhs))

    norm_epi_viol = 0.0
    norm_tight = 0.0
    for u, t in zip(u_vals, t_vals):
        rhs = float(np.linalg.norm(u, ord=2))
        norm_epi_viol = max(norm_epi_viol, max(rhs - t, 0.0))
        norm_tight = max(norm_tight, abs(t - rhs))

    return {"Solver_ReLUPrimalViol": relu_viol, "Solver_QuadEpiViol": quad_epi_viol, "Solver_QuadTight": quad_tight, "Solver_NormEpiViol": norm_epi_viol, "Solver_NormTight": norm_tight}


def run_single_trial(cfg: ExpConfig, passthrough: bool, seed: int, trial_id: int) -> Dict[str, object]:
    torch.manual_seed(seed)
    model = FullSOCICNN(cfg.input_dim, cfg.hidden_dim, cfg.depth, passthrough, cfg.n_quad, cfg.quad_dim, cfg.n_norm, cfg.norm_dim, cfg.quad_alpha_init, cfg.norm_lambda_init)
    x = torch.randn(cfg.input_dim)

    t0 = time.perf_counter()
    cache = model.forward_closed_form(x)
    dual_cache = model.closed_form_dual(x, cache)
    closed_res = model.branch_residuals_from_closed_form(x, cache, dual_cache)
    t1 = time.perf_counter()

    forward_val = float(cache["forward_val"].item())
    dual_val = float(dual_cache["dual_val"].item())
    closed_gap = abs(forward_val - dual_val)

    solver_out = {"status": "SKIPPED", "obj_val": float("nan"), "runtime_sec": float("nan"), "z_vals": None, "s_vals": None, "q_vals": None, "t_vals": None, "u_vals": None}
    if cfg.run_solver_check:
        solver_out = solve_primal_qcp_with_gurobi(model, x, time_limit=cfg.solver_time_limit_sec)
    solver_abs_err = abs(solver_out["obj_val"] - forward_val) if np.isfinite(solver_out["obj_val"]) else float("nan")
    solver_res = branch_residuals_from_solver_solution(model, x, solver_out)

    return {
        "Passthrough": passthrough,
        "Seed": seed,
        "Trial": trial_id,
        "ForwardVal": forward_val,
        "ClosedDualVal": dual_val,
        "ClosedDualGap": closed_gap,
        "ClosedFormRuntimeMs": (t1 - t0) * 1000.0,
        "SolverStatus": solver_out["status"],
        "SolverPrimalVal": solver_out["obj_val"],
        "ForwardVsSolverAbsErr": solver_abs_err,
        "SolverRuntimeMs": solver_out["runtime_sec"] * 1000.0 if np.isfinite(solver_out["runtime_sec"]) else float("nan"),
        **closed_res,
        **solver_res,
    }


SUMMARY_COLS = [
    "ClosedDualGap",
    "ForwardVsSolverAbsErr",
    "ReLUPrimalViol",
    "ReLUDualBoxViol",
    "ReLUCompSlack",
    "QuadEpiViol",
    "QuadTight",
    "NormEpiViol",
    "NormTight",
    "NormDualBallViol",
    "NormDualAlignViol",
    "Solver_ReLUPrimalViol",
    "Solver_QuadEpiViol",
    "Solver_QuadTight",
    "Solver_NormEpiViol",
    "Solver_NormTight",
]


def print_group_summary(df_group: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    print(f"Trials: {len(df_group)}")
    print(f"{'Metric':<26} | {'Mean':>12} | {'Max':>12}")
    print("-" * 58)
    for col in SUMMARY_COLS:
        mean_v = pd.to_numeric(df_group[col], errors="coerce").mean()
        max_v = pd.to_numeric(df_group[col], errors="coerce").max()
        print(f"{col:<26} | {mean_v:12.4e} | {max_v:12.4e}")
    solver_ok_rate = float((df_group["SolverStatus"] == "OK").mean())
    print("-" * 58)
    print(f"{'Solver OK Rate':<26} | {solver_ok_rate:12.4f} | {'-':>12}")


def main():
    # ============================================================
    # Editable experiment parameters
    # ============================================================
    cfg = ExpConfig(
        input_dim=100,
        hidden_dim=256,
        depth=6,
        n_quad=1,
        quad_dim=24,
        n_norm=1,
        norm_dim=24,
        n_seeds=5,
        trials_per_seed=30,
        base_seed=2026,
        quad_alpha_init=1.5,
        norm_lambda_init=2.0,
        passthrough_settings=(True, False),
        run_solver_check=True,
        solver_time_limit_sec=20.0,
        detail_csv="exp1_soc_icnn_equivalence_detail.csv",
        summary_csv="exp1_soc_icnn_equivalence_summary.csv",
    )
    # ============================================================

    all_rows: List[Dict[str, object]] = []
    print("=" * 120)
    print("Experiment 1 | Full SOC-ICNN equivalence check")
    print("=" * 120)
    print(f"Scale: input_dim={cfg.input_dim}, hidden_dim={cfg.hidden_dim}, depth={cfg.depth}")
    print(f"Quad branch: n_quad={cfg.n_quad}, quad_dim={cfg.quad_dim}, alpha_init={cfg.quad_alpha_init}")
    print(f"Norm branch: n_norm={cfg.n_norm}, norm_dim={cfg.norm_dim}, lambda_init={cfg.norm_lambda_init}")
    print(f"Seeds={cfg.n_seeds}, trials_per_seed={cfg.trials_per_seed}")
    print(f"Solver enabled: {cfg.run_solver_check and GUROBI_AVAILABLE}")

    for passthrough in cfg.passthrough_settings:
        for seed_offset in range(cfg.n_seeds):
            seed = cfg.base_seed + seed_offset
            for t in range(cfg.trials_per_seed):
                trial_seed = seed * 1000 + t
                all_rows.append(run_single_trial(cfg, passthrough, trial_seed, t))
        df_tmp = pd.DataFrame([r for r in all_rows if r["Passthrough"] == passthrough])
        print_group_summary(df_tmp, title=f"Summary | Passthrough={passthrough}")

    df = pd.DataFrame(all_rows)
    df.to_csv(cfg.detail_csv, index=False)

    summary_rows = []
    for passthrough, grp in df.groupby("Passthrough"):
        row = {"Passthrough": passthrough, "NumTrials": len(grp), "SolverOKRate": float((grp["SolverStatus"] == "OK").mean())}
        for col in SUMMARY_COLS:
            row[f"{col}_Mean"] = pd.to_numeric(grp[col], errors="coerce").mean()
            row[f"{col}_Max"] = pd.to_numeric(grp[col], errors="coerce").max()
        row["ClosedFormRuntimeMs_Mean"] = pd.to_numeric(grp["ClosedFormRuntimeMs"], errors="coerce").mean()
        row["ClosedFormRuntimeMs_Std"] = pd.to_numeric(grp["ClosedFormRuntimeMs"], errors="coerce").std()
        row["SolverRuntimeMs_Mean"] = pd.to_numeric(grp["SolverRuntimeMs"], errors="coerce").mean()
        row["SolverRuntimeMs_Std"] = pd.to_numeric(grp["SolverRuntimeMs"], errors="coerce").std()
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(cfg.summary_csv, index=False)

    print("\nSaved files:")
    print(cfg.detail_csv)
    print(cfg.summary_csv)


if __name__ == "__main__":
    main()
