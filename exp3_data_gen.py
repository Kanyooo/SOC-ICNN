import json
from pathlib import Path
from typing import Dict, List, Tuple

import cvxpy as cp
import numpy as np


# ============================================================
# Basic utilities
# ============================================================


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def stable_logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    xmax = np.max(x, axis=axis, keepdims=True)
    out = xmax + np.log(np.sum(np.exp(x - xmax), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def huber_np(x: np.ndarray, delta: float) -> np.ndarray:
    abs_x = np.abs(x)
    quad = abs_x <= delta
    return np.where(quad, x * x, 2.0 * delta * abs_x - delta * delta)


def randn_matrix(rng: np.random.Generator, rows: int, cols: int, scale: float = 1.0) -> np.ndarray:
    mat = rng.standard_normal((rows, cols))
    mat = mat / np.sqrt(max(cols, 1))
    return scale * mat


def save_npz(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


# ============================================================
# Task registry
# ============================================================

TASK_SPECS = {
    "simplex_socp": {"family": "socp", "feasible": "simplex"},
    "box_socp": {"family": "socp", "feasible": "box"},
    "budget_twocone_socp": {"family": "socp_two", "feasible": "budget"},
    "simplex_logistic": {"family": "logistic", "feasible": "simplex"},
    "box_logsumexp": {"family": "logsumexp", "feasible": "box"},
    "budget_huber": {"family": "huber", "feasible": "budget"},
}


# ============================================================
# Task template
# ============================================================


def make_task_template(
    task_name: str,
    dim: int,
    context_dim: int,
    seed: int,
) -> Dict:
    if task_name not in TASK_SPECS:
        raise ValueError(f"Unknown task_name: {task_name}")

    spec = TASK_SPECS[task_name]
    family = spec["family"]
    feasible = spec["feasible"]
    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Feasible-set geometry
    # --------------------------------------------------------
    if feasible == "simplex":
        budget = 1.0
        m0 = rng.random(dim)
        m0 = m0 / m0.sum()
        m_ctx_scale = 0.08
        x_lower = np.zeros(dim)
        x_upper = np.ones(dim)
    elif feasible == "box":
        budget = None
        m0 = rng.uniform(0.20, 0.80, size=dim)
        m_ctx_scale = 0.25
        x_lower = np.zeros(dim)
        x_upper = np.ones(dim)
    elif feasible == "budget":
        budget = 0.30 * dim
        m0 = rng.uniform(0.10, 0.55, size=dim)
        m_ctx_scale = 0.20
        x_lower = np.zeros(dim)
        x_upper = np.ones(dim)
    else:
        raise ValueError(f"Unknown feasible set: {feasible}")

    # --------------------------------------------------------
    # Shared quadratic + linear backbone for all tasks
    # --------------------------------------------------------
    quad_scale = {
        "socp": 1.0,
        "socp_two": 1.0,
        "logistic": 0.35,
        "logsumexp": 0.35,
        "huber": 0.25,
    }[family]

    quad_w = rng.uniform(0.8, 1.6, size=dim)
    M_ctx = randn_matrix(rng, dim, context_dim, scale=m_ctx_scale)

    c0 = rng.uniform(-0.30, 0.30, size=dim)
    c_ctx_scale = 0.20 if feasible != "box" else 0.25
    C_ctx = randn_matrix(rng, dim, context_dim, scale=c_ctx_scale)

    template = {
        "task_name": task_name,
        "task_family": family,
        "feasible_type": feasible,
        "dim": dim,
        "context_dim": context_dim,
        "budget": budget,
        "quad_scale": quad_scale,
        "quad_w": quad_w,
        "m0": m0,
        "M_ctx": M_ctx,
        "c0": c0,
        "C_ctx": C_ctx,
        "x_lower": x_lower,
        "x_upper": x_upper,
    }

    # --------------------------------------------------------
    # Family-specific structural parameters
    # --------------------------------------------------------
    if family in {"socp", "socp_two"}:
        if family == "socp":
            cone_dims = [max(4, dim // 3)]
        else:
            cone_dims = [max(4, dim // 4), max(3, dim // 5)]

        lam_bias_low, lam_bias_high = (0.20, 0.60) if feasible == "simplex" else (0.25, 0.80)
        if feasible == "budget":
            lam_bias_low, lam_bias_high = 0.20, 0.70
        lam_ctx_scale = 0.20 if feasible != "box" else 0.25

        lam_W = randn_matrix(rng, len(cone_dims), context_dim, scale=lam_ctx_scale)
        lam_b = rng.uniform(lam_bias_low, lam_bias_high, size=len(cone_dims))
        template["cone_dims"] = np.array(cone_dims, dtype=np.int64)
        template["lam_W"] = lam_W
        template["lam_b"] = lam_b

        for j, k in enumerate(cone_dims):
            template[f"A_{j}"] = randn_matrix(rng, k, dim, scale=1.0)
            template[f"d0_{j}"] = rng.uniform(-0.30, 0.30, size=k)
            template[f"Dctx_{j}"] = randn_matrix(rng, k, context_dim, scale=0.25)

    elif family == "logistic":
        n_atoms = max(6, dim // 3)
        beta_W = randn_matrix(rng, n_atoms, context_dim, scale=0.15)
        beta_b = rng.uniform(0.20, 0.70, size=n_atoms)
        logi_A = randn_matrix(rng, n_atoms, dim, scale=1.0)
        logi_b0 = rng.uniform(-0.40, 0.40, size=n_atoms)
        logi_Bctx = randn_matrix(rng, n_atoms, context_dim, scale=0.25)

        template["n_atoms"] = np.array(n_atoms, dtype=np.int64)
        template["beta_W"] = beta_W
        template["beta_b"] = beta_b
        template["LOGI_A"] = logi_A
        template["LOGI_b0"] = logi_b0
        template["LOGI_Bctx"] = logi_Bctx

    elif family == "logsumexp":
        n_blocks = 2
        block_dim = max(4, dim // 5)
        template["n_blocks"] = np.array(n_blocks, dtype=np.int64)
        template["block_dim"] = np.array(block_dim, dtype=np.int64)
        template["lse_beta_W"] = randn_matrix(rng, n_blocks, context_dim, scale=0.15)
        template["lse_beta_b"] = rng.uniform(0.20, 0.60, size=n_blocks)

        for j in range(n_blocks):
            template[f"LSE_A_{j}"] = randn_matrix(rng, block_dim, dim, scale=1.0)
            template[f"LSE_b0_{j}"] = rng.uniform(-0.35, 0.35, size=block_dim)
            template[f"LSE_Bctx_{j}"] = randn_matrix(rng, block_dim, context_dim, scale=0.22)

    elif family == "huber":
        n_atoms = max(8, dim // 2)
        delta = 0.35
        beta_W = randn_matrix(rng, n_atoms, context_dim, scale=0.10)
        beta_b = rng.uniform(0.10, 0.45, size=n_atoms)
        hub_A = randn_matrix(rng, n_atoms, dim, scale=1.0)
        hub_b0 = rng.uniform(-0.30, 0.30, size=n_atoms)
        hub_Bctx = randn_matrix(rng, n_atoms, context_dim, scale=0.18)

        template["n_atoms"] = np.array(n_atoms, dtype=np.int64)
        template["huber_delta"] = np.array(delta, dtype=np.float64)
        template["beta_W"] = beta_W
        template["beta_b"] = beta_b
        template["HUB_A"] = hub_A
        template["HUB_b0"] = hub_b0
        template["HUB_Bctx"] = hub_Bctx

    else:
        raise ValueError(f"Unknown family: {family}")

    return template


# ============================================================
# Context -> parameters
# ============================================================


def context_to_params(template: Dict, theta: np.ndarray) -> Dict:
    task_name = template["task_name"]
    feasible = template["feasible_type"]
    family = template["task_family"]

    # shared part
    m_raw = template["m0"] + template["M_ctx"] @ theta
    if feasible == "simplex":
        m_pos = softplus(m_raw) + 1e-6
        m = m_pos / np.sum(m_pos)
    else:
        m = np.clip(m_raw, 0.0, 1.0)

    c = np.clip(template["c0"] + template["C_ctx"] @ theta, -2.0, 2.0)

    params = {
        "task_name": task_name,
        "task_family": family,
        "feasible_type": feasible,
        "theta": theta,
        "quad_scale": float(template["quad_scale"]),
        "quad_w": template["quad_w"],
        "m": m,
        "c": c,
    }

    if family in {"socp", "socp_two"}:
        cone_dims = template["cone_dims"]
        lambdas = 0.10 + softplus(template["lam_W"] @ theta + template["lam_b"])
        params["cone_dims"] = cone_dims
        params["lambdas"] = lambdas
        params["A_list"] = [template[f"A_{j}"] for j in range(len(cone_dims))]
        params["d_list"] = [template[f"d0_{j}"] + template[f"Dctx_{j}"] @ theta for j in range(len(cone_dims))]

    elif family == "logistic":
        beta = 0.05 + softplus(template["beta_W"] @ theta + template["beta_b"])
        shift = template["LOGI_b0"] + template["LOGI_Bctx"] @ theta
        params["beta"] = beta
        params["LOGI_A"] = template["LOGI_A"]
        params["shift"] = shift

    elif family == "logsumexp":
        n_blocks = int(template["n_blocks"])
        beta = 0.05 + softplus(template["lse_beta_W"] @ theta + template["lse_beta_b"])
        params["beta"] = beta
        params["A_list"] = [template[f"LSE_A_{j}"] for j in range(n_blocks)]
        params["shift_list"] = [template[f"LSE_b0_{j}"] + template[f"LSE_Bctx_{j}"] @ theta for j in range(n_blocks)]

    elif family == "huber":
        beta = 0.05 + softplus(template["beta_W"] @ theta + template["beta_b"])
        shift = template["HUB_b0"] + template["HUB_Bctx"] @ theta
        params["beta"] = beta
        params["HUB_A"] = template["HUB_A"]
        params["shift"] = shift
        params["delta"] = float(template["huber_delta"])

    else:
        raise ValueError(f"Unknown family: {family}")

    return params


# ============================================================
# True objective evaluation
# ============================================================


def objective_value_batch(template: Dict, params: Dict, X: np.ndarray) -> np.ndarray:
    quad = 0.5 * params["quad_scale"] * np.sum((params["quad_w"][None, :] * (X - params["m"][None, :])) ** 2, axis=1)
    lin = X @ params["c"]
    out = quad + lin
    family = params["task_family"]

    if family in {"socp", "socp_two"}:
        for lam, A, dj in zip(params["lambdas"], params["A_list"], params["d_list"]):
            out += lam * np.linalg.norm(X @ A.T - dj[None, :], axis=1)

    elif family == "logistic":
        z = X @ params["LOGI_A"].T - params["shift"][None, :]
        out += np.sum(params["beta"][None, :] * softplus(z), axis=1)

    elif family == "logsumexp":
        for beta_j, A_j, shift_j in zip(params["beta"], params["A_list"], params["shift_list"]):
            z = X @ A_j.T - shift_j[None, :]
            out += beta_j * stable_logsumexp(z, axis=1)

    elif family == "huber":
        z = X @ params["HUB_A"].T - params["shift"][None, :]
        out += np.sum(params["beta"][None, :] * huber_np(z, params["delta"]), axis=1)

    else:
        raise ValueError(f"Unknown family: {family}")

    return out


# ============================================================
# Solve true optimum
# ============================================================


def build_constraints(template: Dict, x: cp.Variable):
    feasible = template["feasible_type"]
    if feasible == "simplex":
        return [x >= 0.0, cp.sum(x) == 1.0]
    if feasible == "box":
        return [x >= 0.0, x <= 1.0]
    if feasible == "budget":
        return [x >= 0.0, x <= 1.0, cp.sum(x) == template["budget"]]
    raise ValueError(f"Unknown feasible type: {feasible}")


def solve_convex_instance(template: Dict, params: Dict):
    dim = template["dim"]
    family = template["task_family"]
    x = cp.Variable(dim)

    obj_expr = 0.5 * params["quad_scale"] * cp.sum_squares(cp.multiply(params["quad_w"], x - params["m"])) + params["c"] @ x

    if family in {"socp", "socp_two"}:
        for lam, A, dj in zip(params["lambdas"], params["A_list"], params["d_list"]):
            obj_expr += lam * cp.norm(A @ x - dj, 2)

    elif family == "logistic":
        Z = params["LOGI_A"] @ x - params["shift"]
        obj_expr += cp.sum(cp.multiply(params["beta"], cp.logistic(Z)))

    elif family == "logsumexp":
        for beta_j, A_j, shift_j in zip(params["beta"], params["A_list"], params["shift_list"]):
            obj_expr += beta_j * cp.log_sum_exp(A_j @ x - shift_j)

    elif family == "huber":
        Z = params["HUB_A"] @ x - params["shift"]
        obj_expr += cp.sum(cp.multiply(params["beta"], cp.huber(Z, params["delta"])))

    else:
        raise ValueError(f"Unknown family: {family}")

    prob = cp.Problem(cp.Minimize(obj_expr), build_constraints(template, x))

    solver_list = []
    if hasattr(cp, "CLARABEL"):
        solver_list.append(cp.CLARABEL)
    if hasattr(cp, "ECOS"):
        solver_list.append(cp.ECOS)
    solver_list.append(cp.SCS)

    last_err = None
    for solver in solver_list:
        try:
            prob.solve(solver=solver, verbose=False, warm_start=True)
            if prob.status in ["optimal", "optimal_inaccurate"] and x.value is not None:
                return np.asarray(x.value, dtype=np.float64), float(prob.value), prob.status, str(solver)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Convex solve failed. Last error: {last_err}")


# ============================================================
# Feasible-set sampling
# ============================================================


def sample_simplex_points(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    return rng.dirichlet(alpha=np.ones(dim), size=n)


def sample_box_points(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=(n, dim))


def sample_capped_simplex_points(
    rng: np.random.Generator,
    n: int,
    dim: int,
    total_sum: float,
    cap: float = 1.0,
) -> np.ndarray:
    samples = []
    batch_size = max(4 * n, 256)
    while len(samples) < n:
        z = rng.gamma(shape=1.0, scale=1.0, size=(batch_size, dim))
        z = z / z.sum(axis=1, keepdims=True)
        x = total_sum * z
        good = np.max(x, axis=1) <= cap + 1e-12
        accepted = x[good]
        if accepted.shape[0] > 0:
            samples.append(accepted)
    return np.concatenate(samples, axis=0)[:n]


def sample_feasible_points(template: Dict, n: int, rng: np.random.Generator) -> np.ndarray:
    feasible = template["feasible_type"]
    dim = template["dim"]
    if feasible == "simplex":
        return sample_simplex_points(rng, n, dim)
    if feasible == "box":
        return sample_box_points(rng, n, dim)
    if feasible == "budget":
        return sample_capped_simplex_points(rng, n, dim, total_sum=template["budget"], cap=1.0)
    raise ValueError(f"Unknown feasible type: {feasible}")


# ============================================================
# Generate dataset split
# ============================================================


def generate_split_dataset(
    template: Dict,
    n_instances: int,
    n_candidates: int,
    seed: int,
    include_opt_in_candidates: bool = False,
    max_retry_factor: int = 20,
) -> Dict:
    rng = np.random.default_rng(seed)
    task_name = template["task_name"]
    family = template["task_family"]
    dim = template["dim"]
    context_dim = template["context_dim"]

    contexts = np.zeros((n_instances, context_dim), dtype=np.float64)
    x_star = np.zeros((n_instances, dim), dtype=np.float64)
    f_star = np.zeros(n_instances, dtype=np.float64)
    cand_X = np.zeros((n_instances, n_candidates, dim), dtype=np.float64)
    cand_y = np.zeros((n_instances, n_candidates), dtype=np.float64)
    m_arr = np.zeros((n_instances, dim), dtype=np.float64)
    c_arr = np.zeros((n_instances, dim), dtype=np.float64)

    statuses = []
    solvers = []

    # family-specific per-instance arrays
    if family in {"socp", "socp_two"}:
        n_cones = len(template["cone_dims"])
        lambda_arr = np.zeros((n_instances, n_cones), dtype=np.float64)
        d_arr_list = [np.zeros((n_instances, int(k)), dtype=np.float64) for k in template["cone_dims"]]
    elif family == "logistic":
        n_atoms = int(template["n_atoms"])
        beta_arr = np.zeros((n_instances, n_atoms), dtype=np.float64)
        shift_arr = np.zeros((n_instances, n_atoms), dtype=np.float64)
    elif family == "logsumexp":
        n_blocks = int(template["n_blocks"])
        block_dim = int(template["block_dim"])
        beta_arr = np.zeros((n_instances, n_blocks), dtype=np.float64)
        shift_arr_list = [np.zeros((n_instances, block_dim), dtype=np.float64) for _ in range(n_blocks)]
    elif family == "huber":
        n_atoms = int(template["n_atoms"])
        beta_arr = np.zeros((n_instances, n_atoms), dtype=np.float64)
        shift_arr = np.zeros((n_instances, n_atoms), dtype=np.float64)
    else:
        raise ValueError(f"Unknown family: {family}")

    success = 0
    trials = 0
    max_trials = max_retry_factor * n_instances

    while success < n_instances and trials < max_trials:
        trials += 1
        theta = rng.standard_normal(context_dim)
        params = context_to_params(template, theta)

        try:
            x_opt, f_opt, status, solver_name = solve_convex_instance(template, params)
        except Exception:
            continue

        Xcand = sample_feasible_points(template, n_candidates, rng)
        if include_opt_in_candidates:
            Xcand[0] = x_opt
        ycand = objective_value_batch(template, params, Xcand)

        contexts[success] = theta
        x_star[success] = x_opt
        f_star[success] = f_opt
        cand_X[success] = Xcand
        cand_y[success] = ycand
        m_arr[success] = params["m"]
        c_arr[success] = params["c"]

        if family in {"socp", "socp_two"}:
            lambda_arr[success] = params["lambdas"]
            for j, dj in enumerate(params["d_list"]):
                d_arr_list[j][success] = dj
        elif family == "logistic":
            beta_arr[success] = params["beta"]
            shift_arr[success] = params["shift"]
        elif family == "logsumexp":
            beta_arr[success] = params["beta"]
            for j, sh in enumerate(params["shift_list"]):
                shift_arr_list[j][success] = sh
        elif family == "huber":
            beta_arr[success] = params["beta"]
            shift_arr[success] = params["shift"]

        statuses.append(status)
        solvers.append(solver_name)
        success += 1

        if success % 20 == 0:
            print(f"[{task_name} | d={dim}] generated {success}/{n_instances}")

    if success < n_instances:
        raise RuntimeError(
            f"Only generated {success}/{n_instances} instances for {task_name}, dim={dim}. "
            f"Please increase max_retry_factor or adjust task difficulty."
        )

    out = {
        "task_name": np.array(task_name),
        "task_family": np.array(family),
        "feasible_type": np.array(template["feasible_type"]),
        "dim": np.array(dim, dtype=np.int64),
        "context_dim": np.array(context_dim, dtype=np.int64),
        "budget": np.array(-1.0 if template["budget"] is None else template["budget"], dtype=np.float64),
        "quad_scale": np.array(template["quad_scale"], dtype=np.float64),
        "quad_w": template["quad_w"],
        "m0": template["m0"],
        "M_ctx": template["M_ctx"],
        "c0": template["c0"],
        "C_ctx": template["C_ctx"],
        "contexts": contexts,
        "x_star": x_star,
        "f_star": f_star,
        "cand_X": cand_X,
        "cand_y": cand_y,
        "m_arr": m_arr,
        "c_arr": c_arr,
        "statuses": np.array(statuses, dtype="U32"),
        "solvers": np.array(solvers, dtype="U32"),
        "include_opt_in_candidates": np.array(bool(include_opt_in_candidates)),
    }

    if family in {"socp", "socp_two"}:
        out["cone_dims"] = template["cone_dims"]
        out["lam_W"] = template["lam_W"]
        out["lam_b"] = template["lam_b"]
        out["lambda_arr"] = lambda_arr
        for j in range(len(template["cone_dims"])):
            out[f"A_{j}"] = template[f"A_{j}"]
            out[f"d0_{j}"] = template[f"d0_{j}"]
            out[f"Dctx_{j}"] = template[f"Dctx_{j}"]
            out[f"d_arr_{j}"] = d_arr_list[j]

    elif family == "logistic":
        out["n_atoms"] = np.array(int(template["n_atoms"]), dtype=np.int64)
        out["beta_W"] = template["beta_W"]
        out["beta_b"] = template["beta_b"]
        out["LOGI_A"] = template["LOGI_A"]
        out["LOGI_b0"] = template["LOGI_b0"]
        out["LOGI_Bctx"] = template["LOGI_Bctx"]
        out["beta_arr"] = beta_arr
        out["shift_arr"] = shift_arr

    elif family == "logsumexp":
        out["n_blocks"] = np.array(int(template["n_blocks"]), dtype=np.int64)
        out["block_dim"] = np.array(int(template["block_dim"]), dtype=np.int64)
        out["lse_beta_W"] = template["lse_beta_W"]
        out["lse_beta_b"] = template["lse_beta_b"]
        out["beta_arr"] = beta_arr
        for j in range(int(template["n_blocks"])):
            out[f"LSE_A_{j}"] = template[f"LSE_A_{j}"]
            out[f"LSE_b0_{j}"] = template[f"LSE_b0_{j}"]
            out[f"LSE_Bctx_{j}"] = template[f"LSE_Bctx_{j}"]
            out[f"shift_arr_{j}"] = shift_arr_list[j]

    elif family == "huber":
        out["n_atoms"] = np.array(int(template["n_atoms"]), dtype=np.int64)
        out["huber_delta"] = np.array(float(template["huber_delta"]), dtype=np.float64)
        out["beta_W"] = template["beta_W"]
        out["beta_b"] = template["beta_b"]
        out["HUB_A"] = template["HUB_A"]
        out["HUB_b0"] = template["HUB_b0"]
        out["HUB_Bctx"] = template["HUB_Bctx"]
        out["beta_arr"] = beta_arr
        out["shift_arr"] = shift_arr

    return out


# ============================================================
# Flatten to supervised learning arrays
# ============================================================


def flatten_supervised_dataset(npz_file: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_file, allow_pickle=True)
    contexts = data["contexts"]
    cand_X = data["cand_X"]
    cand_y = data["cand_y"]
    N, M, d = cand_X.shape
    theta_flat = np.repeat(contexts, M, axis=0)
    x_flat = cand_X.reshape(N * M, d)
    y_flat = cand_y.reshape(N * M, 1)
    return {"theta": theta_flat, "x": x_flat, "y": y_flat}


# ============================================================
# Generate all datasets
# ============================================================


def generate_all_exp3_datasets(
    out_root: str = "exp3_convex_fair_data",
    dims: Tuple[int, ...] = (10, 20, 50),
    tasks: Tuple[str, ...] = (
        "simplex_socp",
        "box_socp",
        "budget_twocone_socp",
        "simplex_logistic",
        "box_logsumexp",
        "budget_huber",
    ),
    context_dim: int = 8,
    n_train_instances: int = 1000,
    n_test_instances: int = 200,
    n_train_candidates: int = 64,
    n_test_candidates: int = 128,
    include_opt_in_candidates: bool = False,
    base_seed: int = 2026,
):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = []

    for task_idx, task_name in enumerate(tasks):
        for dim in dims:
            print("=" * 80)
            print(f"Generating task={task_name}, dim={dim}")

            template_seed = base_seed + 1000 * task_idx + dim
            train_seed = base_seed + 10000 + 1000 * task_idx + dim
            test_seed = base_seed + 20000 + 1000 * task_idx + dim

            template = make_task_template(task_name, dim, context_dim, template_seed)
            train_data = generate_split_dataset(
                template=template,
                n_instances=n_train_instances,
                n_candidates=n_train_candidates,
                seed=train_seed,
                include_opt_in_candidates=include_opt_in_candidates,
            )
            test_data = generate_split_dataset(
                template=template,
                n_instances=n_test_instances,
                n_candidates=n_test_candidates,
                seed=test_seed,
                include_opt_in_candidates=include_opt_in_candidates,
            )

            task_dir = out_root / task_name / f"d{dim}"
            save_npz(task_dir / "train.npz", train_data)
            save_npz(task_dir / "test.npz", test_data)

            meta = {
                "task_name": task_name,
                "task_family": TASK_SPECS[task_name]["family"],
                "feasible_type": TASK_SPECS[task_name]["feasible"],
                "dim": dim,
                "context_dim": context_dim,
                "n_train_instances": n_train_instances,
                "n_test_instances": n_test_instances,
                "n_train_candidates": n_train_candidates,
                "n_test_candidates": n_test_candidates,
                "include_opt_in_candidates": include_opt_in_candidates,
                "budget": None if template["budget"] is None else float(template["budget"]),
                "train_file": str(task_dir / "train.npz"),
                "test_file": str(task_dir / "test.npz"),
            }
            with open(task_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            summary.append(meta)

    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("All datasets generated successfully.")
    print(f"Saved to: {out_root.resolve()}")


if __name__ == "__main__":
    generate_all_exp3_datasets(
        out_root="exp3_convex_fair_data",
        dims=(10, 20, 50),
        tasks=(
            "simplex_socp",
            "box_socp",
            "budget_twocone_socp",
            "simplex_logistic",
            "box_logsumexp",
            "budget_huber",
        ),
        context_dim=8,
        n_train_instances=1000,
        n_test_instances=200,
        n_train_candidates=64,
        n_test_candidates=128,
        include_opt_in_candidates=False,
        base_seed=2026,
    )
