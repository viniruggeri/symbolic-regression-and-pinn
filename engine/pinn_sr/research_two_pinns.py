import argparse
import copy
import json
import os
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torchdiffeq import odeint
except ImportError as exc:
    raise ImportError("torchdiffeq is required. Install with: pip install torchdiffeq") from exc


@dataclass
class TrainConfig:
    epochs: int = 1200
    lr: float = 1e-3
    weight_decay: float = 1e-6
    hidden_dim: int = 96
    dropout: float = 0.10
    lambda_dyn: float = 0.20
    lambda_ve: float = 0.05
    lambda_smooth: float = 1e-4
    lambda_bounds: float = 1e-3
    grad_clip: float = 1.0
    patience: int = 140
    method: str = "dopri5"
    rtol: float = 1e-5
    atol: float = 1e-6
    device: str = "cpu"
    seed: int = 42


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ControlProfile:
    def __init__(self, t_ref: torch.Tensor, ign_ref: torch.Tensor, tps_ref: torch.Tensor):
        self.t_ref = t_ref
        self.ign_ref = ign_ref
        self.tps_ref = tps_ref

    def at(self, t: torch.Tensor):
        if t.ndim != 0:
            t = t.squeeze()
        idx = torch.searchsorted(self.t_ref, t)
        idx = torch.clamp(idx, 1, self.t_ref.shape[0] - 1)

        t0 = self.t_ref[idx - 1]
        t1 = self.t_ref[idx]
        w = torch.where((t1 - t0) > 0, (t - t0) / (t1 - t0), torch.zeros_like(t))

        ign = self.ign_ref[idx - 1] * (1 - w) + self.ign_ref[idx] * w
        tps = self.tps_ref[idx - 1] * (1 - w) + self.tps_ref[idx] * w
        return ign, tps


class BaseODEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.control_profile = None
        self.boost_offset = 0.0

    def set_control_profile(self, control_profile: ControlProfile):
        self.control_profile = control_profile

    def set_boost(self, boost_offset: float):
        self.boost_offset = float(boost_offset)


class PhysicsInformedODE(BaseODEModel):
    """PINN 1: SINDy base dynamics + residual NN, with SR-consistency losses."""

    def __init__(self, sindy_coeffs: np.ndarray, hidden_dim: int = 96, dropout: float = 0.1):
        super().__init__()
        coeffs = torch.tensor(sindy_coeffs, dtype=torch.float32)
        if coeffs.shape != (2, 15):
            raise ValueError(f"Expected SINDy coeffs shape (2, 15), got {tuple(coeffs.shape)}")
        self.register_buffer("sindy_coeffs", coeffs)

        self.residual = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def sindy_features(self, x: torch.Tensor, u_ign: torch.Tensor, u_tps: torch.Tensor):
        rpm = torch.clamp(x[:, 0], 0.0, 8000.0)
        map_val = torch.clamp(x[:, 1], 0.05, 4.0)
        return torch.stack(
            [
                torch.ones_like(rpm),
                rpm,
                map_val,
                u_ign,
                u_tps,
                rpm**2,
                rpm * map_val,
                rpm * u_ign,
                rpm * u_tps,
                map_val**2,
                map_val * u_ign,
                map_val * u_tps,
                u_ign**2,
                u_ign * u_tps,
                u_tps**2,
            ],
            dim=1,
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        if self.control_profile is None:
            u_ign = torch.full((x.shape[0],), 15.0, device=x.device)
            u_tps = torch.full((x.shape[0],), 100.0, device=x.device)
        else:
            ign, tps = self.control_profile.at(t)
            u_ign = ign.expand(x.shape[0])
            u_tps = tps.expand(x.shape[0])

        feats = self.sindy_features(x, u_ign, u_tps)
        base_dx = feats @ self.sindy_coeffs.T

        x_scaled = torch.stack(
            [
                torch.clamp(x[:, 0], 0.0, 8000.0) / 8000.0,
                torch.clamp(x[:, 1], 0.05, 4.0) / 4.0,
            ],
            dim=1,
        )
        residual_dx = self.residual(x_scaled)

        dx = base_dx + residual_dx
        dx[:, 0] = dx[:, 0] + self.boost_offset * 500.0

        drpm_dt = torch.clamp(dx[:, 0], -3000.0, 6000.0)
        dmap_dt = torch.clamp(dx[:, 1], -3.0, 3.0)
        return torch.stack([drpm_dt, dmap_dt], dim=1)


class EngineeringODE(BaseODEModel):
    """PINN 2: no SINDy/SR equation in loss, only engineering priors + data fit."""

    def __init__(self, hidden_dim: int = 96, dropout: float = 0.1):
        super().__init__()
        self.p_map = nn.Parameter(torch.tensor(1.2))
        self.p_ign = nn.Parameter(torch.tensor(0.4))
        self.p_tps = nn.Parameter(torch.tensor(0.7))
        self.p_drag = nn.Parameter(torch.tensor(1.5))
        self.p_boost = nn.Parameter(torch.tensor(1.0))

        self.m_recover = nn.Parameter(torch.tensor(1.0))
        self.m_leak = nn.Parameter(torch.tensor(0.5))
        self.m_boost = nn.Parameter(torch.tensor(0.6))
        self.m_tps = nn.Parameter(torch.tensor(0.4))

        self.residual = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        rpm = torch.clamp(x[:, 0], 0.0, 8000.0)
        map_val = torch.clamp(x[:, 1], 0.05, 4.0)

        if self.control_profile is None:
            u_ign = torch.full((x.shape[0],), 15.0, device=x.device)
            u_tps = torch.full((x.shape[0],), 100.0, device=x.device)
        else:
            ign, tps = self.control_profile.at(t)
            u_ign = ign.expand(x.shape[0])
            u_tps = tps.expand(x.shape[0])

        ign_n = torch.clamp(u_ign / 20.0, 0.0, 2.0)
        tps_n = torch.clamp(u_tps / 100.0, 0.0, 1.2)
        rpm_n = torch.clamp(rpm / 8000.0, 0.0, 1.0)
        map_n = torch.clamp(map_val / 4.0, 0.0, 1.0)

        a_map = torch.nn.functional.softplus(self.p_map)
        a_ign = torch.nn.functional.softplus(self.p_ign)
        a_tps = torch.nn.functional.softplus(self.p_tps)
        a_drag = torch.nn.functional.softplus(self.p_drag)
        a_boost = torch.nn.functional.softplus(self.p_boost)

        b_recover = torch.nn.functional.softplus(self.m_recover)
        b_leak = torch.nn.functional.softplus(self.m_leak)
        b_boost = torch.nn.functional.softplus(self.m_boost)
        b_tps = torch.nn.functional.softplus(self.m_tps)

        target_map = 1.0 + 1.2 * tps_n

        drpm_base = (
            320.0 * a_map * map_n
            + 120.0 * a_ign * ign_n
            + 150.0 * a_tps * tps_n
            - 250.0 * a_drag * (rpm_n**2)
            + 420.0 * a_boost * self.boost_offset
        )

        dmap_base = (
            b_recover * (target_map - map_val)
            + 0.6 * b_tps * (tps_n - 0.5)
            + 0.8 * b_boost * self.boost_offset
            - 0.4 * b_leak * rpm_n
        )

        aux = torch.stack([rpm_n, map_n, ign_n, tps_n], dim=1)
        dres = self.residual(aux)

        drpm_dt = torch.clamp(drpm_base + 300.0 * dres[:, 0], -3000.0, 6000.0)
        dmap_dt = torch.clamp(dmap_base + 0.4 * dres[:, 1], -3.0, 3.0)
        return torch.stack([drpm_dt, dmap_dt], dim=1)


def ve_torch(rpm: torch.Tensor, map_val: torch.Tensor, ve_eq_str: str):
    expr = ve_eq_str.replace("^", "**")
    x0 = torch.clamp(rpm, 0.0, 8000.0)
    x1 = torch.clamp(map_val, 0.0, 4.0)

    safe_dict = {
        "x0": x0,
        "x1": x1,
        "sin": torch.sin,
        "exp": lambda v: torch.exp(torch.clamp(v, -20.0, 2.0)),
        "log": lambda v: torch.log(torch.clamp(v, min=1e-6)),
        "sqrt": lambda v: torch.sqrt(torch.clamp(v, min=1e-8)),
        "abs": torch.abs,
    }
    try:
        ve = eval(expr, {"__builtins__": None}, safe_dict)
        return torch.clamp(ve, 0.2, 1.2)
    except Exception:
        return torch.full_like(x0, 0.85)


def check_validation(path: str = "models/equation_validation.json"):
    if not os.path.exists(path):
        raise FileNotFoundError("Missing models/equation_validation.json. Run pinn_sr/sim_motor.py first.")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("status") != "approved":
        raise RuntimeError("Equation validation status is not approved. Fix discovery stage first.")


def load_data(path: str = "data/synth_data.csv"):
    df = pd.read_csv(path).sort_values("time").reset_index(drop=True)
    required = ["time", "RPM", "MAP", "Ignition", "tps"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in {path}: {miss}")
    return df


def make_splits(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    if min(n_train, n_val, n_test) < 10:
        raise ValueError("Not enough samples for train/val/test split.")

    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:n_train + n_val].copy()
    df_test = df.iloc[n_train + n_val:].copy()
    return df_train, df_val, df_test


def to_segment(df_split: pd.DataFrame, device: str):
    t_abs = torch.tensor(df_split["time"].values, dtype=torch.float32, device=device)
    t_rel = t_abs - t_abs[0]
    x = torch.tensor(df_split[["RPM", "MAP"]].values, dtype=torch.float32, device=device)
    ign = torch.tensor(df_split["Ignition"].values, dtype=torch.float32, device=device)
    tps = torch.tensor(df_split["tps"].values, dtype=torch.float32, device=device)

    dt = torch.diff(t_rel)
    dx_dt_obs = torch.diff(x, dim=0) / dt.unsqueeze(1)

    seg = {
        "t_abs": t_abs,
        "t_rel": t_rel,
        "x": x,
        "x0": x[0].clone(),
        "ign": ign,
        "tps": tps,
        "dx_dt_obs": dx_dt_obs,
    }
    return seg


def make_control(seg):
    return ControlProfile(seg["t_rel"], seg["ign"], seg["tps"])


def simulate(model: BaseODEModel, seg, boost_offset=0.0, train_mode=False, method="dopri5", rtol=1e-5, atol=1e-6, x0_override=None, control_override=None):
    if train_mode:
        model.train()
    else:
        model.eval()

    model.set_boost(boost_offset)
    model.set_control_profile(control_override if control_override is not None else make_control(seg))

    x0 = seg["x0"] if x0_override is None else x0_override
    return odeint(model, x0, seg["t_rel"], method=method, rtol=rtol, atol=atol)


def rmse_torch(a: torch.Tensor, b: torch.Tensor):
    return torch.sqrt(torch.mean((a - b) ** 2))


def second_diff_penalty(x: torch.Tensor):
    if x.shape[0] < 3:
        return torch.tensor(0.0, device=x.device)
    dd = x[2:] - 2 * x[1:-1] + x[:-2]
    return torch.mean(dd**2)


def bounds_penalty(pred: torch.Tensor):
    rpm = pred[:, 0]
    map_val = pred[:, 1]
    p_rpm = torch.mean(torch.relu(-rpm) + torch.relu(rpm - 8000.0))
    p_map = torch.mean(torch.relu(0.05 - map_val) + torch.relu(map_val - 4.0))
    return p_rpm + p_map


def model_dx_sequence(model: BaseODEModel, seg, x_states: torch.Tensor):
    outs = []
    for i in range(x_states.shape[0] - 1):
        outs.append(model(seg["t_rel"][i], x_states[i]).squeeze(0))
    return torch.stack(outs, dim=0)


def train_one_model(name: str, model: BaseODEModel, train_seg, val_seg, cfg: TrainConfig, out_dir: str, ve_eq_str: str = ""):
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=35)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad()

        pred_train = simulate(
            model,
            train_seg,
            boost_offset=0.0,
            train_mode=True,
            method=cfg.method,
            rtol=cfg.rtol,
            atol=cfg.atol,
        )

        data_loss = torch.mean((pred_train - train_seg["x"]) ** 2)
        smooth_loss = second_diff_penalty(pred_train)
        bound_loss = bounds_penalty(pred_train)

        total = data_loss + cfg.lambda_smooth * smooth_loss + cfg.lambda_bounds * bound_loss

        dyn_loss = torch.tensor(0.0, device=cfg.device)
        ve_loss = torch.tensor(0.0, device=cfg.device)

        if isinstance(model, PhysicsInformedODE):
            dx_model = model_dx_sequence(model, train_seg, train_seg["x"])
            dyn_loss = torch.mean((dx_model - train_seg["dx_dt_obs"]) ** 2)
            ve_pred = ve_torch(pred_train[:, 0], pred_train[:, 1], ve_eq_str)
            ve_obs = ve_torch(train_seg["x"][:, 0], train_seg["x"][:, 1], ve_eq_str)
            ve_loss = torch.mean((ve_pred - ve_obs) ** 2)
            total = total + cfg.lambda_dyn * dyn_loss + cfg.lambda_ve * ve_loss

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        with torch.no_grad():
            pred_val = simulate(
                model,
                val_seg,
                boost_offset=0.0,
                train_mode=False,
                method=cfg.method,
                rtol=cfg.rtol,
                atol=cfg.atol,
            )
            val_data = torch.mean((pred_val - val_seg["x"]) ** 2)
            val_total = val_data + cfg.lambda_smooth * second_diff_penalty(pred_val) + cfg.lambda_bounds * bounds_penalty(pred_val)

            if isinstance(model, PhysicsInformedODE):
                dx_val = model_dx_sequence(model, val_seg, val_seg["x"])
                val_dyn = torch.mean((dx_val - val_seg["dx_dt_obs"]) ** 2)
                ve_val_pred = ve_torch(pred_val[:, 0], pred_val[:, 1], ve_eq_str)
                ve_val_obs = ve_torch(val_seg["x"][:, 0], val_seg["x"][:, 1], ve_eq_str)
                val_ve = torch.mean((ve_val_pred - ve_val_obs) ** 2)
                val_total = val_total + cfg.lambda_dyn * val_dyn + cfg.lambda_ve * val_ve

        scheduler.step(val_total)

        rec = {
            "epoch": epoch,
            "train_total": float(total.item()),
            "train_data": float(data_loss.item()),
            "train_dyn": float(dyn_loss.item()),
            "train_ve": float(ve_loss.item()),
            "val_total": float(val_total.item()),
            "val_data": float(val_data.item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(rec)

        if val_total.item() < best_val:
            best_val = val_total.item()
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 100 == 0:
            print(
                f"[{name}] Epoch {epoch:04d} | "
                f"train={rec['train_total']:.6f} val={rec['val_total']:.6f} "
                f"data={rec['train_data']:.6f} dyn={rec['train_dyn']:.6f} ve={rec['train_ve']:.6f}"
            )

        if bad_epochs >= cfg.patience:
            print(f"[{name}] Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(out_dir, f"{name}_history.csv"), index=False)
    return model, hist_df


def eval_metrics(pred: torch.Tensor, target: torch.Tensor):
    rpm_rmse = rmse_torch(pred[:, 0], target[:, 0]).item()
    map_rmse = rmse_torch(pred[:, 1], target[:, 1]).item()
    rpm_mae = torch.mean(torch.abs(pred[:, 0] - target[:, 0])).item()
    map_mae = torch.mean(torch.abs(pred[:, 1] - target[:, 1])).item()
    return {
        "rpm_rmse": float(rpm_rmse),
        "map_rmse": float(map_rmse),
        "rpm_mae": float(rpm_mae),
        "map_mae": float(map_mae),
    }


def run_mc_dropout(model: BaseODEModel, seg, cfg: TrainConfig, out_dir: str, name: str, n_samples=60):
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            p = simulate(
                model,
                seg,
                boost_offset=0.0,
                train_mode=True,
                method=cfg.method,
                rtol=cfg.rtol,
                atol=cfg.atol,
            )
            preds.append(p.unsqueeze(0))

    stack = torch.cat(preds, dim=0)
    mean = torch.mean(stack, dim=0)
    std = torch.std(stack, dim=0)

    df = pd.DataFrame(
        {
            "time": seg["t_rel"].cpu().numpy(),
            "rpm_mean": mean[:, 0].cpu().numpy(),
            "rpm_std": std[:, 0].cpu().numpy(),
            "map_mean": mean[:, 1].cpu().numpy(),
            "map_std": std[:, 1].cpu().numpy(),
        }
    )
    df.to_csv(os.path.join(out_dir, f"{name}_mc_dropout.csv"), index=False)

    t = seg["t_rel"].cpu().numpy()
    obs = seg["x"].cpu().numpy()
    rpm_mean = mean[:, 0].cpu().numpy()
    rpm_std = std[:, 0].cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(t, obs[:, 0], label="RPM observado", color="black")
    plt.plot(t, rpm_mean, label=f"{name} RPM mean", color="#1f77b4")
    plt.fill_between(t, rpm_mean - 2 * rpm_std, rpm_mean + 2 * rpm_std, color="#1f77b4", alpha=0.2, label="95% CI MC")
    plt.title(f"MC Dropout Uncertainty - {name}")
    plt.xlabel("tempo")
    plt.ylabel("RPM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_mc_dropout_rpm.png"), dpi=150)
    plt.close()


def run_nll_noise(model: BaseODEModel, seg, cfg: TrainConfig, out_dir: str, name: str):
    noise_levels = [0.0, 0.01, 0.03, 0.05, 0.08]
    n_samples = 40
    target = seg["x"]

    rows = []
    with torch.no_grad():
        for nl in noise_levels:
            preds = []
            for _ in range(n_samples):
                noise = torch.tensor(
                    [np.random.normal(0, 100.0 * nl), np.random.normal(0, 0.12 * nl)],
                    dtype=torch.float32,
                    device=cfg.device,
                )
                x0_noisy = seg["x0"] + noise
                p = simulate(
                    model,
                    seg,
                    boost_offset=0.0,
                    train_mode=False,
                    method=cfg.method,
                    rtol=cfg.rtol,
                    atol=cfg.atol,
                    x0_override=x0_noisy,
                )
                preds.append(p.unsqueeze(0))

            stack = torch.cat(preds, dim=0)
            mu = torch.mean(stack, dim=0)
            var = torch.var(stack, dim=0) + 1e-6
            nll = 0.5 * torch.log(2 * torch.pi * var) + 0.5 * ((target - mu) ** 2) / var
            rows.append(
                {
                    "noise_level": nl,
                    "rpm_nll": float(torch.mean(nll[:, 0]).item()),
                    "map_nll": float(torch.mean(nll[:, 1]).item()),
                    "joint_nll": float(torch.mean(nll).item()),
                }
            )

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{name}_nll_noise.csv"), index=False)


def run_counterfactual(model: BaseODEModel, seg, cfg: TrainConfig, out_dir: str, name: str, ve_eq_str: str):
    boosts = [0.0, 0.2, 0.4]
    map_scales = [0.8, 1.0, 1.2]

    rows = []
    with torch.no_grad():
        for b in boosts:
            for ms in map_scales:
                x0 = seg["x0"].clone()
                x0[1] = x0[1] * ms
                pred = simulate(
                    model,
                    seg,
                    boost_offset=b,
                    train_mode=False,
                    method=cfg.method,
                    rtol=cfg.rtol,
                    atol=cfg.atol,
                    x0_override=x0,
                )
                ve = ve_torch(pred[:, 0], pred[:, 1], ve_eq_str)
                torque = 18.5 * pred[:, 1] * ve
                cv = (torque * pred[:, 0]) / 716.2
                rows.append(
                    {
                        "boost_offset": b,
                        "map_scale_init": ms,
                        "rpm_final": float(pred[-1, 0].item()),
                        "map_final": float(pred[-1, 1].item()),
                        "rpm_peak": float(torch.max(pred[:, 0]).item()),
                        "cv_peak": float(torch.max(cv).item()),
                    }
                )

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{name}_counterfactual.csv"), index=False)


def run_feature_missing(model: BaseODEModel, train_seg, test_seg, cfg: TrainConfig, out_dir: str, name: str):
    with torch.no_grad():
        pred_base = simulate(model, test_seg, boost_offset=0.0, train_mode=False, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol)
        base_metrics = eval_metrics(pred_base, test_seg["x"])

    train_rpm_mean = torch.mean(train_seg["x"][:, 0])
    train_map_mean = torch.mean(train_seg["x"][:, 1])
    train_ign_mean = torch.mean(train_seg["ign"])
    train_tps_mean = torch.mean(train_seg["tps"])

    scenarios = []

    with torch.no_grad():
        x0 = test_seg["x0"].clone()
        x0[0] = train_rpm_mean
        p = simulate(model, test_seg, boost_offset=0.0, train_mode=False, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol, x0_override=x0)
        scenarios.append(("missing_init_rpm", p))

        x0 = test_seg["x0"].clone()
        x0[1] = train_map_mean
        p = simulate(model, test_seg, boost_offset=0.0, train_mode=False, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol, x0_override=x0)
        scenarios.append(("missing_init_map", p))

        ign_const = torch.full_like(test_seg["ign"], train_ign_mean)
        ctrl = ControlProfile(test_seg["t_rel"], ign_const, test_seg["tps"])
        p = simulate(model, test_seg, boost_offset=0.0, train_mode=False, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol, control_override=ctrl)
        scenarios.append(("missing_ignition_profile", p))

        tps_const = torch.full_like(test_seg["tps"], train_tps_mean)
        ctrl = ControlProfile(test_seg["t_rel"], test_seg["ign"], tps_const)
        p = simulate(model, test_seg, boost_offset=0.0, train_mode=False, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol, control_override=ctrl)
        scenarios.append(("missing_tps_profile", p))

    rows = []
    for label, pred in scenarios:
        m = eval_metrics(pred, test_seg["x"])
        rows.append(
            {
                "scenario": label,
                "rpm_rmse": m["rpm_rmse"],
                "map_rmse": m["map_rmse"],
                "delta_rpm_rmse": m["rpm_rmse"] - base_metrics["rpm_rmse"],
                "delta_map_rmse": m["map_rmse"] - base_metrics["map_rmse"],
            }
        )

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{name}_feature_missing.csv"), index=False)


def run_sensitivity(model: BaseODEModel, seg, cfg: TrainConfig, out_dir: str, name: str):
    model.eval()
    model.set_boost(0.0)
    model.set_control_profile(make_control(seg))

    x0 = seg["x0"].clone().detach().requires_grad_(True)
    pred = odeint(model, x0, seg["t_rel"], method=cfg.method, rtol=cfg.rtol, atol=cfg.atol)
    y = pred[-1, 0]
    y.backward()

    grad = x0.grad.detach().cpu().numpy()
    payload = {
        "d_rpm_final_d_rpm0": float(grad[0]),
        "d_rpm_final_d_map0": float(grad[1]),
    }
    with open(os.path.join(out_dir, f"{name}_sensitivity.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot_2d_3d(df, full_pred_phy, full_pred_eng, out_dir: str):
    t = df["time"].values
    obs_rpm = df["RPM"].values
    obs_map = df["MAP"].values

    p1 = full_pred_phy.detach().cpu().numpy()
    p2 = full_pred_eng.detach().cpu().numpy()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, obs_rpm, label="observado", color="black", linewidth=1.0)
    plt.plot(t, p1[:, 0], label="PINN física", color="#d62728")
    plt.plot(t, p2[:, 0], label="PINN engenharia", color="#1f77b4")
    plt.title("RPM ao longo do tempo")
    plt.xlabel("tempo")
    plt.ylabel("RPM")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t, obs_map, label="observado", color="black", linewidth=1.0)
    plt.plot(t, p1[:, 1], label="PINN física", color="#d62728")
    plt.plot(t, p2[:, 1], label="PINN engenharia", color="#1f77b4")
    plt.title("MAP ao longo do tempo")
    plt.xlabel("tempo")
    plt.ylabel("MAP")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_2d_stock.png"), dpi=150)
    plt.close()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(t, obs_rpm, obs_map, color="black", label="observado")
    ax.plot(t, p1[:, 0], p1[:, 1], color="#d62728", label="PINN física")
    ax.plot(t, p2[:, 0], p2[:, 1], color="#1f77b4", label="PINN engenharia")
    ax.set_xlabel("tempo")
    ax.set_ylabel("RPM")
    ax.set_zlabel("MAP")
    ax.set_title("Trajetória 3D (tempo, RPM, MAP)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_3d_stock.png"), dpi=150)
    plt.close()


def plot_upgrade(df, model_phy, model_eng, full_seg, cfg: TrainConfig, out_dir: str):
    with torch.no_grad():
        stock_phy = simulate(model_phy, full_seg, boost_offset=0.0, train_mode=False, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol)
        upg_phy = simulate(model_phy, full_seg, boost_offset=0.3, train_mode=False, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol)

        stock_eng = simulate(model_eng, full_seg, boost_offset=0.0, train_mode=False, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol)
        upg_eng = simulate(model_eng, full_seg, boost_offset=0.3, train_mode=False, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol)

    t = df["time"].values
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, stock_phy[:, 0].cpu().numpy(), label="Física stock", color="#d62728")
    plt.plot(t, upg_phy[:, 0].cpu().numpy(), label="Física upgrade", color="#ff9896")
    plt.title("PINN Física: impacto do upgrade")
    plt.xlabel("tempo")
    plt.ylabel("RPM")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t, stock_eng[:, 0].cpu().numpy(), label="Eng stock", color="#1f77b4")
    plt.plot(t, upg_eng[:, 0].cpu().numpy(), label="Eng upgrade", color="#aec7e8")
    plt.title("PINN Engenharia: impacto do upgrade")
    plt.xlabel("tempo")
    plt.ylabel("RPM")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "upgrade_2d_rpm.png"), dpi=150)
    plt.close()


def main(cfg: TrainConfig):
    set_seed(cfg.seed)
    check_validation("models/equation_validation.json")

    data = load_data("data/synth_data.csv")
    train_df, val_df, test_df = make_splits(data)

    out_dir = os.path.join("outputs", datetime.now().strftime("%Y%m%d_%H%M%S_two_pinns"))
    os.makedirs(out_dir, exist_ok=True)

    with open("models/ve_equation.txt", "r", encoding="utf-8") as f:
        ve_eq_str = f.read().strip()

    sindy_coeffs = np.load("models/sindy_coefficients.npy")

    train_seg = to_segment(train_df, cfg.device)
    val_seg = to_segment(val_df, cfg.device)
    test_seg = to_segment(test_df, cfg.device)
    full_seg = to_segment(data, cfg.device)

    model_phy = PhysicsInformedODE(sindy_coeffs, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(cfg.device)
    model_eng = EngineeringODE(hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(cfg.device)

    manifest = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "source": "data/synth_data.csv",
            "split_sizes": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
            },
        },
        "training": {
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
            "early_stopping": cfg.patience,
            "ode_solver": {
                "method": cfg.method,
                "rtol": cfg.rtol,
                "atol": cfg.atol,
            },
            "config": {
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "hidden_dim": cfg.hidden_dim,
                "dropout": cfg.dropout,
                "lambda_dyn": cfg.lambda_dyn,
                "lambda_ve": cfg.lambda_ve,
                "lambda_smooth": cfg.lambda_smooth,
                "lambda_bounds": cfg.lambda_bounds,
                "grad_clip": cfg.grad_clip,
                "seed": cfg.seed,
                "device": cfg.device,
            },
            "losses": {
                "pinn_physics": [
                    "data_mse",
                    "dynamics_consistency_mse",
                    "ve_consistency_mse",
                    "smoothness_penalty",
                    "state_bounds_penalty",
                ],
                "pinn_engineering": [
                    "data_mse",
                    "smoothness_penalty",
                    "state_bounds_penalty",
                ],
            },
        },
        "models": {
            "pinn_physics": {
                "class": "PhysicsInformedODE",
                "trainable_params": count_trainable_params(model_phy),
            },
            "pinn_engineering": {
                "class": "EngineeringODE",
                "trainable_params": count_trainable_params(model_eng),
            },
        },
        "xai": {
            "methods": [
                "mc_dropout",
                "nll_under_initial_state_noise",
                "counterfactual_scenarios",
                "feature_missing_impact",
                "local_initial_condition_sensitivity",
            ]
        },
    }
    with open(os.path.join(out_dir, "training_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Treinando PINN 1 (física: SINDy+SR loss)...")
    model_phy, hist_phy = train_one_model("pinn_physics", model_phy, train_seg, val_seg, cfg, out_dir, ve_eq_str=ve_eq_str)

    print("Treinando PINN 2 (engenharia, sem loss SINDy/SR)...")
    model_eng, hist_eng = train_one_model("pinn_engineering", model_eng, train_seg, val_seg, cfg, out_dir, ve_eq_str=ve_eq_str)

    with torch.no_grad():
        pred_train_phy = simulate(model_phy, train_seg, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol)
        pred_val_phy = simulate(model_phy, val_seg, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol)
        pred_test_phy = simulate(model_phy, test_seg, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol)

        pred_train_eng = simulate(model_eng, train_seg, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol)
        pred_val_eng = simulate(model_eng, val_seg, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol)
        pred_test_eng = simulate(model_eng, test_seg, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol)

        full_pred_phy = simulate(model_phy, full_seg, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol)
        full_pred_eng = simulate(model_eng, full_seg, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol)

    report = {
        "physics": {
            "train": eval_metrics(pred_train_phy, train_seg["x"]),
            "val": eval_metrics(pred_val_phy, val_seg["x"]),
            "test": eval_metrics(pred_test_phy, test_seg["x"]),
        },
        "engineering": {
            "train": eval_metrics(pred_train_eng, train_seg["x"]),
            "val": eval_metrics(pred_val_eng, val_seg["x"]),
            "test": eval_metrics(pred_test_eng, test_seg["x"]),
        },
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    }

    with open(os.path.join(out_dir, "metrics_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    torch.save(model_phy.state_dict(), os.path.join(out_dir, "pinn_physics.pth"))
    torch.save(model_eng.state_dict(), os.path.join(out_dir, "pinn_engineering.pth"))

    df_pred = pd.DataFrame(
        {
            "time": data["time"].values,
            "rpm_obs": data["RPM"].values,
            "map_obs": data["MAP"].values,
            "rpm_phy": full_pred_phy[:, 0].detach().cpu().numpy(),
            "map_phy": full_pred_phy[:, 1].detach().cpu().numpy(),
            "rpm_eng": full_pred_eng[:, 0].detach().cpu().numpy(),
            "map_eng": full_pred_eng[:, 1].detach().cpu().numpy(),
        }
    )
    df_pred.to_csv(os.path.join(out_dir, "full_predictions.csv"), index=False)

    plot_2d_3d(data, full_pred_phy, full_pred_eng, out_dir)
    plot_upgrade(data, model_phy, model_eng, full_seg, cfg, out_dir)

    print("Rodando XAI: MC dropout, NLL-noise, counterfactual, feature-missing e sensitivity...")
    run_mc_dropout(model_phy, test_seg, cfg, out_dir, "pinn_physics")
    run_mc_dropout(model_eng, test_seg, cfg, out_dir, "pinn_engineering")

    run_nll_noise(model_phy, test_seg, cfg, out_dir, "pinn_physics")
    run_nll_noise(model_eng, test_seg, cfg, out_dir, "pinn_engineering")

    run_counterfactual(model_phy, test_seg, cfg, out_dir, "pinn_physics", ve_eq_str)
    run_counterfactual(model_eng, test_seg, cfg, out_dir, "pinn_engineering", ve_eq_str)

    run_feature_missing(model_phy, train_seg, test_seg, cfg, out_dir, "pinn_physics")
    run_feature_missing(model_eng, train_seg, test_seg, cfg, out_dir, "pinn_engineering")

    run_sensitivity(model_phy, test_seg, cfg, out_dir, "pinn_physics")
    run_sensitivity(model_eng, test_seg, cfg, out_dir, "pinn_engineering")

    hist_phy.tail(1).to_csv(os.path.join(out_dir, "pinn_physics_last_epoch.csv"), index=False)
    hist_eng.tail(1).to_csv(os.path.join(out_dir, "pinn_engineering_last_epoch.csv"), index=False)

    print("Pipeline finalizado.")
    print(f"Resultados em: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Research pipeline with two PINNs + XAI")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lambda-dyn", type=float, default=0.20)
    parser.add_argument("--lambda-ve", type=float, default=0.05)
    parser.add_argument("--lambda-smooth", type=float, default=1e-4)
    parser.add_argument("--lambda-bounds", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=140)
    parser.add_argument("--method", type=str, default="dopri5")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lambda_dyn=args.lambda_dyn,
        lambda_ve=args.lambda_ve,
        lambda_smooth=args.lambda_smooth,
        lambda_bounds=args.lambda_bounds,
        grad_clip=args.grad_clip,
        patience=args.patience,
        method=args.method,
        device=args.device,
        seed=args.seed,
    )


def count_trainable_params(model: nn.Module):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
