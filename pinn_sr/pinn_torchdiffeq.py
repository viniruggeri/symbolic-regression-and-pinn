import argparse
import os
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torchdiffeq import odeint
except ImportError as exc:
    raise ImportError(
        "torchdiffeq is required. Install with: pip install torchdiffeq"
    ) from exc


@dataclass
class TrainConfig:
    epochs: int = 1200
    lr: float = 1e-3
    lambda_phys: float = 0.1
    lambda_ve: float = 0.05
    hidden_dim: int = 64
    method: str = "dopri5"
    rtol: float = 1e-5
    atol: float = 1e-6
    device: str = "cpu"


class HybridEA888ODE(nn.Module):
    """Hybrid ODE: SINDy base + trainable residual correction."""

    def __init__(self, sindy_coeffs: np.ndarray, hidden_dim: int = 64):
        super().__init__()
        coeffs = torch.tensor(sindy_coeffs, dtype=torch.float32)
        if coeffs.shape != (2, 15):
            raise ValueError(
                f"Expected SINDy coeffs shape (2, 15), got {tuple(coeffs.shape)}"
            )

        self.register_buffer("sindy_coeffs", coeffs)
        self.register_buffer("u_ign", torch.tensor(15.0, dtype=torch.float32))
        self.register_buffer("u_tps", torch.tensor(100.0, dtype=torch.float32))

        self.residual_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        rpm = torch.clamp(x[:, 0], 0.0, 8000.0)
        map_val = torch.clamp(x[:, 1], 0.1, 4.0)

        u_ign = self.u_ign.expand_as(rpm)
        u_tps = self.u_tps.expand_as(rpm)

        feats = torch.stack(
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
        return feats

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        feats = self._features(x)
        base_dx = feats @ self.sindy_coeffs.T

        # Scale state before residual network for stable training.
        x_scaled = torch.stack(
            [
                torch.clamp(x[:, 0], 0.0, 8000.0) / 8000.0,
                torch.clamp(x[:, 1], 0.1, 4.0) / 4.0,
            ],
            dim=1,
        )
        residual_dx = self.residual_net(x_scaled)

        dx = base_dx + residual_dx
        drpm_dt = torch.clamp(dx[:, 0], -2500.0, 5000.0)
        dmap_dt = torch.clamp(dx[:, 1], -2.5, 2.5)
        return torch.stack([drpm_dt, dmap_dt], dim=1)


def _to_device(*tensors: torch.Tensor, device: str):
    return [t.to(device) for t in tensors]


def load_training_data(csv_path: str):
    df = pd.read_csv(csv_path)
    required_cols = ["time", "RPM", "MAP"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    df = df.sort_values("time").reset_index(drop=True)

    t = torch.tensor(df["time"].values, dtype=torch.float32)
    x = torch.tensor(df[["RPM", "MAP"]].values, dtype=torch.float32)

    dt = torch.diff(t)
    if torch.any(dt <= 0):
        raise ValueError("Time vector must be strictly increasing")

    dx_dt_obs = torch.diff(x, dim=0) / dt.unsqueeze(1)
    return t, x, dx_dt_obs


def check_equation_validation(path: str = "models/equation_validation.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Missing models/equation_validation.json. "
            "Run pinn_sr/sim_motor.py to validate discovered equations before PINN training."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("status") != "approved":
        raise RuntimeError(
            "Equation validation is not approved. "
            "Review models/equation_validation.json and improve SINDy/SR first."
        )
    return payload


def ve_numpy(rpm: np.ndarray, map_val: np.ndarray, ve_eq_str: str) -> np.ndarray:
    x0 = np.clip(rpm, 0, 8000)
    x1 = np.clip(map_val, 0, 4.0)
    safe_dict = {
        "x0": x0,
        "x1": x1,
        "sin": np.sin,
        "exp": lambda v: np.exp(np.clip(v, -20, 2)),
        "log": lambda v: np.log(np.maximum(v, 1e-6)),
    }
    try:
        ve = eval(ve_eq_str, {"__builtins__": None}, safe_dict)
        return np.clip(np.asarray(ve, dtype=np.float32), 0.2, 1.2)
    except Exception:
        return np.full_like(x0, 0.85, dtype=np.float32)


def train(config: TrainConfig):
    sindy_path = "models/sindy_coefficients.npy"
    data_path = "data/synth_data.csv"

    if not os.path.exists(sindy_path):
        raise FileNotFoundError(
            "Missing models/sindy_coefficients.npy. Run pinn_sr/sr_sindy.py first."
        )
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Missing data/synth_data.csv. Run pinn_sr/sim_data.py first."
        )

    check_equation_validation("models/equation_validation.json")

    ve_eq_path = "models/ve_equation.txt"
    if not os.path.exists(ve_eq_path):
        raise FileNotFoundError(
            "Missing models/ve_equation.txt. Run pinn_sr/sr_sindy.py first."
        )
    with open(ve_eq_path, "r", encoding="utf-8") as f:
        ve_eq_str = f.read().strip()

    sindy_coeffs = np.load(sindy_path)
    t, x_obs, dx_dt_obs = load_training_data(data_path)
    ve_obs = ve_numpy(x_obs[:, 0].numpy(), x_obs[:, 1].numpy(), ve_eq_str)
    ve_obs = torch.tensor(ve_obs, dtype=torch.float32)

    x0 = x_obs[0]

    model = HybridEA888ODE(sindy_coeffs=sindy_coeffs, hidden_dim=config.hidden_dim).to(
        config.device
    )

    t, x_obs, dx_dt_obs, x0, ve_obs = _to_device(
        t, x_obs, dx_dt_obs, x0, ve_obs, device=config.device
    )

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()

        pred_traj = odeint(
            model,
            x0,
            t,
            method=config.method,
            rtol=config.rtol,
            atol=config.atol,
        )

        data_loss = torch.mean((pred_traj - x_obs) ** 2)

        model_dx = model(torch.tensor(0.0, device=config.device), x_obs[:-1])
        phys_loss = torch.mean((model_dx - dx_dt_obs) ** 2)

        # VE consistency using symbolic regression equation evaluated on predicted trajectory.
        pred_rpm = torch.clamp(pred_traj[:, 0], 0.0, 8000.0)
        pred_map = torch.clamp(pred_traj[:, 1], 0.1, 4.0)
        ve_pred = torch.tensor(
            ve_numpy(pred_rpm.detach().cpu().numpy(), pred_map.detach().cpu().numpy(), ve_eq_str),
            dtype=torch.float32,
            device=config.device,
        )
        ve_loss = torch.mean((ve_pred - ve_obs) ** 2)

        loss = data_loss + config.lambda_phys * phys_loss + config.lambda_ve * ve_loss
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 100 == 0:
            print(
                f"Epoch {epoch:04d} | total={loss.item():.6f} "
                f"data={data_loss.item():.6f} phys={phys_loss.item():.6f} ve={ve_loss.item():.6f}"
            )

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/pinn_torchdiffeq.pth")

    with torch.no_grad():
        pred_final = odeint(
            model,
            x0,
            t,
            method=config.method,
            rtol=config.rtol,
            atol=config.atol,
        ).cpu().numpy()

    out_df = pd.DataFrame(
        {
            "time": t.cpu().numpy(),
            "rpm_obs": x_obs[:, 0].cpu().numpy(),
            "map_obs": x_obs[:, 1].cpu().numpy(),
            "rpm_pred": pred_final[:, 0],
            "map_pred": pred_final[:, 1],
        }
    )
    os.makedirs("outputs", exist_ok=True)
    out_df.to_csv("outputs/pinn_torchdiffeq_fit.csv", index=False)

    print("Training complete.")
    print("Saved model: models/pinn_torchdiffeq.pth")
    print("Saved fit curve: outputs/pinn_torchdiffeq_fit.csv")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train hybrid PINN with torchdiffeq")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-phys", type=float, default=0.1)
    parser.add_argument("--lambda-ve", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--method", type=str, default="dopri5")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    return TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        lambda_phys=args.lambda_phys,
        lambda_ve=args.lambda_ve,
        hidden_dim=args.hidden_dim,
        method=args.method,
        device=args.device,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
