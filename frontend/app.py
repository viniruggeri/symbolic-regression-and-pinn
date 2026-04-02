import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from pinn_sr import research_two_pinns as r2p
except Exception:
    r2p = None

st.set_page_config(page_title="Motor PINN Lab", layout="wide")
st.title("Motor PINN Research Dashboard")
st.caption("Sem black box: métricas, treino, simulação e explicabilidade lado a lado.")


def find_runs():
    if not OUTPUTS_DIR.exists():
        return []
    runs = [p for p in OUTPUTS_DIR.iterdir() if p.is_dir() and p.name.endswith("_two_pinns")]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def nearest_counterfactual(df: pd.DataFrame, boost: float, map_scale: float):
    if df is None or df.empty:
        return None
    d = np.sqrt((df["boost_offset"] - boost) ** 2 + (df["map_scale_init"] - map_scale) ** 2)
    idx = d.idxmin()
    return df.loc[idx]


@st.cache_resource(show_spinner=False)
def load_runtime_models(run_path_str: str):
    if r2p is None:
        return None

    run_path = Path(run_path_str)
    manifest = read_json(run_path / "training_manifest.json") or {}
    cfg_data = manifest.get("training", {}).get("config", {})

    hidden_dim = int(cfg_data.get("hidden_dim", 96))
    dropout = float(cfg_data.get("dropout", 0.10))
    method = manifest.get("training", {}).get("ode_solver", {}).get("method", "dopri5")
    rtol = float(manifest.get("training", {}).get("ode_solver", {}).get("rtol", 1e-5))
    atol = float(manifest.get("training", {}).get("ode_solver", {}).get("atol", 1e-6))

    data = r2p.load_data(str(ROOT / "data" / "synth_data.csv"))
    full_seg = r2p.to_segment(data, "cpu")

    sindy_coeffs = np.load(str(ROOT / "models" / "sindy_coefficients.npy"))
    model_phy = r2p.PhysicsInformedODE(sindy_coeffs, hidden_dim=hidden_dim, dropout=dropout).to("cpu")
    model_eng = r2p.EngineeringODE(hidden_dim=hidden_dim, dropout=dropout).to("cpu")

    model_phy.load_state_dict(torch.load(run_path / "pinn_physics.pth", map_location="cpu"))
    model_eng.load_state_dict(torch.load(run_path / "pinn_engineering.pth", map_location="cpu"))
    model_phy.eval()
    model_eng.eval()

    with open(ROOT / "models" / "ve_equation.txt", "r", encoding="utf-8") as f:
        ve_eq_str = f.read().strip()

    return {
        "model_phy": model_phy,
        "model_eng": model_eng,
        "full_seg": full_seg,
        "data": data,
        "method": method,
        "rtol": rtol,
        "atol": atol,
        "ve_eq_str": ve_eq_str,
    }


def run_live_sim(runtime, model_choice: str, boost: float, map_scale: float, rpm_delta: float, ign_scale: float, tps_scale: float):
    model = runtime["model_phy"] if model_choice == "physics" else runtime["model_eng"]
    seg = runtime["full_seg"]

    x0 = seg["x0"].clone()
    x0[0] = torch.clamp(x0[0] + float(rpm_delta), min=0.0, max=8000.0)
    x0[1] = torch.clamp(x0[1] * float(map_scale), min=0.05, max=4.0)

    ign = seg["ign"] * float(ign_scale)
    tps = seg["tps"] * float(tps_scale)
    ctrl = r2p.ControlProfile(seg["t_rel"], ign, tps)

    with torch.no_grad():
        pred = r2p.simulate(
            model,
            seg,
            boost_offset=float(boost),
            train_mode=False,
            method=runtime["method"],
            rtol=runtime["rtol"],
            atol=runtime["atol"],
            x0_override=x0,
            control_override=ctrl,
        )
        ve = r2p.ve_torch(pred[:, 0], pred[:, 1], runtime["ve_eq_str"])
        torque = 18.5 * pred[:, 1] * ve
        cv = (torque * pred[:, 0]) / 716.2

    return {
        "time": seg["t_rel"].cpu().numpy(),
        "pred": pred.cpu().numpy(),
        "cv": cv.cpu().numpy(),
    }


runs = find_runs()
if not runs:
    st.error("Nenhum run *_two_pinns encontrado em outputs/. Rode o pipeline research_two_pinns primeiro.")
    st.stop()

run_names = [r.name for r in runs]
default_idx = 0
selected_name = st.sidebar.selectbox("Run", run_names, index=default_idx)
run_dir = OUTPUTS_DIR / selected_name

st.sidebar.markdown("### Caminho")
st.sidebar.code(str(run_dir))

metrics = read_json(run_dir / "metrics_report.json")
manifest = read_json(run_dir / "training_manifest.json")
full_preds = read_csv(run_dir / "full_predictions.csv")
hist_phy = read_csv(run_dir / "pinn_physics_history.csv")
hist_eng = read_csv(run_dir / "pinn_engineering_history.csv")
cf_phy = read_csv(run_dir / "pinn_physics_counterfactual.csv")
cf_eng = read_csv(run_dir / "pinn_engineering_counterfactual.csv")
miss_phy = read_csv(run_dir / "pinn_physics_feature_missing.csv")
miss_eng = read_csv(run_dir / "pinn_engineering_feature_missing.csv")
nll_phy = read_csv(run_dir / "pinn_physics_nll_noise.csv")
nll_eng = read_csv(run_dir / "pinn_engineering_nll_noise.csv")
mc_phy = read_csv(run_dir / "pinn_physics_mc_dropout.csv")
mc_eng = read_csv(run_dir / "pinn_engineering_mc_dropout.csv")
sens_phy = read_json(run_dir / "pinn_physics_sensitivity.json")
sens_eng = read_json(run_dir / "pinn_engineering_sensitivity.json")


tab_overview, tab_training, tab_sim, tab_xai, tab_explain = st.tabs(
    ["Resumo", "Treino", "Simulador Upgrade", "XAI", "Como foi treinada"]
)

with tab_overview:
    st.subheader("Métricas train/val/test")
    if metrics is None:
        st.warning("Arquivo metrics_report.json não encontrado.")
    else:
        split = metrics.get("split_sizes", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Train", split.get("train", "-"))
        c2.metric("Val", split.get("val", "-"))
        c3.metric("Test", split.get("test", "-"))

        rows = []
        for model_key in ["physics", "engineering"]:
            for split_name in ["train", "val", "test"]:
                v = metrics.get(model_key, {}).get(split_name, {})
                rows.append(
                    {
                        "model": model_key,
                        "split": split_name,
                        "rpm_rmse": v.get("rpm_rmse"),
                        "map_rmse": v.get("map_rmse"),
                        "rpm_mae": v.get("rpm_mae"),
                        "map_mae": v.get("map_mae"),
                    }
                )
        dfm = pd.DataFrame(rows)
        st.dataframe(dfm, use_container_width=True)

    st.subheader("Comparação observação vs predição")
    if full_preds is None:
        st.warning("Arquivo full_predictions.csv não encontrado.")
    else:
        fig_rpm = go.Figure()
        fig_rpm.add_trace(go.Scatter(x=full_preds["time"], y=full_preds["rpm_obs"], name="rpm_obs", line=dict(color="black")))
        fig_rpm.add_trace(go.Scatter(x=full_preds["time"], y=full_preds["rpm_phy"], name="rpm_phy", line=dict(color="#d62728")))
        fig_rpm.add_trace(go.Scatter(x=full_preds["time"], y=full_preds["rpm_eng"], name="rpm_eng", line=dict(color="#1f77b4")))
        fig_rpm.update_layout(height=350, title="RPM")
        st.plotly_chart(fig_rpm, use_container_width=True)

        fig_map = go.Figure()
        fig_map.add_trace(go.Scatter(x=full_preds["time"], y=full_preds["map_obs"], name="map_obs", line=dict(color="black")))
        fig_map.add_trace(go.Scatter(x=full_preds["time"], y=full_preds["map_phy"], name="map_phy", line=dict(color="#d62728")))
        fig_map.add_trace(go.Scatter(x=full_preds["time"], y=full_preds["map_eng"], name="map_eng", line=dict(color="#1f77b4")))
        fig_map.update_layout(height=350, title="MAP")
        st.plotly_chart(fig_map, use_container_width=True)

        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter3d(x=full_preds["time"], y=full_preds["rpm_obs"], z=full_preds["map_obs"], mode="lines", name="obs", line=dict(color="black")))
        fig3d.add_trace(go.Scatter3d(x=full_preds["time"], y=full_preds["rpm_phy"], z=full_preds["map_phy"], mode="lines", name="physics", line=dict(color="#d62728")))
        fig3d.add_trace(go.Scatter3d(x=full_preds["time"], y=full_preds["rpm_eng"], z=full_preds["map_eng"], mode="lines", name="engineering", line=dict(color="#1f77b4")))
        fig3d.update_layout(height=500, title="Trajetória 3D: tempo x RPM x MAP")
        st.plotly_chart(fig3d, use_container_width=True)

with tab_training:
    st.subheader("Curvas de treino")
    col_a, col_b = st.columns(2)

    if hist_phy is not None:
        fig = px.line(hist_phy, x="epoch", y=["train_total", "val_total", "train_data"], title="PINN física")
        col_a.plotly_chart(fig, use_container_width=True)
    else:
        col_a.warning("pinn_physics_history.csv ausente")

    if hist_eng is not None:
        fig = px.line(hist_eng, x="epoch", y=["train_total", "val_total", "train_data"], title="PINN engenharia")
        col_b.plotly_chart(fig, use_container_width=True)
    else:
        col_b.warning("pinn_engineering_history.csv ausente")

    if hist_phy is not None and "lr" in hist_phy.columns:
        st.plotly_chart(px.line(hist_phy, x="epoch", y="lr", title="Learning rate schedule (física)"), use_container_width=True)

with tab_sim:
    st.subheader("Simulador de upgrade")
    st.caption("Agora em modo realtime: executa inferência direta com os pesos da PINN selecionada.")

    runtime = load_runtime_models(str(run_dir))
    c1, c2, c3 = st.columns(3)
    boost = c1.slider("boost_offset", min_value=0.0, max_value=0.5, value=0.3, step=0.05)
    map_scale = c2.slider("map_scale_init", min_value=0.7, max_value=1.3, value=1.0, step=0.05)
    model_choice = c3.radio("Modelo", options=["physics", "engineering"], horizontal=True)

    c4, c5, c6 = st.columns(3)
    rpm_delta = c4.slider("rpm0_delta", min_value=-500.0, max_value=500.0, value=0.0, step=25.0)
    ign_scale = c5.slider("ignition_scale", min_value=0.7, max_value=1.3, value=1.0, step=0.05)
    tps_scale = c6.slider("tps_scale", min_value=0.7, max_value=1.3, value=1.0, step=0.05)

    do_run = st.button("Simular upgrade em tempo real", type="primary")

    if runtime is None:
        st.warning("Nao foi possivel carregar runtime das PINNs. Verifique dependencias do backend.")
    elif do_run:
        live = run_live_sim(
            runtime=runtime,
            model_choice=model_choice,
            boost=boost,
            map_scale=map_scale,
            rpm_delta=rpm_delta,
            ign_scale=ign_scale,
            tps_scale=tps_scale,
        )

        t = live["time"]
        pred = live["pred"]
        cv = live["cv"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RPM final", f"{pred[-1, 0]:.2f}")
        m2.metric("MAP final", f"{pred[-1, 1]:.4f}")
        m3.metric("RPM pico", f"{pred[:, 0].max():.2f}")
        m4.metric("CV pico", f"{cv.max():.2f}")

        fig_live = go.Figure()
        fig_live.add_trace(go.Scatter(x=t, y=pred[:, 0], name="RPM", line=dict(color="#2ca02c")))
        fig_live.add_trace(go.Scatter(x=t, y=pred[:, 1], name="MAP", yaxis="y2", line=dict(color="#ff7f0e")))
        fig_live.update_layout(
            title=f"Simulacao realtime - {model_choice}",
            xaxis=dict(title="tempo"),
            yaxis=dict(title="RPM"),
            yaxis2=dict(title="MAP", overlaying="y", side="right"),
            height=400,
        )
        st.plotly_chart(fig_live, use_container_width=True)

        fig_cv = px.line(x=t, y=cv, title="Potencia estimada (CV)")
        fig_cv.update_traces(line_color="#9467bd")
        st.plotly_chart(fig_cv, use_container_width=True)

    table = cf_phy if model_choice == "physics" else cf_eng
    pick = nearest_counterfactual(table, boost, map_scale)

    if pick is None:
        st.warning("Counterfactual CSV não encontrado para o modelo selecionado.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("RPM final", f"{pick['rpm_final']:.2f}")
        m2.metric("MAP final", f"{pick['map_final']:.4f}")
        m3.metric("CV pico", f"{pick['cv_peak']:.2f}")

        st.info(
            f"Cenário mais próximo no grid: boost={pick['boost_offset']:.2f}, map_scale={pick['map_scale_init']:.2f}"
        )

    if cf_phy is not None and not cf_phy.empty:
        piv = cf_phy.pivot_table(index="boost_offset", columns="map_scale_init", values="cv_peak")
        heat = px.imshow(piv, text_auto=True, aspect="auto", title="CV pico - PINN física")
        st.plotly_chart(heat, use_container_width=True)

with tab_xai:
    st.subheader("MC Dropout e incerteza")
    cols = st.columns(2)
    if mc_phy is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mc_phy["time"], y=mc_phy["rpm_mean"], name="mean", line=dict(color="#d62728")))
        fig.add_trace(go.Scatter(x=mc_phy["time"], y=mc_phy["rpm_mean"] + 2 * mc_phy["rpm_std"], name="+2std", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=mc_phy["time"], y=mc_phy["rpm_mean"] - 2 * mc_phy["rpm_std"], name="-2std", fill="tonexty", line=dict(width=0), fillcolor="rgba(214,39,40,0.2)", showlegend=False))
        fig.update_layout(height=350, title="PINN física - intervalo MC")
        cols[0].plotly_chart(fig, use_container_width=True)
    if mc_eng is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mc_eng["time"], y=mc_eng["rpm_mean"], name="mean", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=mc_eng["time"], y=mc_eng["rpm_mean"] + 2 * mc_eng["rpm_std"], name="+2std", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=mc_eng["time"], y=mc_eng["rpm_mean"] - 2 * mc_eng["rpm_std"], name="-2std", fill="tonexty", line=dict(width=0), fillcolor="rgba(31,119,180,0.2)", showlegend=False))
        fig.update_layout(height=350, title="PINN engenharia - intervalo MC")
        cols[1].plotly_chart(fig, use_container_width=True)

    st.subheader("NLL sob ruído")
    c1, c2 = st.columns(2)
    if nll_phy is not None:
        c1.plotly_chart(px.line(nll_phy, x="noise_level", y=["joint_nll", "rpm_nll", "map_nll"], title="NLL - física"), use_container_width=True)
    if nll_eng is not None:
        c2.plotly_chart(px.line(nll_eng, x="noise_level", y=["joint_nll", "rpm_nll", "map_nll"], title="NLL - engenharia"), use_container_width=True)

    st.subheader("Feature missing impact")
    c3, c4 = st.columns(2)
    if miss_phy is not None:
        c3.plotly_chart(px.bar(miss_phy, x="scenario", y=["delta_rpm_rmse", "delta_map_rmse"], barmode="group", title="Impacto - física"), use_container_width=True)
    if miss_eng is not None:
        c4.plotly_chart(px.bar(miss_eng, x="scenario", y=["delta_rpm_rmse", "delta_map_rmse"], barmode="group", title="Impacto - engenharia"), use_container_width=True)

    st.subheader("Sensibilidade local")
    s1, s2 = st.columns(2)
    if sens_phy is not None:
        s1.json(sens_phy)
    if sens_eng is not None:
        s2.json(sens_eng)

with tab_explain:
    st.subheader("Por que não é black box")
    if manifest is None:
        st.warning("training_manifest.json não encontrado neste run.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Treino e solver**")
            st.json(manifest.get("training", {}))
        with c2:
            st.markdown("**Modelos e parâmetros**")
            st.json(manifest.get("models", {}))

        st.markdown("**XAI aplicado**")
        st.json(manifest.get("xai", {}))

        st.markdown("**Arquivos-chave deste run**")
        st.write("- metrics_report.json")
        st.write("- training_manifest.json")
        st.write("- pinn_physics_history.csv / pinn_engineering_history.csv")
        st.write("- counterfactual, feature_missing, nll_noise, mc_dropout e sensitivity")
