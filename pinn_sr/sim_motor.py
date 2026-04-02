import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import json
import os


# 1. CARREGAMENTO
sindy_coeffs = np.load('models/sindy_coefficients.npy')
with open('models/ve_equation.txt', 'r') as f:
    ve_eq_str = f.read()


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

def get_ve_pysr(rpm, map_val):
    # Proteção de entrada
    x0, x1 = np.clip(rpm, 0, 8000), np.clip(map_val, 0, 4.0)
    safe_dict = {
        'x0': x0, 'x1': x1, 'sin': np.sin, 
        'exp': lambda v: np.exp(np.clip(v, -20, 2)), # Limite estrito no exp
        'log': lambda v: np.log(np.maximum(v, 1e-6))
    }
    try:
        return np.clip(eval(ve_eq_str, {"__builtins__": None}, safe_dict), 0.2, 1.2)
    except:
        return 0.85

# 2. MODELO DINÂMICO ESCALONADO
def ea888_system(t, x, boost_offset, control_fn=None):
    # x[0] é RPM, x[1] é MAP
    rpm = np.clip(x[0], 0, 8000)
    map_val = np.clip(x[1], 0.1, 4.0)
    
    # Se houver perfil de controle temporal, usamos os valores do instante t.
    if control_fn is None:
        u_ign, u_tps = 15.0, 100.0
    else:
        u_ign, u_tps = control_fn(float(t))
    
    # Reconstrução das features (Polinomial Grau 2)
    features = np.array([
        1, rpm, map_val, u_ign, u_tps,
        rpm**2, rpm*map_val, rpm*u_ign, rpm*u_tps,
        map_val**2, map_val*u_ign, map_val*u_tps,
        u_ign**2, u_ign*u_tps, u_tps**2
    ])
    
    # Derivadas
    drpm_dt = np.dot(sindy_coeffs[0], features)
    dmap_dt = np.dot(sindy_coeffs[1], features)
    
    # Adicionando o efeito do upgrade de forma estável
    drpm_dt += (boost_offset * 500) # Adiciona 500 RPM/s de aceleração extra por 1 bar
    
    # Limitação de segurança para o solver não divergir
    drpm_dt = np.clip(drpm_dt, -2000, 5000)
    dmap_dt = np.clip(dmap_dt, -2, 2)

    return [drpm_dt, dmap_dt]

# 3. INTEGRAÇÃO ROBUSTA (solve_ivp)
def simulate(boost_offset=0.0, x0=None, t_eval=None, method='Radau', control_fn=None):
    if x0 is None:
        x0 = [2000.0, 1.0]
    if t_eval is None:
        t_eval = np.linspace(0, 5, 500)
    t_span = (float(t_eval[0]), float(t_eval[-1]))
    return solve_ivp(
        ea888_system,
        t_span,
        x0,
        t_eval=t_eval,
        args=(boost_offset, control_fn),
        method=method,
    )


def build_control_fn(time, ignition, tps):
    t_ref = np.asarray(time)
    ign_ref = np.asarray(ignition)
    tps_ref = np.asarray(tps)

    def control_fn(t):
        u_ign = float(np.interp(t, t_ref, ign_ref))
        u_tps = float(np.interp(t, t_ref, tps_ref))
        return u_ign, u_tps

    return control_fn


def validate_equations_on_data(data_path='data/synth_data.csv'):
    if not os.path.exists(data_path):
        raise FileNotFoundError("Arquivo data/synth_data.csv não encontrado. Rode pinn_sr/sim_data.py primeiro.")

    df = pd.read_csv(data_path).sort_values('time').reset_index(drop=True)
    required_cols = ['time', 'RPM', 'MAP', 'Ignition', 'tps']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes em {data_path}: {missing}")

    t_eval = df['time'].values
    x0 = [float(df['RPM'].iloc[0]), float(df['MAP'].iloc[0])]

    control_fn = build_control_fn(
        df['time'].values,
        df['Ignition'].values,
        df['tps'].values,
    )
    sol = simulate(boost_offset=0.0, x0=x0, t_eval=t_eval, control_fn=control_fn)
    if not sol.success:
        raise RuntimeError(f"Falha na integração do modelo: {sol.message}")

    rpm_pred, map_pred = sol.y[0], sol.y[1]
    rpm_true, map_true = df['RPM'].values, df['MAP'].values

    ve_pred = np.array([get_ve_pysr(r, m) for r, m in zip(rpm_pred, map_pred)])
    ve_true = np.array([get_ve_pysr(r, m) for r, m in zip(rpm_true, map_true)])

    metrics = {
        'rpm_rmse': rmse(rpm_true, rpm_pred),
        'map_rmse': rmse(map_true, map_pred),
        've_rmse': rmse(ve_true, ve_pred),
        'rpm_max_abs': float(np.max(np.abs(rpm_true - rpm_pred))),
        'map_max_abs': float(np.max(np.abs(map_true - map_pred))),
    }

    # Critério simples de aprovação para liberar treino PINN.
    validation_ok = (
        metrics['rpm_rmse'] < 450.0
        and metrics['map_rmse'] < 0.25
        and metrics['ve_rmse'] < 0.08
    )

    os.makedirs('models', exist_ok=True)
    payload = {
        'status': 'approved' if validation_ok else 'rejected',
        'metrics': metrics,
        'thresholds': {'rpm_rmse': 450.0, 'map_rmse': 0.25, 've_rmse': 0.08},
        'source_data': data_path,
    }
    with open('models/equation_validation.json', 'w') as f:
        json.dump(payload, f, indent=2)

    out = pd.DataFrame(
        {
            'time': t_eval,
            'rpm_true': rpm_true,
            'map_true': map_true,
            'rpm_pred': rpm_pred,
            'map_pred': map_pred,
            've_true': ve_true,
            've_pred': ve_pred,
        }
    )
    os.makedirs('outputs', exist_ok=True)
    out.to_csv('outputs/sindy_sr_validation.csv', index=False)
    return payload

# 4. MÉTRICAS E PLOT
def plot_results(sol, label, color):
    rpms = sol.y[0]
    maps = sol.y[1]
    ves = np.array([get_ve_pysr(r, m) for r, m in zip(rpms, maps)])
    torque = 18.5 * maps * ves
    cv = (torque * rpms) / 716.2
    
    plt.subplot(1, 2, 1)
    plt.plot(sol.t, rpms, label=label, color=color)
    plt.subplot(1, 2, 2)
    plt.plot(rpms, cv, label=label, color=color)

plt.figure(figsize=(12, 5))
print("Validando equações descobertas (SINDy + SR)...")
validation = validate_equations_on_data('data/synth_data.csv')
print(f"Status da validação: {validation['status']}")
print(f"Métricas: {validation['metrics']}")

print("Simulando cenário stock e upgrade para inspeção visual...")
sol_orig = simulate(boost_offset=0.0)
sol_upgr = simulate(boost_offset=0.3)

plot_results(sol_orig, 'Stock', 'blue')
plot_results(sol_upgr, 'Upgrade (+0.3)', 'red')

plt.subplot(1, 2, 1); plt.title('Desenvolvimento RPM'); plt.legend()
plt.subplot(1, 2, 2); plt.title('Curva de Potência'); plt.legend()
plt.show()