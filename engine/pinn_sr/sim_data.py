import numpy as np
import pandas as pd

def generate_synth(duration_sec=6, fs=100):
    np.random.seed(42)
    n_samples = duration_sec * fs  
    t = np.linspace(0, duration_sec, n_samples)
    
   
    map_signal = 1.0 + 1.2 * (1 - np.exp(-2 * t))
    dmap_dt_truth = 2.4 * np.exp(-2 * t)
    tps = np.full(n_samples, 100.0)
    ignition = 20 - 5 * (map_signal - 1)
    lambda_val = 0.82 + 0.05 * np.random.randn(n_samples) 
    ve_profile = 0.85 + 0.1 * np.sin(np.linspace(0, 3, n_samples))
    
    torque_kgfm = 35 + map_signal * ve_profile * (ignition / 20)
    
    drpm_dt = (torque_kgfm * 7162) / (2000 + 500 * t)
    rpm = 2000 + np.cumsum(drpm_dt) * (1/fs) 
    
    rpm_noisy = rpm + np.random.normal(0, 15, n_samples)
    map_noisy = map_signal + np.random.normal(0,0.02, n_samples)
    
    df = pd.DataFrame({
        'time': t,
        'RPM': rpm_noisy,
        'MAP': map_noisy,
        'tps': tps,
        'Ignition': ignition,
        'Lambda': lambda_val,
        'dRPM_dt_truth': drpm_dt,
        'dMAP_dt_truth': dmap_dt_truth,
    })
    
    return df
data = generate_synth()
data.to_csv('data/synth_data.csv', index=False)
print(data.head())