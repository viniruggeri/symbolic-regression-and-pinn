import os
import pickle
import urllib.request
import numpy as np
import pandas as pd
import torch
import pysindy as ps
from pysr import PySRRegressor

# 1. Configuração de Ambiente (Evitar Erro 403)
opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# 2. Carregamento e Preparação de Dados
if not os.path.exists('data/synth_data.csv'):
    raise FileNotFoundError("Gere os dados sintéticos primeiro!")

df = pd.read_csv('data/synth_data.csv')
t_vector = df['time'].values 
X = df[['RPM', 'MAP']].values
U = df[['Ignition', 'tps']].values 

x_dot_truth = None
if {'dRPM_dt_truth', 'dMAP_dt_truth'}.issubset(df.columns):
    x_dot_truth = df[['dRPM_dt_truth', 'dMAP_dt_truth']].values

# 3. Treinamento SINDy (Dinâmica do Motor)
print("\n--- Treinando SINDy ---")
diff_method = ps.SmoothedFiniteDifference()
feature_lib = ps.PolynomialLibrary(degree=2, include_interaction=True)
optm = ps.STLSQ(threshold=1e-5)

model_sindy = ps.SINDy(
    optimizer=optm,
    feature_library=feature_lib,
    differentiation_method=diff_method
)

if x_dot_truth is not None:
    model_sindy.fit(X, t=t_vector, u=U, x_dot=x_dot_truth)
else:
    model_sindy.fit(X, t=t_vector, u=U)
model_sindy.print()

# 4. Treinamento PySR (Eficiência Volumétrica)
print("\n--- Treinando PySR ---")
# Criando um target realista baseado no RPM para o PySR descobrir
ve_target = (0.85 + 0.1 * np.sin(df['RPM'] / 1000)).values.astype('float32')
X_sr = df[['RPM', 'MAP']].values.astype('float32')

model_sr = PySRRegressor(
    niterations=50, # Reduzi para teste rápido, aumente se necessário
    binary_operators=['+', '-', '*', '/'],
    unary_operators=['sin', 'exp', 'log'],
    maxsize=20,
    model_selection="best"
)

model_sr.fit(X_sr, ve_target)
print(f"Melhor equação PySR: {model_sr.get_best()}")

# 5. Exportação e Salvamento (Sem duplicatas)
if not os.path.exists('models'):
    os.makedirs('models')

print("\n--- Salvando Artefatos ---")

# --- SALVANDO O PYSR (VE) ---
# 1. Jeito padrão Python (Pickle) para o regressor completo
with open('models/pysr_ve_model.pkl', 'wb') as f:
    pickle.dump(model_sr, f)

# 2. Exportando a melhor equação como PyTorch
pytorch_ve_model = model_sr.pytorch() 

# CORREÇÃO AQUI: Em vez de torch.save(model), usamos jit.script
try:
    model_scripted = torch.jit.script(pytorch_ve_model)
    model_scripted.save('models/ve_layer_pytorch.pth')
    print("✅ Modelo PySR salvo com sucesso via JIT!")
except Exception as e:
    # Se o JIT falhar por causa da complexidade da equação, 
    # salvamos a equação como string para carregar manualmente
    best_eq = model_sr.get_best()['equation']
    with open('models/ve_equation.txt', 'w') as f:
        f.write(best_eq)
    print("⚠️ JIT falhou, equação salva como texto em 'models/ve_equation.txt'")

# Salvando SINDy (Coeficientes e Metadados)
np.save('models/sindy_coefficients.npy', model_sindy.coefficients())

feature_names = model_sindy.get_feature_names()
with open('models/sindy_features.txt', 'w') as f:
    f.write(",".join(feature_names))

print("✅ Tudo pronto! Modelos físicos salvos na pasta /models")