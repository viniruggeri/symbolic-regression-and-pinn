import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Criação do diretório 4s8p, caso não exista
output_dir = '4s8p'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# =========================
# 1) Dados da matriz
# =========================
data = pd.DataFrame({
    "cenario": ["200", "201", "204", "302", "400", "401", "403", "404", "408", "409", "422", "429", "499", "500", "502", "503", "504"],
    "impacto": [9, 7, 4, 8, 7, 8, 8, 6, 7, 5, 7, 8, 8, 10, 9, 9, 9],
    "custo": [2, 3, 1, 2, 3, 2, 3, 2, 3, 2, 3, 4, 3, 6, 5, 5, 5],
    "risco": [9, 6, 3, 8, 6, 8, 7, 5, 7, 4, 7, 9, 8, 10, 9, 9, 9]
})

print("=== MATRIZ ===")
print(data)

# =========================
# 2) Pareto Frontier
# maximiza impacto e risco
# minimiza custo
# =========================
def is_dominated(row_i, row_j):
    better_or_equal = (
        row_j["impacto"] >= row_i["impacto"] and
        row_j["risco"] >= row_i["risco"] and
        row_j["custo"] <= row_i["custo"]
    )
    strictly_better = (
        row_j["impacto"] > row_i["impacto"] or
        row_j["risco"] > row_i["risco"] or
        row_j["custo"] < row_i["custo"]
    )
    return better_or_equal and strictly_better

def pareto_front(df):
    pareto_idx = []
    for i, row_i in df.iterrows():
        dominated = False
        for j, row_j in df.iterrows():
            if i == j:
                continue
            if is_dominated(row_i, row_j):
                dominated = True
                break
        if not dominated:
            pareto_idx.append(i)
    return df.loc[pareto_idx].copy()

pareto_df = pareto_front(data)

print("\n=== FRENTE DE PARETO ===")
print(pareto_df.sort_values(["custo", "impacto", "risco"], ascending=[True, False, False]))

# =========================
# 3) Plot 2D
# =========================
plt.figure(figsize=(8, 6))
plt.scatter(data["custo"], data["impacto"], label="Todos os cenários")
plt.scatter(pareto_df["custo"], pareto_df["impacto"], label="Pareto", s=120)
for _, row in data.iterrows():
    plt.annotate(row["cenario"], (row["custo"], row["impacto"]), textcoords="offset points", xytext=(5, 5))
plt.xlabel("Custo")
plt.ylabel("Impacto")
plt.title("Fronteira de Pareto - 2D")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Salvar a imagem no diretório 4s8p
plt.savefig(f"{output_dir}/pareto_front_2d.png")
plt.show()

# =========================
# 4) Plot 3D
# =========================
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(data["custo"], data["impacto"], data["risco"], label="Todos os cenários")
ax.scatter(pareto_df["custo"], pareto_df["impacto"], pareto_df["risco"], label="Pareto", s=120)

for _, row in data.iterrows():
    ax.text(row["custo"], row["impacto"], row["risco"], str(row["cenario"]))

ax.set_xlabel("Custo")
ax.set_ylabel("Impacto")
ax.set_zlabel("Risco")
ax.set_title("Fronteira de Pareto - 3D")
ax.legend()
plt.tight_layout()

# Salvar a imagem no diretório 4s8p
plt.savefig(f"{output_dir}/pareto_front_3d.png")
plt.show()

# =========================
# 5) NSGA-II com pymoo
# aqui cada cenário é uma "solução"
# objetivo:
# minimizar -impacto
# minimizar -risco
# minimizar custo
# =========================
try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.termination import get_termination
    from pymoo.optimize import minimize

    class ScenarioSelectionProblem(Problem):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)
            super().__init__(
                n_var=1,
                n_obj=3,
                n_constr=0,
                xl=0,
                xu=len(self.df) - 1
            )

        def _evaluate(self, X, out, *args, **kwargs):
            idx = np.clip(np.round(X).astype(int).flatten(), 0, len(self.df) - 1)
            impacto = -self.df.loc[idx, "impacto"].to_numpy()
            risco = -self.df.loc[idx, "risco"].to_numpy()
            custo = self.df.loc[idx, "custo"].to_numpy()
            out["F"] = np.column_stack([impacto, risco, custo])

    problem = ScenarioSelectionProblem(data)

    algorithm = NSGA2(pop_size=40)
    termination = get_termination("n_gen", 80)

    res = minimize(
        problem,
        algorithm,
        termination,
        verbose=False,
        seed=42
    )

    X = np.round(res.X).astype(int).flatten()
    selected = data.iloc[np.unique(X)].copy()

    print("\n=== NSGA-II (cenários selecionados) ===")
    print(selected.sort_values(["custo", "impacto", "risco"], ascending=[True, False, False]))

    plt.figure(figsize=(8, 6))
    plt.scatter(data["custo"], data["impacto"], label="Todos os cenários")
    plt.scatter(selected["custo"], selected["impacto"], label="NSGA-II", s=120)
    for _, row in selected.iterrows():
        plt.annotate(row["cenario"], (row["custo"], row["impacto"]), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Custo")
    plt.ylabel("Impacto")
    plt.title("NSGA-II sobre a matriz")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Salvar a imagem no diretório 4s8p
    plt.savefig(f"{output_dir}/nsgaii_selected_scenarios.png")
    plt.show()

except ImportError:
    print("\n[pymoo não instalado]")
    print("Rode: pip install pymoo")
    plt.show()