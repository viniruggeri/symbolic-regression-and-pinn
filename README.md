# Symbolic Regression & PINN — Motor EA888

> **Repositório de estudo** sobre Programação Genética, Regressão Simbólica e Redes Neurais Informadas pela Física (PINN), aplicados à modelagem dinâmica de um motor de combustão interna (VW EA888).

---

## Sumário

1. [Visão Geral](#visão-geral)
2. [Contexto do Domínio](#contexto-do-domínio)
3. [Estrutura do Repositório](#estrutura-do-repositório)
4. [Pipeline Completo](#pipeline-completo)
5. [Módulos em Detalhe](#módulos-em-detalhe)
   - [sim_data.py — Geração de Dados Sintéticos](#sim_datapy--geração-de-dados-sintéticos)
   - [sr_sindy.py — SINDy + Regressão Simbólica (PySR)](#sr_sindypy--sindy--regressão-simbólica-pysr)
   - [sim_motor.py — Simulação e Validação do Motor](#sim_motorpy--simulação-e-validação-do-motor)
   - [pinn_torchdiffeq.py — PINN Híbrida (SINDy + Neural ODE)](#pinn_torchdiffeqpy--pinn-híbrida-sindy--neural-ode)
   - [research_two_pinns.py — Estudo Comparativo: Dois PINNs](#research_two_pinnspy--estudo-comparativo-dois-pinns)
   - [frontend/app.py — Dashboard Streamlit](#frontendapppy--dashboard-streamlit)
6. [Arquitetura dos Modelos](#arquitetura-dos-modelos)
   - [SINDy — Identificação Esparsa da Dinâmica](#sindy--identificação-esparsa-da-dinâmica)
   - [PySR / Programação Genética — Regressão Simbólica](#pysr--programação-genética--regressão-simbólica)
   - [PINN 1 — PhysicsInformedODE](#pinn-1--physicsinformedode)
   - [PINN 2 — EngineeringODE](#pinn-2--engineeringode)
7. [XAI e Incerteza](#xai-e-incerteza)
8. [Artefatos Salvos](#artefatos-salvos)
9. [Instalação e Dependências](#instalação-e-dependências)
10. [Como Executar](#como-executar)
11. [Resultados de Validação](#resultados-de-validação)

---

## Visão Geral

Este repositório explora três abordagens complementares para **descoberta de modelos físicos a partir de dados**:

| Técnica | Biblioteca | Objetivo |
|---|---|---|
| **SINDy** (Sparse Identification of Nonlinear Dynamics) | `pysindy` | Descobrir equações diferenciais esparsas que governam RPM e MAP |
| **Regressão Simbólica / Programação Genética** | `PySR` | Encontrar uma expressão simbólica fechada para a Eficiência Volumétrica (VE) |
| **PINN** (Physics-Informed Neural Network) | `PyTorch` + `torchdiffeq` | Treinar uma Neural ODE híbrida que incorpora as equações descobertas como restrições físicas |

O sistema modela a **dinâmica de aceleração** de um motor EA888, prevendo RPM e pressão no coletor (MAP) ao longo do tempo, com análise de incerteza e explicabilidade.

---

## Contexto do Domínio

O **motor Volkswagen EA888** é um motor 4 cilindros turbo de 1.8/2.0L presente em veículos como Golf GTI e Audi A3. O estudo usa as seguintes variáveis:

| Variável | Descrição | Faixa típica |
|---|---|---|
| `RPM` | Rotações por minuto | 0 – 8.000 rpm |
| `MAP` | Manifold Absolute Pressure (pressão no coletor) | 0.1 – 4.0 bar |
| `Ignition` | Avanço de ignição | ~10 – 25° |
| `tps` | Throttle Position Sensor (posição do acelerador) | 0 – 100% |
| `Lambda` | Relação ar/combustível normalizada | ~0.75 – 0.95 |
| `VE` | Volumetric Efficiency (eficiência volumétrica) | 0.2 – 1.2 |

A **eficiência volumétrica** é um coeficiente chave que relaciona quanto ar o motor realmente aspira em comparação com o volume do cilindro — quanto maior, mais torque e potência. O objetivo de longo prazo é comparar um motor **stock** com um motor com **upgrade de turbo** (`boost_offset > 0`).

---

## Estrutura do Repositório

```
symbolic-regression-and-pinn/
│
├── pinn_sr/                     # Código principal do pipeline
│   ├── sim_data.py              # 1. Gera dados sintéticos do motor
│   ├── sr_sindy.py              # 2. Treina SINDy e PySR (descoberta de equações)
│   ├── sim_motor.py             # 3. Valida equações e simula cenários de upgrade
│   ├── pinn_torchdiffeq.py      # 4a. PINN simples com torchdiffeq (SINDy + residual NN)
│   └── research_two_pinns.py    # 4b. Pesquisa avançada: dois PINNs + XAI + incerteza
│
├── frontend/
│   ├── app.py                   # Dashboard Streamlit interativo
│   └── requirements.txt         # Dependências do frontend
│
├── models/                      # Artefatos treinados (gerados pelo pipeline)
│   ├── sindy_coefficients.npy   # Coeficientes SINDy (matriz 2×15)
│   ├── sindy_features.txt       # Nomes das features do espaço polinomial
│   ├── pysr_ve_model.pkl        # Modelo PySR serializado (pickle)
│   ├── ve_layer_pytorch.pth     # Modelo VE exportado via TorchScript (quando possível)
│   ├── ve_equation.txt          # Melhor equação VE como string (fallback do JIT)
│   ├── equation_validation.json # Métricas e status de aprovação das equações
│   └── pinn_torchdiffeq.pth     # Pesos da PINN treinada (pinn_torchdiffeq.py)
│
└── data/                        # Dados de treino (gerados por sim_data.py)
    └── synth_data.csv
```

---

## Pipeline Completo

O projeto segue um **pipeline sequencial** onde cada etapa depende dos artefatos gerados pela anterior:

```
┌─────────────────────┐
│  1. sim_data.py      │  Gera dados sintéticos do motor (RPM, MAP, VE, ...)
└────────┬────────────┘
         │ data/synth_data.csv
         ▼
┌─────────────────────┐
│  2. sr_sindy.py      │  SINDy → equações diferenciais esparsas (dRPM/dt, dMAP/dt)
│                      │  PySR  → equação simbólica para VE (Programação Genética)
└────────┬────────────┘
         │ models/sindy_coefficients.npy
         │ models/ve_equation.txt
         │ models/pysr_ve_model.pkl
         ▼
┌─────────────────────┐
│  3. sim_motor.py     │  Valida as equações descobertas contra dados reais
│                      │  Gera models/equation_validation.json (status: approved/rejected)
└────────┬────────────┘
         │ models/equation_validation.json (status: "approved")
         ▼
┌──────────────────────────────────────────────────────────┐
│  4a. pinn_torchdiffeq.py      (PINN básica)              │
│  4b. research_two_pinns.py    (estudo comparativo)       │
└────────┬─────────────────────────────────────────────────┘
         │ models/pinn_torchdiffeq.pth
         │ outputs/<run>_two_pinns/
         ▼
┌─────────────────────┐
│  5. frontend/app.py  │  Dashboard interativo (Streamlit)
└─────────────────────┘
```

> **Nota:** `pinn_torchdiffeq.py` e `research_two_pinns.py` podem ser executados de forma independente como experimentos paralelos.

---

## Módulos em Detalhe

### `sim_data.py` — Geração de Dados Sintéticos

Gera 600 amostras (6 segundos a 100 Hz) de um cenário de aceleração realista:

- **MAP**: cresce exponencialmente simulando boost do turbo: `MAP(t) = 1.0 + 1.2·(1 - e^{-2t})`
- **RPM**: integra numericamente `dRPM/dt = (torque × 7162) / (2000 + 500t)`, com torque calculado a partir de MAP, VE e ignição
- **VE**: perfil senoidal `0.85 + 0.1·sin(linspace(0, 3, n))`
- Ruído gaussiano adicionado (σ = 15 rpm, σ = 0.02 bar)

**Saída:** `data/synth_data.csv` com colunas `time, RPM, MAP, tps, Ignition, Lambda, dRPM_dt_truth, dMAP_dt_truth`

---

### `sr_sindy.py` — SINDy + Regressão Simbólica (PySR)

#### SINDy (Sparse Identification of Nonlinear Dynamics)

Usa a biblioteca `pysindy` para identificar equações diferenciais esparsas na forma:

```
ẋ = Θ(x, u) · ξ
```

Onde:
- **x** = `[RPM, MAP]` — estado do sistema
- **u** = `[Ignition, tps]` — entradas de controle
- **Θ** = biblioteca de features polinomiais de grau 2 (15 features: `1, RPM, MAP, u_ign, u_tps, RPM², RPM·MAP, ...`)
- **ξ** = coeficientes esparsos encontrados por STLSQ (Sequential Threshold Least Squares)

O resultado é uma matriz `(2, 15)` de coeficientes que descrevem `dRPM/dt` e `dMAP/dt` em termos das features.

**Artefatos gerados:**
- `models/sindy_coefficients.npy` — matriz de coeficientes `(2, 15)`
- `models/sindy_features.txt` — nomes das 15 features

#### PySR — Programação Genética para VE

Usa o `PySRRegressor` para descobrir, via **algoritmo genético evolutivo**, uma expressão matemática fechada que descreve a eficiência volumétrica:

```python
model_sr = PySRRegressor(
    niterations=50,
    binary_operators=['+', '-', '*', '/'],
    unary_operators=['sin', 'exp', 'log'],
    maxsize=20,
    model_selection="best"
)
```

O algoritmo evolui uma **população de expressões simbólicas** através de operações de mutação, cruzamento e seleção, minimizando o erro entre a expressão e o target `VE = 0.85 + 0.1·sin(RPM/1000)`.

**Exemplo de equação descoberta:**
```
VE = sin(sin(log((x0 + (log(x0) + x0)) / 0.921)) + 0.533) - 0.044
```

**Artefatos gerados:**
- `models/pysr_ve_model.pkl` — modelo completo serializado
- `models/ve_layer_pytorch.pth` — modelo exportado via TorchScript JIT
- `models/ve_equation.txt` — equação como string (fallback quando JIT falha)

---

### `sim_motor.py` — Simulação e Validação do Motor

Integra o modelo SINDy descoberto usando `scipy.integrate.solve_ivp` (método Radau) e compara com os dados observados.

#### Validação das Equações

A função `validate_equations_on_data()` simula o modelo e calcula métricas de qualidade:

| Métrica | Threshold de aprovação | Resultado típico |
|---|---|---|
| RMSE RPM | < 450 rpm | ~17 rpm |
| RMSE MAP | < 0.25 bar | ~0.026 bar |
| RMSE VE | < 0.08 | ~0.001 |

O resultado é salvo em `models/equation_validation.json` com `"status": "approved"` ou `"rejected"`. **A etapa PINN só prossegue se o status for `"approved"`**, garantindo que a base física do modelo seja sólida antes de adicionar a rede neural.

#### Simulação de Upgrade

Compara dois cenários: motor stock (`boost_offset=0`) e motor com upgrade de turbo (`boost_offset=0.3`), gerando curvas de RPM e potência (CV) para análise.

---

### `pinn_torchdiffeq.py` — PINN Híbrida (SINDy + Neural ODE)

Implementa o modelo `HybridEA888ODE`: uma **Neural ODE** que combina a física descoberta pelo SINDy com uma rede residual treinável.

#### Arquitetura

```
entrada: x = [RPM, MAP]
         │
         ├──► features Θ(x) [1×15] ──► @ sindy_coeffs.T ──► base_dx [1×2]
         │                                                         │
         └──► [RPM/8000, MAP/4.0] ──► residual_net ──────────► residual_dx [1×2]
                                      (Linear-Tanh-Linear-Tanh-Linear)
                                                                   │
                                                         dx = base_dx + residual_dx
                                                         (clampado para estabilidade)
```

#### Função de Loss

```
L = L_data + λ_phys · L_phys + λ_ve · L_ve
```

- **L_data**: MSE entre trajetória predita (ODE integrada) e dados observados
- **L_phys**: MSE entre derivadas do modelo e derivadas finitas observadas
- **L_ve**: MSE entre VE prevista e VE observada (consistência com equação simbólica)

O solver ODE `dopri5` (Dormand–Prince, ordem 4/5) do `torchdiffeq` é usado para integrar o sistema, propagando gradientes através do tempo via adjoint method.

---

### `research_two_pinns.py` — Estudo Comparativo: Dois PINNs

O módulo mais complexo do repositório. Treina e compara **dois PINNs com filosofias distintas**, com pipeline de avaliação rigoroso incluindo XAI.

#### Divisão dos Dados

Split temporal (sem shuffle, preservando causalidade):
- **Train**: 70% dos pontos
- **Val**: 15% dos pontos
- **Test**: 15% dos pontos

#### PINN 1 — `PhysicsInformedODE`

Incorpora o conhecimento do SINDy **diretamente na loss** via termos de física. Ver [seção de arquitetura](#pinn-1--physicsinformedode).

#### PINN 2 — `EngineeringODE`

Usa **priors de engenharia** (parâmetros interpretáveis para torque, drag, boost, MAP recovery) em vez das equações SINDy. Os parâmetros `p_map`, `p_ign`, `p_drag`, `p_boost`, etc. são **treináveis e interpretáveis**.

#### Treinamento Avançado

- **Adam** com weight decay `1e-6`
- **ReduceLROnPlateau**: reduz LR por fator 0.5 após 35 épocas sem melhora na validação
- **Early stopping**: para após `patience=140` épocas sem melhora
- **Gradient clipping**: `max_norm=1.0` para estabilidade
- **Checkpointing**: salva o melhor estado via `best_state = copy.deepcopy(model.state_dict())`

#### XAI e Métricas Exportadas

Cada execução (`run`) gera um diretório `outputs/<timestamp>_two_pinns/` com:

| Arquivo | Conteúdo |
|---|---|
| `metrics_report.json` | RMSE e MAE de RPM/MAP em train/val/test |
| `training_manifest.json` | Configuração completa, solver ODE, XAI aplicado |
| `pinn_physics_history.csv` | Curva de loss por época (PINN física) |
| `pinn_engineering_history.csv` | Curva de loss por época (PINN engenharia) |
| `full_predictions.csv` | Observações vs predições (série temporal completa) |
| `pinn_*_mc_dropout.csv` | Amostras MC Dropout para incerteza |
| `pinn_*_nll_noise.csv` | NLL sob diferentes níveis de ruído |
| `pinn_*_feature_missing.csv` | Impacto da remoção de features |
| `pinn_*_sensitivity.json` | Sensibilidade local a perturbações nas entradas |
| `pinn_*_counterfactual.csv` | Grid de simulações boost × MAP para análise "e se" |

---

### `frontend/app.py` — Dashboard Streamlit

Interface web interativa com **5 abas**:

| Aba | Conteúdo |
|---|---|
| **Resumo** | Tabela comparativa de métricas train/val/test, gráficos RPM e MAP observado vs predito, trajetória 3D |
| **Treino** | Curvas de loss por época para ambos os PINNs, schedule de learning rate |
| **Simulador Upgrade** | Simulação em tempo real com sliders: `boost_offset`, `map_scale`, `rpm0_delta`, `ignition_scale`, `tps_scale`; gráficos de RPM, MAP e potência (CV) |
| **XAI** | Intervalos de confiança MC Dropout, NLL sob ruído, impacto de features ausentes, sensibilidade local |
| **Como foi treinada** | JSON completo do `training_manifest.json`: configuração, solver, modelos, XAI aplicado |

---

## Arquitetura dos Modelos

### SINDy — Identificação Esparsa da Dinâmica

```
dRPM/dt = ξ₀ᵀ · [1, RPM, MAP, u_ign, u_tps, RPM², RPM·MAP, RPM·u_ign, RPM·u_tps,
                   MAP², MAP·u_ign, MAP·u_tps, u_ign², u_ign·u_tps, u_tps²]

dMAP/dt = ξ₁ᵀ · [mesmas 15 features]
```

O STLSQ aplica um **threshold de esparsidade** (`1e-5`), zerando coeficientes pequenos e resultando em equações interpretáveis com poucos termos ativos.

---

### PySR / Programação Genética — Regressão Simbólica

**Algoritmo Evolutivo:**

```
Inicializar população de expressões aleatórias
    │
    ▼
Para cada geração:
    ├── Avaliar fitness (MSE com target)
    ├── Seleção (melhores expressões)
    ├── Cruzamento (combinar subárvores)
    └── Mutação (trocar operadores, constantes, variáveis)
    │
    ▼
Retornar expressão com menor complexidade e erro (Pareto-ótima)
```

**Operadores disponíveis:**
- Binários: `+`, `-`, `*`, `/`
- Unários: `sin`, `exp`, `log`
- Tamanho máximo da expressão: 20 nós

---

### PINN 1 — `PhysicsInformedODE`

```
PhysicsInformedODE
├── sindy_coeffs: buffer (2, 15)  — coeficientes SINDy (congelados)
└── residual: Sequential
    ├── Linear(2 → hidden_dim=96)
    ├── Tanh
    ├── Dropout(0.10)
    ├── Linear(hidden_dim → hidden_dim)
    ├── Tanh
    ├── Dropout(0.10)
    └── Linear(hidden_dim → 2)
```

**Loss composta:**
```
L = L_data + λ_smooth·L_smooth + λ_bounds·L_bounds + λ_dyn·L_dyn + λ_ve·L_ve
```

| Componente | Peso padrão | Descrição |
|---|---|---|
| `L_data` | 1.0 | MSE trajetória vs dados |
| `L_smooth` | 1e-4 | Penalidade de segunda diferença (suavidade) |
| `L_bounds` | 1e-3 | Penalidade por sair dos limites físicos (RPM, MAP) |
| `L_dyn` | 0.20 | MSE derivadas vs derivadas SINDy |
| `L_ve` | 0.05 | Consistência com equação simbólica da VE |

---

### PINN 2 — `EngineeringODE`

```
EngineeringODE
├── Parâmetros treináveis (priors de engenharia):
│   ├── p_map, p_ign, p_tps, p_drag, p_boost  → ganhos de RPM
│   └── m_recover, m_leak, m_boost, m_tps     → dinâmica de MAP
└── residual: Sequential
    ├── Linear(4 → hidden_dim=96)
    ├── SiLU
    ├── Dropout(0.10)
    ├── Linear(hidden_dim → hidden_dim)
    ├── SiLU
    ├── Dropout(0.10)
    └── Linear(hidden_dim → 2)
```

A dinâmica base é formulada a partir de princípios de engenharia:
```
dRPM/dt_base = 320·p_map·MAP_n + 120·p_ign·ign_n + 150·p_tps·tps_n
             - 250·p_drag·RPM_n² + 420·p_boost·boost_offset
```

Todos os parâmetros `p_*` e `m_*` passam por `softplus` para garantir positividade, tornando o modelo **interpretável por design**.

---

## XAI e Incerteza

### MC Dropout (Monte Carlo Dropout)

Durante a inferência, o dropout permanece **ativo** e o modelo é chamado `n=60` vezes. A média e desvio padrão das predições fornecem **intervalos de confiança** para RPM e MAP.

```python
# Usa train_mode=True para manter dropout ativo na inferência
p = simulate(model, seg, train_mode=True)
```

### NLL sob Ruído

Avalia a robustez do modelo injetando ruído gaussiano crescente nos dados de entrada e calculando a **Negative Log-Likelihood** (NLL) para RPM e MAP.

### Feature Missing Impact

Remove sistematicamente cada feature de controle (`Ignition`, `tps`) e mede o aumento no RMSE, quantificando a **importância relativa** de cada entrada.

### Sensibilidade Local

Perturba cada dimensão do estado inicial `x0` e das entradas de controle em ±5%, medindo a variação na trajetória predita — equivalente ao **gradiente de sensibilidade de primeira ordem**.

### Análise Contrafactual

Varre um grid de `boost_offset × map_scale_init` e simula o motor para cada combinação, gerando um **mapa de calor** do CV (potência em cavalos) para apoiar decisões de upgrade.

---

## Artefatos Salvos

| Arquivo | Gerado por | Descrição |
|---|---|---|
| `data/synth_data.csv` | `sim_data.py` | Dados sintéticos do motor (600 amostras) |
| `models/sindy_coefficients.npy` | `sr_sindy.py` | Matriz de coeficientes SINDy `(2, 15)` |
| `models/sindy_features.txt` | `sr_sindy.py` | Nomes das 15 features polinomiais |
| `models/pysr_ve_model.pkl` | `sr_sindy.py` | Regressor PySR completo |
| `models/ve_layer_pytorch.pth` | `sr_sindy.py` | Modelo VE como TorchScript |
| `models/ve_equation.txt` | `sr_sindy.py` | Equação VE como string |
| `models/equation_validation.json` | `sim_motor.py` | Status de aprovação + métricas |
| `models/pinn_torchdiffeq.pth` | `pinn_torchdiffeq.py` | Pesos da PINN básica |
| `outputs/<run>_two_pinns/` | `research_two_pinns.py` | Todos os artefatos de XAI e métricas |

---

## Instalação e Dependências

### Backend (pipeline de treinamento)

```bash
pip install numpy pandas scipy matplotlib torch torchdiffeq pysindy pysr
```

> **Nota:** O `PySR` requer Julia instalada. Consulte a [documentação oficial do PySR](https://astroautomata.com/PySR/) para instruções de instalação.

### Frontend (dashboard)

```bash
pip install -r frontend/requirements.txt
# streamlit>=1.33.0, plotly>=5.20.0, pandas>=2.2.0, numpy>=1.26.0
```

---

## Como Executar

Execute os scripts **na ordem do pipeline**:

```bash
# 1. Gerar dados sintéticos
python pinn_sr/sim_data.py

# 2. Treinar SINDy e PySR (Programação Genética)
python pinn_sr/sr_sindy.py

# 3. Validar equações e simular cenários
python pinn_sr/sim_motor.py

# 4a. Treinar PINN básica
python pinn_sr/pinn_torchdiffeq.py

# 4b. (Alternativa) Treinar os dois PINNs com XAI completo
python pinn_sr/research_two_pinns.py

# 5. Iniciar o dashboard
streamlit run frontend/app.py
```

### Opções da PINN básica (`pinn_torchdiffeq.py`)

```bash
python pinn_sr/pinn_torchdiffeq.py \
  --epochs 1200 \
  --lr 1e-3 \
  --lambda-phys 0.1 \
  --lambda-ve 0.05 \
  --hidden-dim 64 \
  --method dopri5 \
  --device cpu
```

---

## Resultados de Validação

Métricas obtidas na execução com dados sintéticos (`models/equation_validation.json`):

| Métrica | Resultado | Threshold |
|---|---|---|
| RMSE RPM | **17.1 rpm** | < 450 rpm ✅ |
| RMSE MAP | **0.026 bar** | < 0.25 bar ✅ |
| RMSE VE | **0.0011** | < 0.08 ✅ |
| Max abs RPM | 52.3 rpm | — |
| Max abs MAP | 0.080 bar | — |

**Status:** `approved` — pipeline PINN liberado para execução.

---

## Conceitos-Chave para Estudo

| Conceito | Onde é aplicado |
|---|---|
| **Programação Genética** | `PySR` em `sr_sindy.py` — evolução de expressões simbólicas |
| **Regressão Simbólica** | Descoberta da equação de VE com `PySR` |
| **SINDy** | Identificação esparsa de EDOs em `sr_sindy.py` |
| **Neural ODE** | Integração de rede neural como sistema dinâmico via `torchdiffeq` |
| **PINN** | Restrições físicas (SINDy, VE) na função de loss |
| **Physics-constrained ML** | Combinação de priors físicos com dados via `EngineeringODE` |
| **MC Dropout** | Estimativa de incerteza epistêmica em `research_two_pinns.py` |
| **XAI** | Sensibilidade, impacto de features, análise contrafactual |

---

## Licença

Este projeto está licenciado sob os termos do arquivo [LICENSE](LICENSE).
