# Applied Dynamical Systems (SciML Studies)

Repositorio de estudos em dinamicas, sistemas nao lineares e Scientific Machine Learning (SciML), com foco em modelagem orientada por fisica e dinamica latente para series temporais e campos fluidos.

## Objetivos

- Explorar modelos classicos de sistemas dinamicos e EDPs.
- Treinar modelos baseados em redes neurais com restricoes fisicas (physics-informed).
- Estudar SDEs neurais e representacoes latentes para dinamicas estocasticas.
- Conectar teorias de sistemas dinamicos com arquiteturas modernas (PINNs, SINDy, neural operators, latent SDEs).

## Estrutura Do Projeto

- `engine/`
- `engine/pinn_sr/`: codigos de PINN e symbolic regression.
- `engine/models/`: pesos e artefatos treinados.
- `fluids/`
- `fluids/generator.py`: gerador estocastico (processos tipo OU em JAX).
- `fluids/neural_sde.py`: baseline de neural SDE para trajetorias 1D.
- `fluids/latent_sde_sciml_score_bridge.ipynb`: latent SDE 1D com regularizacao fisica + score bridge.
- `fluids/latent_sde_burgers2d_cnn.ipynb`: campo Burgers 2D com encoder/decoder CNN + latent SDE.
- `fluids/latent_sde_navier_stokes2d_cnn.ipynb`: campo Navier-Stokes 2D incompressivel com latent SDE.
- `pyproject.toml`: dependencias do projeto.

## Stack

- Python 3.12
- JAX / Equinox / Optax
- PyTorch / TorchSDE
- DeepXDE
- NumPy / SciPy / Matplotlib

## Configuracao Rapida

### Usando Mamba (recomendado, mais rápido)

```bash
mamba env create -f environment.yml
mamba activate dynamical-sys
```

### Usando Conda (se mamba não estiver instalado)

```bash
conda env create -f environment.yml
conda activate dynamical-sys
```

**Se o solver travar**, use o ambiente minimalista:

```bash
conda env create -f environment-minimal.yml
conda activate dynamical-sys-minimal
```

### Atualizar dependências

```bash
# Com mamba
mamba env update -f environment.yml --prune

# Com conda
conda env update -f environment.yml --prune
```

### Uso no VS Code / Jupyter

Selecione o kernel `dynamical-sys` nos notebooks Python (.ipynb).

Para o frontend do Streamlit, use o Python desse mesmo ambiente e instale apenas as dependências de `engine/frontend/requirements.txt` se necessário.

## Setup de R e Julia (SciML Avançado)

Como o conda não lida bem nativamente com R e Julia atualizados no Windows, recomenda-se instalá-los nos seus canais oficiais e injetar os kernels no ambiente.

### 1. Preparando Julia (Equações Diferenciais e SciML)
- Baixe e instale a versão mais recente em [julialang.org](https://julialang.org/downloads/) e adicione ao PATH.
- Abra o terminal do ambiente e execute o script de instalação (ele configura o kernel `IJulia` e o pacote `DifferentialEquations.jl`):
```bash
julia utils/julia-pckg.jl
```

### 2. Preparando R (Análise Estatística e deSolve)
- Baixe e instale o R base em [cran.r-project.org](https://cran.r-project.org/bin/windows/base/) e adicione ao PATH.
- Abra o terminal (como Admin, se der erro de permissão no jupyter config) e execute o script:
```bash
Rscript utils/r-pckg.R
```

Ao religar o VS Code, os novos kernels de R e Julia estarão perfeitamente integrados com seu ambiente, sem poluir as dependências Python originais!

## Trilhas De Estudo Sugeridas

1. `fluids/neural_sde.py`
2. `fluids/latent_sde_sciml_score_bridge.ipynb`
3. `fluids/latent_sde_burgers2d_cnn.ipynb`
4. `fluids/latent_sde_navier_stokes2d_cnn.ipynb`

Essa ordem vai de dinamicas 1D para campos 2D, incluindo mais estrutura fisica a cada etapa.

## Ideias Para Proximos Experimentos

- Substituir encoder/decoder CNN por FNO (Fourier Neural Operator).
- Incluir treinamento multi-resolucao para generalizacao espacial.
- Adicionar analise de atratores, estabilidade e bifurcacoes no espaco latente.
- Acoplar SINDy no latente para extrair leis dinamicas interpretaveis.

## Nota

Este repositorio e voltado para estudo e experimentacao. Resultados podem variar conforme hiperparametros, resolucao espacial e custo computacional disponivel.