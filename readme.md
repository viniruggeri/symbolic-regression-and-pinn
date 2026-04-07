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

1. Criar ambiente e instalar dependencias com Poetry:

```bash
poetry install
```

2. Ativar shell do ambiente:

```bash
poetry shell
```

3. Rodar notebooks no VS Code/Jupyter com o kernel do ambiente.

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