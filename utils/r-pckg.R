# Instala pacotes do R para modelagem matemática, ML e integração com Jupyter

local_packages <- c(
    "IRkernel",     # Para usar o kernel R no VS Code / Jupyter
    "deSolve",      # Solvers essenciais de equações diferenciais
    "pracma",       # Matemática aplicada (álgebra linear, cálculo, equações diferenciais)
    "reticulate",   # Integração direta com Python (reaproveitar numpy/pandas/pytorch)
    "tidyverse",    # Suite padrão de manipulação e organização de dados
    "ggplot2"       # Para plotagens científicas
)

install_if_missing <- function(p) {
    if (!requireNamespace(p, quietly = TRUE)) {
        cat(paste("\nInstalando pacote:", p, "\n"))
        install.packages(p, repos = "http://cran.us.r-project.org")
    } else {
        cat(paste("\nPacote já instalado:", p, "\n"))
    }
}

invisible(lapply(local_packages, install_if_missing))

# Instala e registra o kernel do R para uso interativo com os notebooks do workspace
cat("\nRegistrando IRkernel no Jupyter...\n")
IRkernel::installspec(user = FALSE)
cat("Ambiente R para SciML pronto!\n")
