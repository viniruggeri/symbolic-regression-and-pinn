# ----------------------
# Pacotes
# ----------------------

local_packages <- c(
    "IRkernel",
    "deSolve",
    "pracma",
    "reticulate",
    "tidyverse",
    "ggplot2"
)

install_if_missing <- function(p) {
    if (!requireNamespace(p, quietly = TRUE)) {
        cat(paste("\nInstalando pacote:", p, "\n"))
        install.packages(p, repos = "https://cran.r-project.org")
    } else {
        cat(paste("\nPacote já instalado:", p, "\n"))
    }
}

invisible(lapply(local_packages, install_if_missing))

# ----------------------
# Integração com Python 
# ----------------------

cat("\nConfigurando reticulate...\n")
reticulate::use_condaenv("dynamical-sys", required = TRUE)

# ----------------------
# Jupyter Kernel
# ----------------------

cat("\nRegistrando IRkernel no Jupyter...\n")
IRkernel::installspec(user = FALSE)

cat("Ambiente R para SciML pronto!\n")