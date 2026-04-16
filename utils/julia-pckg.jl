using Pkg

Pkg.activate("sciml-julia")

packages = [
    # ----------------------
    # Core SciML
    # ----------------------
    "DifferentialEquations",
    "SciMLSensitivity",
    "ModelingToolkit",

    # ----------------------
    # Neural + SciML
    # ----------------------
    "Lux",
    "Flux",
    "DiffEqFlux",

    # Optimization
    "Optimization",
    "OptimizationOptimJL",
    "OptimizationFlux",

    # ----------------------
    # Data structures
    # ----------------------
    "ComponentArrays",
    "StaticArrays",

    # ----------------------
    # Symbolics
    # ----------------------
    "Symbolics",

    # ----------------------
    # Plotting
    # ----------------------
    "Plots", # trocar por Makie se quiser algo mais moderno

    # ----------------------
    # Solvers específicos (opcional, mas explícito)
    # ----------------------
    "OrdinaryDiffEq",
    "StochasticDiffEq"
]

Pkg.add(packages)

println("Ambiente SciML Julia pronto.")