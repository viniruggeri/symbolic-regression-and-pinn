using Pkg

Pkg.update()

packages = [
    # Core SciML
    "DifferentialEquations",
    "SciMLSensitivity",
    "ModelingToolkit",

    # Neural + SciML
    "Lux",
    "Flux",
    "Optimization",
    "OptimizationOptimJL",
    "OptimizationFlux",

    # Data structures
    "ComponentArrays",
    "StaticArrays",

    # Symbolics
    "Symbolics",

    # Plotting
    "Plots",

    # Extras úteis
    "DiffEqFlux",
    "OrdinaryDiffEq",
    "StochasticDiffEq"
]

Pkg.add(packages)

println("Ambiente SciML Julia FULL pronto.")