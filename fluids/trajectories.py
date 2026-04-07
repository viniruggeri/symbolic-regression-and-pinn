import matplotlib.pyplot as plt
import numpy as np
from generator import X
from jax import random

x = X


def plot_trajectories(X, n_plot=20):
    """
    X: (n_paths, n_steps)
    """
 
    X_np = np.array(x)  # converte de JAX → NumPy
 
    n_paths, n_steps = X_np.shape
    t = np.arange(n_steps)
 
    plt.figure(figsize=(10, 5))
 
    for i in range(min(n_plot, n_paths)):
        plt.plot(t, X_np[i], alpha=0.6)
 
    plt.title("Ornstein-Uhlenbeck Trajectories")
    plt.xlabel("Time step")
    plt.ylabel("x(t)")
    plt.grid(True)
    plt.show()
 
def plot_distribution(X):
    X_np = np.array(x)
    final_values = X_np[:, -1]
 
    plt.figure(figsize=(6, 4))
    plt.hist(final_values, bins=50, density=True)
    plt.title("Final Distribution")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()
 
 
def plot_autocorrelation(X, lag=100):
    X_np = np.array(X)
 
    x = X_np[0]  # pega uma trajetória
    acf = [np.corrcoef(x[:-l], x[l:])[0,1] for l in range(1, lag)]
 
    plt.figure(figsize=(6,4))
    plt.plot(acf)
    plt.title("Autocorrelation")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.grid(True)
    plt.show()


plot_trajectories(x)
plot_distribution(x)
plot_autocorrelation(x)