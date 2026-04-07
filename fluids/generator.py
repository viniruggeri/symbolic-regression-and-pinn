import jax
import jax.numpy as jnp
from jax import random, lax
from typing import Optional
 
def generate_ou_process_jax(
    key: Optional, # type: ignore
    n_paths=1024,
    n_steps=1000,
    dt=0.01,
    theta=0.7,
    mu=0.0,
    sigma=0.3,
    x0=0.0
):
    """
    Ornstein-Uhlenbeck SDE:
        dx = theta*(mu - x)*dt + sigma*dW
 
    Returns:
        X: (n_paths, n_steps)
    """
 
    # ruído browniano
    key, subkey = random.split(key)
    dW = jnp.sqrt(dt) * random.normal(subkey, (n_paths, n_steps - 1))
 
    def step(x_prev, dw):
        x_next = x_prev + theta * (mu - x_prev) * dt + sigma * dw
        return x_next, x_next
 
    # estado inicial (batch)
    x0_batch = jnp.ones((n_paths,)) * x0
 
    # scan no tempo (sem loop Python)
    _, X = lax.scan(step, x0_batch, dW.T)
 
    # reorganiza shape: (time, batch) → (batch, time)
    X = X.T
 
    # adiciona condição inicial
    X = jnp.concatenate([x0_batch[:, None], X], axis=1)
 
    return X


generate_ou_process_jax_jit = jax.jit(generate_ou_process_jax)

key = random.PRNGKey(0)

X = generate_ou_process_jax_jit(key)