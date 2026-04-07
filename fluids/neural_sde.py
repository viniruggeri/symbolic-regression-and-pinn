import jax
import jax.numpy as jnp
import equinox as eqx
import optax
 
 
class SDE(eqx.Module):
    drift: eqx.nn.MLP
    diffusion: eqx.nn.MLP
 
    def __init__(self, key):
        k1, k2 = jax.random.split(key)
 
        # entrada: [x, t]
        self.drift = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            width_size=64,
            depth=2,
            key=k1
        )
 
        self.diffusion = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            width_size=64,
            depth=2,
            key=k2
        )
 
    def f(self, x, t):
        x = jnp.asarray(x).reshape(-1, 1)
        t = jnp.asarray(t).reshape(-1, 1)
        xt = jnp.concatenate([x, t], axis=-1)  # (N, 2)
        return jax.vmap(self.drift)(xt)


    def g(self, x, t):
        x = jnp.asarray(x).reshape(-1, 1)
        t = jnp.asarray(t).reshape(-1, 1)
        xt = jnp.concatenate([x, t], axis=-1)
        return jax.nn.softplus(jax.vmap(self.diffusion)(xt)) + 1e-4
 
 
 
def loss_fn(model, x, t, dt):
    x_t = x[:, :-1]
    dx = x[:, 1:] - x[:, :-1]

    t = t[:-1, None]                          # (T,1)
    t = jnp.broadcast_to(t, x_t.shape)        # (B,T,1)

    x_flat = x_t.reshape(-1, 1)
    t_flat = t.reshape(-1, 1)
    dx_flat = dx.reshape(-1, 1)

    f = model.f(x_flat, t_flat)
    g = model.g(x_flat, t_flat)

    mean = f * dt
    var = jnp.clip((g ** 2) * dt, 1e-5, 1.0)

    log_prob = -0.5 * (((dx_flat - mean) ** 2) / var + jnp.log(var))

    return -jnp.mean(log_prob)
 

@eqx.filter_jit
def train_step(model, opt_state, x, t, dt, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, t, dt)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
 
 

 
def simulate(model, key, x0, t, dt):
    def step(x, inputs):
        key, t_scalar = inputs

        t_input = jnp.full_like(x, t_scalar)

        dW = jax.random.normal(key, shape=x.shape) * jnp.sqrt(dt)

        drift = model.f(x, t_input)
        diffusion = model.g(x, t_input)

        x_next = x + drift * dt + diffusion * dW
        return x_next, x_next

    keys = jax.random.split(key, len(t))
    _, traj = jax.lax.scan(step, x0, (keys, t))
    return traj

 
def train(X, steps=2000, lr=1e-3):
    key = jax.random.PRNGKey(0)
    model = SDE(key)
 
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
 
    dt = 0.01
    t = jnp.arange(X.shape[1]) * dt
 
    for step in range(steps):
        model, opt_state, loss = train_step(model, opt_state, X, t, dt, optimizer)
 
        if step % 100 == 0:
            print(f"step {step} | loss {loss:.4f}")
 
    return model
 

 
if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
 
    n_paths = 128
    n_steps = 100
 
    noise = jax.random.normal(key, (n_paths, n_steps, 1)) * 0.1
    X = jnp.cumsum(noise, axis=1)
 
    model = train(X)
 
    t = jnp.arange(n_steps) * 0.01
    x0 = jnp.zeros((1, 1))
 
    traj = simulate(model, key, x0, t, 0.01)
 
    print("Simulated trajectory shape:", traj.shape)