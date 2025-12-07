# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax",
# ]
# ///

import jax
import jax.numpy as jnp
import runge_kutta


def solve_hr_classic(T, dt, v0):
    num_iterations = int(T / dt)
    states = runge_kutta.solve_rk4_(
        hr_classic,
        v0,
        dt,
        num_iterations,
    )
    return states


def hr_classic(
    state,
    a=1.0,
    b=3.0,
    c=1.0,
    d=5.0,
    r=0.005,
    I=3.25,
    s=4.0,
    x0=-1.6,
):
    x, y, z = state

    F_x = y - a * x**3 + b * x**2 - z + I
    y_dot = c - d * x**2 - y
    z_dot = r * (s * (x - x0) - z)

    x_dot = F_x

    return jnp.array([x_dot, y_dot, z_dot])


def initialize_from_chaos(
    N: int,
    T: int,
    dt: float,
    key: jax.random.PRNGKey,
):
    v0 = jnp.array([0.1, 0.0, 0.0])
    traj = solve_hr_classic(T, dt, v0)
    burn_in = 50_000
    attractor_traj = traj[burn_in:]

    T_eff = attractor_traj.shape[0]
    idx = jax.random.randint(key, shape=(N,), minval=0, maxval=T_eff)
    initial_states = attractor_traj[idx]  # (N, 3)

    return initial_states


def attractor_trajectory(T: int = 2000, dt: float = 0.01):
    v0 = jnp.array([0.1, 0.0, 0.0])
    traj = solve_hr_classic(T, dt, v0)
    burn_in = 50_000
    attractor_traj = traj[burn_in:]

    return attractor_traj


if __name__ == "__main__":
    key = jax.random.PRNGKey(1912)
    N = 200
    T = 2000
    dt = 0.01
    initial_states = initialize_from_chaos(N, T, dt, key)
    print(initial_states.shape)
