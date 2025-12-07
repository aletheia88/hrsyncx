# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax",
#     "tqdm",
# ]
# ///

from pathlib import Path
from tqdm import tqdm
import hindmarsh_rose as hr
import initialize as init
import jax
import jax.numpy as jnp
import json


def _to_serializable(obj):
    """Recursively convert JAX/NumPy types to plain Python types."""
    # JAX / NumPy arrays -> list or scalar
    if isinstance(obj, (jnp.ndarray,)):
        if obj.shape == ():  # scalar
            return float(obj)
        return obj.tolist()

    # JAX scalar types
    if isinstance(obj, (jnp.floating, jnp.bfloat16)):
        return float(obj)
    if isinstance(obj, jnp.integer):
        return int(obj)

    # Built-in container types
    if isinstance(obj, dict):
        # ensure keys are strings
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]

    # Everything else (plain float/int/str/bool/None) is fine
    return obj


def compute_fig1(
    N: int = 200,
    T: int = 5000,
    dt: float = 0.01,
    save_path: str = "fig1-50N.json",
):
    """
    For chosen values of α, plot synchronization error from fixing p_sw =
    0.1, k_1 = 3, ε = 0.5 while varying p_r.
    """
    alphas = [3.1, 3.3, 3.5, 3.7, 3.9]
    p_rs = jnp.linspace(1e-6, 1.0, 31)
    p_sw = 0.1
    k_sw = 3
    epsilon = 0.5
    num_iterations = int(T / dt)

    results = {}

    key = jax.random.PRNGKey(1976)
    key0, subkey = jax.random.split(key)
    # initialize network state at t=0
    # v0 = init.initialize_from_chaos(N, T, dt, key)
    v0 = jax.random.normal(subkey, (N, 3))

    log_dict = {
        "N": N,
        "T": T,
        "dt": dt,
        "alphas": alphas,
        "p_rs": [float(p) for p in p_rs],
        "p_sw": p_sw,
        "k_sw": k_sw,
        "epsilon": epsilon,
        "sync_error_results": results,
    }

    for alpha in alphas:
        alpha_key = str(alpha)
        log_dict["sync_error_results"][alpha_key] = []
        for p_r in tqdm(p_rs):
            A0 = hr.initialize_adjacency_matrix(N, k_sw)
            traj = hr.solve_hindmarsh_rose(
                v0,
                A0,
                key0,
                epsilon,
                alpha,
                p_r,
                p_sw,
                dt,
                num_iterations,
            )
            sync_error = hr.synchrony_error(traj)
            log_dict["sync_error_results"][alpha_key].append(float(sync_error))

        serializable_log = _to_serializable(log_dict)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(serializable_log, f, indent=4)

    return results


def compute_fig2(
    N: int = 200,
    T: int = 1000,
    dt: float = 0.01,
    save_path: str = "fig2-200N.json",
):
    epsilons = [0.15, 0.17, 0.19, 0.21, 0.23]
    p_rs = jnp.linspace(1e-6, 1.0, 31)
    p_sw = 0.1
    k_sw = 3
    alpha = 2.5
    num_iterations = int(T / dt)

    results = {}
    key = jax.random.PRNGKey(1976)
    key0, subkey = jax.random.split(key)
    # initialize network state at t=0
    v0 = init.initialize_from_chaos(N, T, dt, key)

    log_dict = {
        "N": N,
        "T": T,
        "dt": dt,
        "alpha": alpha,
        "p_rs": [float(p) for p in p_rs],
        "p_sw": p_sw,
        "k_sw": k_sw,
        "epsilon": epsilons,
        "sync_error_results": results,
    }

    for epsilon in epsilons:
        epsilon_key = str(epsilon)
        log_dict["sync_error_results"][epsilon_key] = []
        for p_r in tqdm(p_rs):
            A0 = hr.initialize_adjacency_matrix(N, k_sw)
            traj = hr.solve_hindmarsh_rose(
                v0,
                A0,
                key0,
                epsilon,
                alpha,
                p_r,
                p_sw,
                dt,
                num_iterations,
            )
            sync_error = hr.synchrony_error(traj)
            log_dict["sync_error_results"][epsilon_key].append(float(sync_error))

        serializable_log = _to_serializable(log_dict)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(serializable_log, f, indent=4)

    return results


if __name__ == "__main__":
    N = 200
    save_path = f"results/fig2-{N}N_init-from-chaos.json"
    T = 1000
    dt = 0.01
    compute_fig2(N, T, dt, save_path=save_path)
