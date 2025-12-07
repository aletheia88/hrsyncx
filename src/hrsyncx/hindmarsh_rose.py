# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax",
#     "matplotlib",
# ]
# ///

from functools import partial
import jax
import jax.numpy as jnp
import runge_kutta as rk


@partial(jax.jit, static_argnames=("num_iterations",))
def solve_hindmarsh_rose(
    v0,
    A0,
    key0,
    epsilon,
    alpha,
    p_r,
    p_sw,
    dt,
    num_iterations,
):
    init_carry = (v0, A0, key0, p_r, p_sw, alpha, epsilon, dt)
    auxillaries = jax.lax.scan(step, init_carry, xs=None, length=num_iterations)
    # states: (num_iterations, N, 3)
    states = auxillaries[-1]
    return states


@jax.jit
def step(carry, _):
    network_state, A_t, key, p_r, p_sw, alpha, epsilon, dt = carry

    # update adjacency matrix and key
    A_t, key = update_adjacency_matrix(A_t, key, p_r, p_sw)

    # compute k_max and distance
    # network_distance: (N, N) shortest path lengths from node i to j
    k_max, network_distance = network_diameter(A_t)
    # find nonzero entries: paths of length >= 1 and <= k_max
    nonzeros = (network_distance >= 1.0) & (
        network_distance <= k_max.astype(jnp.float32)
    )
    # compute B_t
    B_t = jnp.where(nonzeros, 1.0 / (network_distance**alpha), 0.0)

    # run RK4 with this updated B_t
    next_state = rk.solve_rk4(network_state, dt, epsilon, B_t)

    new_carry = (next_state, A_t, key, p_r, p_sw, alpha, epsilon, dt)
    return new_carry, next_state


def hindmarsh_rose_network(network_state, epsilon, B_t):
    x = network_state[:, 0]  # x: (N, )
    y = network_state[:, 1]  # y: (N, )
    z = network_state[:, 2]  # z: (N, )

    # HR model parameters
    a = 1
    b = 3
    c = 1
    d = 5
    r = 0.005
    I = 3.25
    s = 4
    x0 = -1.6

    degree = B_t.sum(axis=1)  # degree: (N, )
    coupling = B_t @ x - degree * x

    # F(x) := isolated node dynamics
    F_x = y - a * x**3 + b * x**2 - z + I

    x_dot = F_x + epsilon * coupling
    y_dot = c - d * x**2 - y
    z_dot = r * (s * (x - x0) - z)

    # return shape (N, 3)
    return jnp.stack([x_dot, y_dot, z_dot], axis=1)


@jax.jit
def network_diameter(A: jnp.ndarray) -> jnp.int32:
    N = A.shape[0]
    inf = 1e6
    dist = jnp.where(A > 0, 1.0, inf)
    dist = dist.at[jnp.arange(N), jnp.arange(N)].set(0.0)

    def update_dist(k, dist):
        dist_ik = dist[:, k][:, None]
        dist_kj = dist[k, :][None, :]
        # jax broadcasting:
        # candidate[i, j] = dist_ik[i, 0] + dist_kj[0, j]
        #                 = dist[i, k] + dist[k, j]
        candidate = dist_ik + dist_kj
        # equivalent to:
        # dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
        new_dist = jnp.minimum(dist, candidate)
        return new_dist

    dist = jax.lax.fori_loop(0, N, update_dist, dist)
    diameter = jnp.max(dist)
    return diameter.astype(jnp.int32), dist


def initialize_adjacency_matrix(N, k_sw):
    assert 1 <= k_sw < N // 2
    # 1) ring lattice
    A = jnp.zeros((N, N), dtype=int)
    for i in range(N):
        for d in range(1, k_sw + 1):
            j = (i + d) % N  # neighbor on the "right"
            A = A.at[i, j].set(1)
            A = A.at[j, i].set(1)  # undirected
    return A


@jax.jit
def update_adjacency_matrix(A_t, key, p_r, p_sw):
    key, k_rewire = jax.random.split(key)
    do_rewire = jax.random.bernoulli(k_rewire, p=p_r)

    def update(A):
        A_next, key_next = rewire_network(A, key, p_sw)
        return A_next, key_next

    def keep(A):
        return A, key

    return jax.lax.cond(do_rewire, update, keep, A_t)


@jax.jit
def rewire_network(A_t: jnp.ndarray, key: jax.Array, p_sw: float):
    """Rewire edges in the current adjacency matrix A_t using a Watts-Strogatz
    rewiring rule."""

    N = A_t.shape[0]
    i_idx, j_idx = jnp.triu_indices(N, k=1)
    M = i_idx.shape[0]

    key, key_flags = jax.random.split(key)
    rewire_flags = jax.random.bernoulli(key_flags, p=p_sw, shape=(M,))

    def body(e, carry):
        A_current, key_current = carry

        i = i_idx[e]
        j = j_idx[e]
        has_edge = A_current[i, j] == 1
        do_rewire = jnp.logical_and(has_edge, rewire_flags[e])

        def rewire_edge(carry_inner):
            A_inner, key_inner = carry_inner

            # remove old edge
            A_inner = A_inner.at[i, j].set(0).at[j, i].set(0)

            # jax while loop helpers
            def init_state():
                key0 = key_inner
                # dummy candidate to be overwritten
                candidate0 = jnp.int32(0)
                ok0 = jnp.bool_(False)
                return (key0, candidate0, ok0)

            def cond_fn(state):
                key_s, candidate_s, ok_s = state
                return jnp.logical_not(ok_s)

            def body_fn(state):
                key_s, _, _ = state
                key_s, subkey = jax.random.split(key_s)
                candidate_s = jax.random.randint(
                    subkey,
                    shape=(),
                    minval=0,
                    maxval=N,
                )
                is_self = candidate_s == i
                is_neighbor = A_inner[i, candidate_s] == 1
                ok_s = jnp.logical_not(jnp.logical_or(is_self, is_neighbor))
                return (key_s, candidate_s, ok_s)

            key_new, new_j, _ = jax.lax.while_loop(cond_fn, body_fn, init_state())
            # add new edge
            A_inner = A_inner.at[i, new_j].set(1).at[new_j, i].set(1)

            return (A_inner, key_new)

        def no_rewire(carry_inner):
            return carry_inner

        return jax.lax.cond(
            do_rewire,
            rewire_edge,
            no_rewire,
            (A_current, key_current),
        )

    A_next, key_next = jax.lax.fori_loop(0, M, body, (A_t, key))
    return A_next, key_next


@jax.jit
def synchrony_error(network_states):
    x0 = network_states[:, 0, 0]
    y0 = network_states[:, 0, 1]
    z0 = network_states[:, 0, 2]

    def single_node_error(x, y, z):
        return jnp.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)

    # network_states: (int(T/dt), N, 3)
    num_steps, N, _ = network_states.shape
    x = network_states[:, :, 0]
    y = network_states[:, :, 1]
    z = network_states[:, :, 2]

    net_error = jnp.mean(
        (1 / (N - 1)) * jnp.sum(jax.vmap(single_node_error, in_axes=1)(x, y, z), axis=0)
    )
    return net_error


if __name__ == "__main__":
    epsilon = 0.23
    alpha = 2.5
    N = 200
    T = 10
    dt = 0.01
    k_sw = 3
    p_r = 1.0
    p_sw = 0.1

    key = jax.random.PRNGKey(2001)
    key0, subkey = jax.random.split(key)
    # network state at t=0
    v0 = jax.random.normal(subkey, (N, 3))
    A0 = initialize_adjacency_matrix(N, k_sw)
    num_iterations = int(T / dt)

    traj = solve_hindmarsh_rose(
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
    print(traj.shape)
    sync_error = synchrony_error(traj)
    print(f"synchronization error: {sync_error}")

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    neuron_id = 24
    x = 0
    y = 1
    z = 2
    ax.plot(traj[:, neuron_id, x], label="x")
    ax.plot(traj[:, neuron_id, y], label="y")
    ax.plot(traj[:, neuron_id, z], label="z")
    ax.legend()
    plt.show()
