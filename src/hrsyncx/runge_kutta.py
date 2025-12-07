from functools import partial
import hindmarsh_rose as hr
import jax


@jax.jit
def solve_rk4(network_state, h, epsilon, B_t):
    def f(y):
        return hr.hindmarsh_rose_network(y, epsilon, B_t)

    k1 = f(network_state)
    k2 = f(network_state + h * k1 / 2)
    k3 = f(network_state + h * k2 / 2)
    k4 = f(network_state + h * k3)
    next_state = network_state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return next_state


@partial(jax.jit, static_argnames=("f", "num_iterations"))
def solve_rk4_(f, y0, h, num_iterations):
    def step(current_y, _):
        k1 = f(current_y)
        k2 = f(current_y + h * k1 / 2)
        k3 = f(current_y + h * k2 / 2)
        k4 = f(current_y + h * k3)
        next_y = current_y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return next_y, next_y

    _, y = jax.lax.scan(step, y0, length=num_iterations)
    return y
