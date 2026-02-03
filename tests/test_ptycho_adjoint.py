import jax
import jax.numpy as jnp
from ptycho.ops import O_xi, adjoint_scatter


def test_adjoint_consistency():
    key = jax.random.PRNGKey(1)
    H, W, C = 20, 20, 1
    theta = (jax.random.normal(key, (H, W, C)) + 1j * jax.random.normal(key, (H, W, C))).astype(jnp.complex64)

    xi = jnp.array([5.3, 6.7], dtype=jnp.float32)
    ph, pw = 7, 7

    patch = O_xi(theta, xi, (ph, pw))
    back = (jax.random.normal(key, (ph, pw, C)) + 1j * jax.random.normal(key, (ph, pw, C))).astype(jnp.complex64)

    left = jnp.vdot(patch, back)
    adj = adjoint_scatter((H, W, C), xi, back, (ph, pw))
    right = jnp.vdot(theta, adj)

    assert jnp.allclose(left, right, atol=1e-4)
