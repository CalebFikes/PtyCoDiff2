import jax
import jax.numpy as jnp
from diffusion.model import create_complexUnet


def test_complexunet_shape_and_dtype():
    key = jax.random.PRNGKey(2)
    params, apply_fn = create_complexUnet(key, input_shape=(32, 32, 1, 1))

    xr = jax.random.normal(key, (32, 32, 1))
    xi = jax.random.normal(key, (32, 32, 1))
    x = (xr + 1j * xi).astype(jnp.complex64)

    out = apply_fn(params, x, 0.25)
    assert out.shape == x.shape
    assert jnp.iscomplexobj(out)
