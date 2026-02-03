import pickle
import numpy as np
import jax
import jax.numpy as jnp

from ptycho.forward import forward_batch
from ptycho.sampler import generate_ptycho_posterior_samples_ula


def run_smoke():
    key = jax.random.PRNGKey(0)
    H = W = 16
    C = 1
    ph = pw = 8
    # true object
    key, k1, k2 = jax.random.split(key, 3)
    true_real = jax.random.normal(k1, (H, W, C))
    true_imag = jax.random.normal(k2, (H, W, C))
    theta_true = (true_real + 1j * true_imag).astype(jnp.complex64)

    # probe
    probe = jnp.ones((ph, pw, C), dtype=jnp.complex64)

    # scan positions (J)
    J = 4
    ys = jnp.linspace(0, H - ph, J)
    xs = jnp.linspace(0, W - pw, J)
    xis = jnp.stack([ys, xs], axis=1)

    # forward to get intensities
    a_js = forward_batch(theta_true, xis, probe, (ph, pw))
    lam = jnp.real(a_js * jnp.conj(a_js))

    R = 2.0
    rng = np.random.RandomState(1)
    f = rng.poisson(R * np.array(lam)).astype(np.float32)

    # run ULA short
    P = 2
    samples = generate_ptycho_posterior_samples_ula(jax.random.PRNGKey(1), P,
                                                     init=None,
                                                     prior_score_apply=None,
                                                     prior_params=None,
                                                     xis=xis,
                                                     f_measurements=jnp.array(f),
                                                     probe=probe,
                                                     patch_shape=(ph, pw),
                                                     n_steps=5,
                                                     step_size=1e-5,
                                                     measurement_weight=0.1,
                                                     R=R)

    print('samples shape:', samples.shape)
    print('sample dtype:', samples.dtype)
    # show simple stats
    s = np.array(samples)
    print('samples real min/max:', s.real.min(), s.real.max())
    print('samples imag min/max:', s.imag.min(), s.imag.max())


if __name__ == '__main__':
    run_smoke()
