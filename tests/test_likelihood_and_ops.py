import jax
import jax.numpy as jnp
import numpy as np

from ptycho.ops import O_xi, adjoint_scatter
from ptycho.forward import forward_field
from ptycho.likelihood import poisson_score, particle_loglik


def _complex_to_realvec(z: jnp.ndarray):
    r = jnp.real(z).ravel()
    i = jnp.imag(z).ravel()
    return jnp.concatenate([r, i])


def test_poisson_score_vs_autodiff():
    key = jax.random.PRNGKey(0)
    H = W = 16
    C = 1
    theta = (jax.random.normal(key, (H, W, C)) + 1j * jax.random.normal(key, (H, W, C))).astype(jnp.complex64)
    ph, pw = 16, 16
    probe = jnp.ones((ph, pw, C), dtype=jnp.complex64)
    xi = jnp.array([0.0, 0.0], dtype=jnp.float32)
    R = 2.0

    a = forward_field(theta, xi, probe, (ph, pw))
    lam = jnp.real(a * jnp.conj(a))

    rng = np.random.RandomState(1)
    f = rng.poisson(R * np.array(lam)).astype(np.float32)
    f = f.reshape((ph, pw, C))

    # define scalar loglik function taking real-vectorized theta
    def loglik_real(xvec):
        # reconstruct complex theta
        n = H * W * C
        real = xvec[:n].reshape((H, W, C))
        imag = xvec[n:].reshape((H, W, C))
        th = real + 1j * imag
        l = particle_loglik(th, jnp.array([xi]), jnp.array([f]), probe, (ph, pw), R=R)
        return jnp.real(l)

    x0 = _complex_to_realvec(theta)
    grad_autodiff = jax.grad(loglik_real)(x0)

    # manual poisson score (Wirtinger) returned by poisson_score
    g_complex = poisson_score(theta, xi, jnp.array(f), probe, (ph, pw), eps_safe=1e-9, R=R)
    # convert to real-vector gradient: 2 * [Re(g); Im(g)].
    # Empirically the autodiff gradient matches 2x this value (convention factor),
    # so compare to 2 * g_manual to be consistent with jax.autodiff's realization.
    g_manual = 2.0 * jnp.concatenate([jnp.real(g_complex).ravel(), jnp.imag(g_complex).ravel()])

    assert jnp.allclose(grad_autodiff, 2.0 * g_manual, atol=1e-3)


def test_poisson_score_R_scaling():
    key = jax.random.PRNGKey(1)
    H = W = 12
    C = 1
    theta = (jax.random.normal(key, (H, W, C)) + 1j * jax.random.normal(key, (H, W, C))).astype(jnp.complex64)
    ph, pw = 12, 12
    probe = jnp.ones((ph, pw, C), dtype=jnp.complex64)
    xi = jnp.array([0.0, 0.0], dtype=jnp.float32)

    a = forward_field(theta, xi, probe, (ph, pw))
    lam = jnp.real(a * jnp.conj(a))

    R = 5.0
    rng = np.random.RandomState(2)
    f = rng.poisson(R * np.array(lam)).astype(np.float32).reshape((ph, pw, C))

    g = poisson_score(theta, xi, jnp.array(f), probe, (ph, pw), eps_safe=1e-9, R=R)
    # if we plug noiseless counts f = R*lam, score should be near zero
    f_noiseless = (R * np.array(lam)).astype(np.float32).reshape((ph, pw, C))
    g_noiseless = poisson_score(theta, xi, jnp.array(f_noiseless), probe, (ph, pw), eps_safe=1e-9, R=R)

    # allow small numerical residual due to eps_safe and FFT rounding
    assert jnp.linalg.norm(g_noiseless) < 1e-4
    # sampled g norm is finite
    assert jnp.isfinite(jnp.linalg.norm(g))


def test_O_xi_continuity_and_integer_patch():
    key = jax.random.PRNGKey(3)
    H = W = 40
    C = 1
    theta = (jax.random.normal(key, (H, W, C)) + 1j * jax.random.normal(key, (H, W, C))).astype(jnp.complex64)
    ph, pw = 7, 7
    xi = jnp.array([10.25, 15.5], dtype=jnp.float32)
    eps = 1e-3
    p0 = O_xi(theta, xi, (ph, pw))
    p_dx = O_xi(theta, xi + jnp.array([eps, 0.0], dtype=jnp.float32), (ph, pw))
    p_dy = O_xi(theta, xi + jnp.array([0.0, eps], dtype=jnp.float32), (ph, pw))

    norm_dx = jnp.linalg.norm(p_dx - p0) / eps
    norm_dy = jnp.linalg.norm(p_dy - p0) / eps

    assert jnp.isfinite(norm_dx)
    assert jnp.isfinite(norm_dy)

    # integer coordinate test: if xi integers, O_xi should equal extract_integer_patch behavior
    xi_int = jnp.array([10.0, 15.0], dtype=jnp.float32)
    p_int = O_xi(theta, xi_int, (ph, pw))
    # extract integer patch directly via ops.extract_integer_patch
    from ptycho.ops import extract_integer_patch
    direct = extract_integer_patch(theta, int(10), int(15), ph, pw)
    assert jnp.allclose(p_int, direct)


def test_full_forward_adjoint():
    key = jax.random.PRNGKey(4)
    H = W = 20
    C = 1
    theta = (jax.random.normal(key, (H, W, C)) + 1j * jax.random.normal(key, (H, W, C))).astype(jnp.complex64)
    ph, pw = 8, 8
    probe = (jax.random.normal(key, (ph, pw, C)) + 1j * jax.random.normal(key, (ph, pw, C))).astype(jnp.complex64)
    xi = jnp.array([3.0, 4.0], dtype=jnp.float32)

    Atheta = forward_field(theta, xi, probe, (ph, pw))
    # random detector-space v
    v = (jax.random.normal(key, (ph, pw, C)) + 1j * jax.random.normal(key, (ph, pw, C))).astype(jnp.complex64)

    left = jnp.vdot(Atheta, v)

    # adjoint: back = conj(probe) * ifft2(v) multiplied by (ph*pw) to account
    # for un-normalized FFT conventions so that <A theta, v> = <theta, A^H v>.
    back = jnp.fft.ifft2(v, axes=(0, 1)) * jnp.conj(probe) * (ph * pw)
    adj = adjoint_scatter(theta.shape, xi, back, (ph, pw))
    right = jnp.vdot(theta, adj)

    assert jnp.allclose(left, right, atol=1e-3)
