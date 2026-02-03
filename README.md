IPC sampling repo skeleton

This folder contains a minimal implementation of the core ptychography ops, a poisson likelihood score, and a compiled sampler factory (JAX).

Files:
- ptycho/ops.py          : O_xi and adjoint scatter
- ptycho/forward.py      : forward_field and forward_batch
- ptycho/likelihood.py   : poisson_score and particle_loglik
- ptycho/model.py        : minimal score_apply stub (placeholder for ComplexUNetV2)
- ptycho/sampler.py      : make_sampler factory using jax.jit + lax.scan
- smoke_test.py          : quick smoke test to run the compiled sampler

Run smoke test:

python3 smoke_test.py

Notes:
- This is a scaffold for the full project described in the spec. Replace the score stub with a full ComplexUNetV2 implementation (Flax/Haiku) when ready.
