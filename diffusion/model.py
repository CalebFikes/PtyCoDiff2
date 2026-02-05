"""Minimal diffusion score stub and placeholder ComplexUNetV2.

All score / UNet code lives here per refactor request.
"""

from .complex_unet import complexUnet_init
import warnings
import jax
import jax.numpy as jnp
import copy
import numpy as _np


def create_complexUnet(rng, input_shape=(32, 32, 1, 1), mixing: float = 0.3, base_ch: int = 32, att_scale: float = 0.1):
    """Factory to create params and apply function for the complexUnet.

    Returns (params, apply_fn).
    """
    # normalize input_shape: accept (H,W,C) or (H,W,C,1)
    if isinstance(input_shape, (tuple, list)) and len(input_shape) == 4 and input_shape[3] == 1:
        norm_shape = (input_shape[0], input_shape[1], input_shape[2])
    else:
        norm_shape = tuple(input_shape)
    params_full, apply_full = complexUnet_init(rng, norm_shape, base_ch, mixing=mixing, att_scale=att_scale)

    # Separate static integer leaves (e.g., group counts 'G') from trainable params.
    statics = {}
    def _walk_extract(obj, path=''):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk_extract(v, path + '/' + k)
        else:
            if isinstance(obj, (int, bool)) or (hasattr(obj, 'dtype') and _np.issubdtype(getattr(obj, 'dtype', None), _np.integer)):
                statics[path] = obj

    # ensure att_scale is treated as a static hyperparameter (not trainable)
    if isinstance(params_full, dict) and 'att_scale' in params_full:
        statics['/att_scale'] = params_full.get('att_scale')

    _walk_extract(params_full)

    # Create trainable params: copy and convert integer leaves to float arrays
    train_params = copy.deepcopy(params_full)
    def _walk_replace(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    _walk_replace(v)
                else:
                    if isinstance(v, (int, bool)):
                        obj[k] = jnp.array(v, dtype=jnp.float32)
                    else:
                        try:
                            d = getattr(v, 'dtype', None)
                            if d is not None and _np.issubdtype(d, _np.integer):
                                obj[k] = jnp.array(v, dtype=jnp.float32)
                        except Exception:
                            pass
        return obj

    train_params = _walk_replace(train_params)

    # Lightweight merge: build a new nested dict that references trainable arrays
    # but overrides static leaves with their Python values. This avoids deep-copying
    # large arrays on every call while keeping the apply function a pure Python
    # closure suitable to be treated as `static` by JIT boundaries.
    def _merge_with_statics(p, path=''):
        if isinstance(p, dict):
            out = {}
            for k, v in p.items():
                out[k] = _merge_with_statics(v, path + '/' + k)
            return out
        # leaf
        return statics.get(path, p)

    def apply_fn(train_p, x_t, t):
        full = _merge_with_statics(train_p, '')
        return apply_full(full, x_t, t)

    return train_params, apply_fn


def load_checkpoint_params(path, rng, input_shape=(28, 28, 1), mixing: float = 0.0, att_scale: float = 0.0):
    """Load a checkpoint saved by `training.save_pytree_npz` (.npz) and return
    (params, apply_fn). This loader intentionally does NOT attempt legacy
    real/imag conversion — use training's save format so the model can be
    reconstructed from metadata.

    The saved .npz must contain 'treedef_params', 'params_XXXX' leaves and
    'metadata'. The metadata should include `base_ch`, `mixing` (optional)
    and `att_scale`.
    """
    import pickle as _pickle
    import numpy as _np
    try:
        archive = _np.load(path, allow_pickle=False)
    except Exception as e:
        raise ValueError(f"Failed to open checkpoint {path}: {e}")

    if 'treedef_params' not in archive:
        raise ValueError(f"Checkpoint {path} is not a training .npz (missing treedef_params). Legacy conversion is disabled.")

    treedef_bytes = archive['treedef_params'].tobytes()
    treedef = _pickle.loads(treedef_bytes)
    leaves = []
    i = 0
    while True:
        k = f'params_{i:04d}'
        if k in archive:
            leaves.append(jnp.array(archive[k]))
            i += 1
        else:
            break

    params_recon = jax.tree_util.tree_unflatten(treedef, leaves)

    meta_bytes = archive['metadata'].tobytes()
    meta = _pickle.loads(meta_bytes)
    base_ch = meta.get('base_ch', None)
    mixing_meta = meta.get('mixing', None)
    att_meta = meta.get('att_scale', att_scale)

    if base_ch is None:
        base_ch = input_shape[2] if len(input_shape) >= 3 else 32
    mix_for_init = mixing_meta if mixing_meta is not None else mixing
    train_params, apply_fn = create_complexUnet(rng, input_shape=input_shape, base_ch=base_ch, mixing=mix_for_init, att_scale=att_meta)
    return params_recon, apply_fn


def score_apply(params, x_t: jnp.ndarray, t: float) -> jnp.ndarray:
    """Apply the score model. If `params` is None, fall back to a lightweight stub.

    This keeps imports lightweight; to use the full `complexUnet`, call
    `create_complexUnet` to obtain `(params, apply_fn)` and pass `params` here.
    """
    if params is None:
        # fallback stub: return a score (∇_x log p) approximation
        sigma = jnp.sin(jnp.pi * 0.5 * t)
        sigma = jnp.maximum(sigma, 1e-6)
        return -x_t / (sigma ** 2)
    # if params is a tuple (params, apply_fn) accept it
    if isinstance(params, tuple) and len(params) == 2 and callable(params[1]):
        p, apply_fn = params
        # model returns eps_pred; convert to score s = -eps_pred / sigma
        eps_pred = apply_fn(p, x_t, t)
        # Use the same cosine schedule as training (small s-offset) to compute sigma
        s0 = 0.008
        t_ = jnp.clip(t, 0.0, 1.0)
        f = jnp.cos((t_ + s0) / (1.0 + s0) * jnp.pi / 2.0)
        sigma = jnp.sqrt(jnp.clip(1.0 - f ** 2, 1e-12, 1.0))
        return -eps_pred / sigma
    # otherwise assume params is params dict and user provided apply via closure
    # try to call params as (params, apply_fn) style
    raise ValueError("score_apply requires params=None (stub) or (params, apply_fn) from create_complexUnet")
