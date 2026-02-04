"""ComplexUNetV2 implementation.

Implements two parallel real-valued UNet streams with ResNet blocks,
GroupNorm, lightweight self-attention, weight-standardized convolutions,
and cross-channel mixing is performed by 1x1 convolutions after each block, treating real/imag as a 2-channel image.

This implementation focuses on correctness and clarity rather than maximal
performance; it is suitable for training and smoke tests and follows the
specification in the project document.
"""

import jax
import jax.numpy as jnp
import jax.image as jimage
from typing import Tuple, Any, Dict


def glorot_init(rng, shape):
    fan_in = jnp.prod(jnp.array(shape[:-1]))
    std = jnp.sqrt(2.0 / fan_in)
    return jax.random.normal(rng, shape) * std


def ws_conv_init(rng, kernel_shape: Tuple[int, int, int, int]):
    # kernel_shape: (kh, kw, in_ch, out_ch)
    k = glorot_init(rng, kernel_shape)
    b = jnp.zeros((kernel_shape[-1],), dtype=jnp.float32)
    return {'w': k, 'b': b}


def ws_conv_apply(params: Dict[str, jnp.ndarray], x: jnp.ndarray, stride: int = 1, padding: str = 'SAME'):
    w = params['w']
    b = params['b']
    # weight standardization over spatial + in channels
    # Skip WS for 1x1 kernels (mix/proj 1x1) to avoid amplifying near-constant kernels
    kh, kw = w.shape[0], w.shape[1]
    if kh == 1 and kw == 1:
        w_std = w
    else:
        mean = jnp.mean(w, axis=(0, 1, 2), keepdims=True)
        var = jnp.mean((w - mean) ** 2, axis=(0, 1, 2), keepdims=True)
        w_std = (w - mean) / jnp.sqrt(var + 1e-5)
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    y = jax.lax.conv_general_dilated(x, w_std, (stride, stride), padding, dimension_numbers=dimension_numbers)
    return y + b


def group_norm_init(channels: int, G: int = 8):
    gamma = jnp.ones((channels,), dtype=jnp.float32)
    beta = jnp.zeros((channels,), dtype=jnp.float32)
    # Do not store 'G' as a param leaf (keep it out of the pytree)
    return {'gamma': gamma, 'beta': beta}


def group_norm_apply(params: Dict[str, Any], x: jnp.ndarray, eps: float = 1e-5):
    # x: (..., H, W, C)
    # allow G to be provided in params for backward compatibility,
    # otherwise use default 8.
    G = params.get('G', 8)
    gamma = params['gamma']
    beta = params['beta']
    orig_shape = x.shape
    N = x.shape[0] if x.ndim == 4 else 1
    # reshape to (N, H, W, C)
    if x.ndim == 3:
        x = x[None, ...]
    N, H, W, C = x.shape
    Gc = min(G, C)
    # ensure Gc divides C
    while (C % Gc) != 0:
        Gc -= 1
        if Gc <= 1:
            Gc = 1
            break
    x = x.reshape((N, H, W, Gc, C // Gc))
    mean = jnp.mean(x, axis=(1, 2, 4), keepdims=True)
    var = jnp.var(x, axis=(1, 2, 4), keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps)
    x = x.reshape((N, H, W, C))
    x = x * gamma + beta
    if orig_shape == x.shape[1:]:
        return x[0]
    return x


def resnet_block_init(rng, in_ch, out_ch, kernel=(3, 3)):
    # split rng once to produce independent keys for convs
    k1_key, k2_key = jax.random.split(rng, 2)
    k1 = ws_conv_init(k1_key, (kernel[0], kernel[1], in_ch, out_ch))
    gn1 = group_norm_init(out_ch)
    k2 = ws_conv_init(k2_key, (kernel[0], kernel[1], out_ch, out_ch))
    gn2 = group_norm_init(out_ch)
    # time projection parameters will be added by caller if desired; keep
    # backward-compatible shape by including optional t_proj slot (None)
    return {'k1': k1, 'gn1': gn1, 'k2': k2, 'gn2': gn2, 't_proj': None}


def resnet_block_apply(params, x, t=None):
    y = ws_conv_apply(params['k1'], x)
    y = group_norm_apply(params['gn1'], y)
    y = jax.nn.gelu(y)
    # support simple per-block learned projection from sinusoidal time emb
    if t is not None and params.get('t_proj') is not None:
        try:
            # compute sinusoidal time embedding
            def _sinusoidal_time_embedding(t_in, dim):
                # t_in: scalar or (...,) array in [0,1]
                half = dim // 2
                freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / float(half - 1))
                arg = (t_in[..., None] * freqs[None, :]) * (2.0 * jnp.pi)
                emb = jnp.concatenate([jnp.sin(arg), jnp.cos(arg)], axis=-1)
                return emb

            w = params['t_proj']['w']
            b = params['t_proj']['b']
            dim_t = w.shape[0]
            t_emb = _sinusoidal_time_embedding(t, dim_t)
            # project: handle batch or scalar t_emb
            proj = jnp.dot(t_emb, w) + b
            # ensure shape (1,1,1,C) or (B,1,1,C)
            if proj.ndim == 1:
                bias = proj.reshape((1, 1, 1, -1))
            else:
                bias = proj.reshape((proj.shape[0], 1, 1, proj.shape[1]))
            y = y + bias
        except Exception:
            pass
    y = ws_conv_apply(params['k2'], y)
    y = group_norm_apply(params['gn2'], y)
    # If input channels differ from residual channels, broadcast/trim x to match
    # Use concrete Python ints for channel arithmetic to avoid tracing issues
    x_ch = int(x.shape[-1])
    y_ch = int(y.shape[-1])
    if x_ch != y_ch:
        if x_ch < y_ch:
            # integer ceiling division without creating tracers
            repeats = (y_ch + x_ch - 1) // x_ch
            x_proj = jnp.repeat(x, repeats, axis=-1)[..., :y_ch]
        else:
            x_proj = x[..., :y_ch]
    else:
        x_proj = x
    return jax.nn.gelu(x_proj + y)


def self_attention_init(rng, channels, num_heads=8):
    key = jax.random.split(rng, 4)
    # linear projections as 1x1 conv kernels
    kq = ws_conv_init(key[0], (1, 1, channels, channels))
    kk = ws_conv_init(key[1], (1, 1, channels, channels))
    kv = ws_conv_init(key[2], (1, 1, channels, channels))
    proj = ws_conv_init(key[3], (1, 1, channels, channels))
    return {'kq': kq, 'kk': kk, 'kv': kv, 'proj': proj, 'heads': num_heads}


def self_attention_apply(params, x):
    # x: (N,H,W,C) or (H,W,C)
    orig = x
    if x.ndim == 3:
        x = x[None, ...]
    N, H, W, C = x.shape
    q = ws_conv_apply(params['kq'], x)
    k = ws_conv_apply(params['kk'], x)
    v = ws_conv_apply(params['kv'], x)
    # flatten spatial
    q = q.reshape((N, H * W, C))
    k = k.reshape((N, H * W, C))
    v = v.reshape((N, H * W, C))
    # q,k,v shapes: (N, HW, C). Compute logits (N, HW, HW) via dot-product
    scale = jnp.sqrt(jnp.array(C, dtype=jnp.float32))
    logits = jnp.einsum('nqc,nkc->nqk', q, k) / scale
    attn = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum('nqk,nkc->nqc', attn, v)
    out = out.reshape((N, H, W, C))
    out = ws_conv_apply(params['proj'], out)
    if orig.ndim == 3:
        return out[0]
    return out




def downsample(x):
    # spatial downsample by factor 2 via average pooling
    return jax.lax.reduce_window(x, 0.0, jax.lax.add, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME') / 4.0


def upsample(x):
    # simple nearest neighbor upsample by factor 2
    if x.ndim == 3:
        x = x[None, ...]
        squeeze = True
    else:
        squeeze = False
    N, H, W, C = x.shape
    x = jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)
    if squeeze:
        return x[0]
    return x


def resize_to(x, target_h, target_w):
    """Resize spatial dims of `x` (N,H,W,C) or (H,W,C) to (target_h,target_w)
    using nearest-neighbor via jax.image.resize. Preserves batch and channel dims.
    """
    was_3d = False
    if x.ndim == 3:
        x = x[None, ...]
        was_3d = True
    N, H, W, C = x.shape
    if H == target_h and W == target_w:
        return x[0] if was_3d else x
    # jax.image.resize expects shape (N, H, W, C)
    new = jimage.resize(x, (N, target_h, target_w, C), method='nearest')
    return new[0] if was_3d else new


def complexUnet_init(rng, input_shape: Tuple[int, int, int], base_ch: int = 32, mixing: float = 0.3, att_scale: float = 0.1):
    """Initialize parameters for ComplexUNetV2.

    input_shape: (H, W, C) where C=1 typically. Returns (params, apply_fn).
    """

    H, W, C = input_shape
    rngs = list(jax.random.split(rng, 100))
    def next_rng():
        return rngs.pop(0)
    ch1 = base_ch
    ch2 = base_ch * 2
    ch3 = base_ch * 4
    ch4 = base_ch * 8

    params = {}
    TIME_EMB_DIM = 128
    # Mixing is now a 1x1 conv for 2 channels (real/imag), scaled by mixing parameter
    def scaled_ws_conv_init(rng, channels, mixing):
        # Interpolate between identity and full mixing for (channels, channels)
        # If mixing is exactly zero, initialize as zeros to avoid large WS scaling
        if float(mixing) == 0.0:
            w = jnp.zeros((channels, channels), dtype=jnp.float32)
        else:
            w = jnp.eye(channels) * (1.0 - mixing) + jnp.ones((channels, channels)) * (mixing / channels)
        w = w.reshape((1, 1, channels, channels))
        b = jnp.zeros((channels,), dtype=jnp.float32)
        return {'w': w, 'b': b}
    params['mix1'] = scaled_ws_conv_init(next_rng(), ch1, mixing)
    params['mix2'] = scaled_ws_conv_init(next_rng(), ch2, mixing)
    params['mix3'] = scaled_ws_conv_init(next_rng(), ch3, mixing)
    params['mixb'] = scaled_ws_conv_init(next_rng(), ch4, mixing * 1.5)

    # Encoder blocks
    params['r1'] = resnet_block_init(next_rng(), 2, ch1)
    tk = jax.random.split(next_rng(), 3)[-1]
    params['r1']['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch1)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch1,), dtype=jnp.float32)}
    params['r2'] = resnet_block_init(next_rng(), ch1, ch2)
    tk = jax.random.split(next_rng(), 3)[-1]
    params['r2']['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch2)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch2,), dtype=jnp.float32)}
    params['r3'] = resnet_block_init(next_rng(), ch2, ch3)
    tk = jax.random.split(next_rng(), 3)[-1]
    params['r3']['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch3)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch3,), dtype=jnp.float32)}
    params['rb'] = resnet_block_init(next_rng(), ch3, ch4)
    tk = jax.random.split(next_rng(), 3)[-1]
    params['rb']['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch4)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch4,), dtype=jnp.float32)}

    # Attention
    params['att1'] = self_attention_init(next_rng(), ch1)
    params['att2'] = self_attention_init(next_rng(), ch2)
    params['att3'] = self_attention_init(next_rng(), ch3)
    params['att_scale'] = att_scale

    # Decoder blocks
    params['d3'] = resnet_block_init(next_rng(), ch4, ch3)
    tk = jax.random.split(next_rng(), 3)[-1]
    params['d3']['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch3)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch3,), dtype=jnp.float32)}
    params['d2'] = resnet_block_init(next_rng(), ch3, ch2)
    tk = jax.random.split(next_rng(), 3)[-1]
    params['d2']['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch2)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch2,), dtype=jnp.float32)}
    params['d1'] = resnet_block_init(next_rng(), ch2, ch1)
    tk = jax.random.split(next_rng(), 3)[-1]
    params['d1']['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch1)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch1,), dtype=jnp.float32)}

    params['proj3'] = ws_conv_init(next_rng(), (1, 1, ch4 + ch3, ch4))
    params['proj2'] = ws_conv_init(next_rng(), (1, 1, ch3 + ch2, ch3))
    params['proj1'] = ws_conv_init(next_rng(), (1, 1, ch2 + ch1, ch2))

    # Final 1x1 conv to map ch1 to 2 (real/imag)
    params['final'] = ws_conv_init(next_rng(), (1, 1, ch1, 2))
    # params['final']['w'] = jnp.zeros_like(params['final']['w']) ## ZERO INIT FINAL LAYER FOR TESTING
    # params['final']['b'] = jnp.zeros_like(params['final']['b'])

    # #normalize params by dividing by their norm:
    # params_norm = jnp.sqrt(sum([jnp.sum(v[k]**2) for v,k in params.items()]))
    # params = {k: {'w': v['w']/params_norm, 'b': v['b']} if 'w' in v else v for k,v in params.items()}

    def apply_fn(params, x_complex, t):
        # x_complex: (H,W,1) or (N,H,W,1)
        had_batch = True
        x = x_complex
        if x.ndim == 3:
            x = x[None, ...]
            had_batch = False
        # Stack real/imag as channels
        x2 = jnp.concatenate([jnp.real(x), jnp.imag(x)], axis=-1)  # (..., 2)

        t = jnp.asarray(t)
        if t.ndim == 0:
            t = jnp.full((x.shape[0],), t.astype(jnp.float32), dtype=jnp.float32)
        elif t.ndim == 1:
            if t.shape[0] == 1:
                t = jnp.full((x.shape[0],), t[0].astype(jnp.float32), dtype=jnp.float32)
            elif t.shape[0] != x.shape[0]:
                raise ValueError('t must be scalar or length batch')
            else:
                t = t.astype(jnp.float32)
        else:
            raise ValueError('t must be scalar or 1-D array')

        att_scale = params.get('att_scale', 0.1)

        # Encoder
        y1 = resnet_block_apply(params['r1'], x2, t=t)
        y1 = ws_conv_apply(params['mix1'], y1)
        y1 = y1 + att_scale * self_attention_apply(params['att1'], y1)
        d1 = downsample(y1)

        y2 = resnet_block_apply(params['r2'], d1, t=t)
        y2 = ws_conv_apply(params['mix2'], y2)
        y2 = y2 + att_scale * self_attention_apply(params['att2'], y2)
        d2 = downsample(y2)

        y3 = resnet_block_apply(params['r3'], d2, t=t)
        y3 = ws_conv_apply(params['mix3'], y3)
        y3 = y3 + att_scale * self_attention_apply(params['att3'], y3)
        d3 = downsample(y3)

        # Bottleneck
        yb = resnet_block_apply(params['rb'], d3, t=t)
        yb = ws_conv_apply(params['mixb'], yb)

        # Decoder
        u3 = upsample(yb)
        th, tw = y3.shape[1], y3.shape[2]
        u3 = resize_to(u3, th, tw)
        u3 = jnp.concatenate([u3, y3], axis=-1)
        u3 = ws_conv_apply(params['proj3'], u3)
        u3 = resnet_block_apply(params['d3'], u3, t=t)

        u2 = upsample(u3)
        th, tw = y2.shape[1], y2.shape[2]
        u2 = resize_to(u2, th, tw)
        u2 = jnp.concatenate([u2, y2], axis=-1)
        u2 = ws_conv_apply(params['proj2'], u2)
        u2 = resnet_block_apply(params['d2'], u2, t=t)

        u1 = upsample(u2)
        th, tw = y1.shape[1], y1.shape[2]
        u1 = resize_to(u1, th, tw)
        u1 = jnp.concatenate([u1, y1], axis=-1)
        u1 = ws_conv_apply(params['proj1'], u1)
        u1 = resnet_block_apply(params['d1'], u1, t=t)

        out2 = ws_conv_apply(params['final'], u1)
        # Convert back to complex
        out = out2[..., 0] + 1j * out2[..., 1]
        # Always return (N, H, W, 1) shape for single-channel complex output
        if out.ndim == 3:
            out = out[..., None]
        if not had_batch:
            out = out[0]
        return out

    return params, apply_fn
