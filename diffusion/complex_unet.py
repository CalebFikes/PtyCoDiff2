"""ComplexUNetV2 implementation.

Implements two parallel real-valued UNet streams with ResNet blocks,
GroupNorm, lightweight self-attention, weight-standardized convolutions,
and cross-channel mixing modules inserted after ResNet/attention blocks.

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


def mix_module_init(rng, channels, mixing=0.3):
    # mixing weights as per-channel or scalar
    mix = jnp.array(mixing, dtype=jnp.float32)
    return {'mix': mix}


def mix_module_apply(params, feat_real, feat_imag):
    mix = params['mix']
    real_out = (1.0 - mix) * feat_real + mix * feat_imag
    imag_out = (1.0 - mix) * feat_imag - mix * feat_real
    return real_out, imag_out


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
    rngs = list(jax.random.split(rng, 200))
    def next_rng():
        return rngs.pop(0)
    # encoder channels
    ch1 = base_ch
    ch2 = base_ch * 2
    ch3 = base_ch * 4
    ch4 = base_ch * 8

    params = {}
    # stream params
    params['real'] = {}
    params['imag'] = {}

    # level 1
    # Add per-block time projection parameters: small learned linear map from
    # sinusoidal embedding to out_ch bias. Choose embedding dim = 128.
    TIME_EMB_DIM = 128
    key = next_rng()
    rb = resnet_block_init(key, C, ch1)
    tk = jax.random.split(key, 3)[-1]
    rb['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch1)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch1,), dtype=jnp.float32)}
    key = next_rng()
    ib = resnet_block_init(key, C, ch1)
    tk = jax.random.split(key, 3)[-1]
    ib['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch1)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch1,), dtype=jnp.float32)}
    params['real']['r1'] = rb
    params['imag']['r1'] = ib
    params['m1'] = mix_module_init(next_rng(), ch1, mixing)

    # level 2
    key = next_rng()
    rb2 = resnet_block_init(key, ch1, ch2)
    tk = jax.random.split(key, 3)[-1]
    rb2['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch2)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch2,), dtype=jnp.float32)}
    key = next_rng()
    ib2 = resnet_block_init(key, ch1, ch2)
    tk = jax.random.split(key, 3)[-1]
    ib2['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch2)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch2,), dtype=jnp.float32)}
    params['real']['r2'] = rb2
    params['imag']['r2'] = ib2
    params['m2'] = mix_module_init(next_rng(), ch2, mixing)

    # level 3
    key = next_rng()
    rb3 = resnet_block_init(key, ch2, ch3)
    tk = jax.random.split(key, 3)[-1]
    rb3['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch3)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch3,), dtype=jnp.float32)}
    key = next_rng()
    ib3 = resnet_block_init(key, ch2, ch3)
    tk = jax.random.split(key, 3)[-1]
    ib3['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch3)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch3,), dtype=jnp.float32)}
    params['real']['r3'] = rb3
    params['imag']['r3'] = ib3
    params['m3'] = mix_module_init(next_rng(), ch3, mixing)

    # bottleneck
    key = next_rng()
    rbb = resnet_block_init(key, ch3, ch4)
    tk = jax.random.split(key, 3)[-1]
    rbb['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch4)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch4,), dtype=jnp.float32)}
    key = next_rng()
    ibb = resnet_block_init(key, ch3, ch4)
    tk = jax.random.split(key, 3)[-1]
    ibb['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch4)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch4,), dtype=jnp.float32)}
    params['real']['rb'] = rbb
    params['imag']['rb'] = ibb
    params['mb'] = mix_module_init(next_rng(), ch4, mixing * 1.5)

    # attention modules
    params['att1'] = self_attention_init(next_rng(), ch1)
    params['att2'] = self_attention_init(next_rng(), ch2)
    params['att3'] = self_attention_init(next_rng(), ch3)

    # store attention scale as a static/leaf so callers can override or disable
    params['att_scale'] = att_scale

    # decoder resnets
    key = next_rng()
    rd3 = resnet_block_init(key, ch4, ch3)
    tk = jax.random.split(key, 3)[-1]
    rd3['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch3)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch3,), dtype=jnp.float32)}
    key = next_rng()
    id3 = resnet_block_init(key, ch4, ch3)
    tk = jax.random.split(key, 3)[-1]
    id3['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch3)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch3,), dtype=jnp.float32)}
    key = next_rng()
    rd2 = resnet_block_init(key, ch3, ch2)
    tk = jax.random.split(key, 3)[-1]
    rd2['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch2)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch2,), dtype=jnp.float32)}
    key = next_rng()
    id2 = resnet_block_init(key, ch3, ch2)
    tk = jax.random.split(key, 3)[-1]
    id2['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch2)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch2,), dtype=jnp.float32)}
    key = next_rng()
    rd1 = resnet_block_init(key, ch2, ch1)
    tk = jax.random.split(key, 3)[-1]
    rd1['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch1)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch1,), dtype=jnp.float32)}
    key = next_rng()
    id1 = resnet_block_init(key, ch2, ch1)
    tk = jax.random.split(key, 3)[-1]
    id1['t_proj'] = {'w': jax.random.normal(tk, (TIME_EMB_DIM, ch1)) * (1.0 / jnp.sqrt(TIME_EMB_DIM)), 'b': jnp.zeros((ch1,), dtype=jnp.float32)}
    params['real']['d3'] = rd3
    params['imag']['d3'] = id3
    params['real']['d2'] = rd2
    params['imag']['d2'] = id2
    params['real']['d1'] = rd1
    params['imag']['d1'] = id1

    # projection convs to map concatenated skip+upsampled channels -> expected in_ch
    params['proj3'] = ws_conv_init(next_rng(), (1, 1, ch4 + ch3, ch4))
    params['proj2'] = ws_conv_init(next_rng(), (1, 1, ch3 + ch2, ch3))
    params['proj1'] = ws_conv_init(next_rng(), (1, 1, ch2 + ch1, ch2))

    # final convs (1x1) as weight-standardized convs
    params['final_real'] = ws_conv_init(next_rng(), (1, 1, ch1, C))
    params['final_imag'] = ws_conv_init(next_rng(), (1, 1, ch1, C))
    # zero-init final convs for stable diffusion training (start as identity-like)
    params['final_real']['w'] = jnp.zeros_like(params['final_real']['w'])
    params['final_real']['b'] = jnp.zeros_like(params['final_real']['b'])
    params['final_imag']['w'] = jnp.zeros_like(params['final_imag']['w'])
    params['final_imag']['b'] = jnp.zeros_like(params['final_imag']['b'])

    def apply_fn(params, x_complex, t):
        # x_complex: (H,W,1) or (N,H,W,1)
        had_batch = True
        x = x_complex
        if x.ndim == 3:
            x = x[None, ...]
            had_batch = False
        # split streams
        xr = jnp.real(x)
        xi = jnp.imag(x)

        # ensure t is a per-example vector of shape (B,)
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

        # attention residual scaling (can be overridden via params['att_scale'])
        att_scale = params.get('att_scale', 0.1)

        # level 1
        y_r1 = resnet_block_apply(params['real']['r1'], xr, t=t)
        y_i1 = resnet_block_apply(params['imag']['r1'], xi, t=t)
        y_r1, y_i1 = mix_module_apply(params['m1'], y_r1, y_i1)
        y_r1 = y_r1 + att_scale * self_attention_apply(params['att1'], y_r1)
        y_i1 = y_i1 + att_scale * self_attention_apply(params['att1'], y_i1)
        # downsample
        d_r1 = downsample(y_r1)
        d_i1 = downsample(y_i1)

        # level 2
        y_r2 = resnet_block_apply(params['real']['r2'], d_r1, t=t)
        y_i2 = resnet_block_apply(params['imag']['r2'], d_i1, t=t)
        y_r2, y_i2 = mix_module_apply(params['m2'], y_r2, y_i2)
        y_r2 = y_r2 + att_scale * self_attention_apply(params['att2'], y_r2)
        y_i2 = y_i2 + att_scale * self_attention_apply(params['att2'], y_i2)
        d_r2 = downsample(y_r2)
        d_i2 = downsample(y_i2)

        # level 3
        y_r3 = resnet_block_apply(params['real']['r3'], d_r2, t=t)
        y_i3 = resnet_block_apply(params['imag']['r3'], d_i2, t=t)
        y_r3, y_i3 = mix_module_apply(params['m3'], y_r3, y_i3)
        y_r3 = y_r3 + att_scale * self_attention_apply(params['att3'], y_r3)
        y_i3 = y_i3 + att_scale * self_attention_apply(params['att3'], y_i3)
        d_r3 = downsample(y_r3)
        d_i3 = downsample(y_i3)

        # bottleneck
        y_rb = resnet_block_apply(params['real']['rb'], d_r3, t=t)
        y_ib = resnet_block_apply(params['imag']['rb'], d_i3, t=t)
        y_rb, y_ib = mix_module_apply(params['mb'], y_rb, y_ib)

        # decoder
        u3_r = upsample(y_rb)
        u3_i = upsample(y_ib)
        # compute robustly: handle batched and unbatched
        if u3_r.ndim == 4:
            th = y_r3.shape[1] if y_r3.ndim == 4 else y_r3.shape[0]
            tw = y_r3.shape[2] if y_r3.ndim == 4 else y_r3.shape[1]
        else:
            th = y_r3.shape[0]
            tw = y_r3.shape[1]
        u3_r = resize_to(u3_r, th, tw)
        u3_i = resize_to(u3_i, th, tw)
        u3_r = jnp.concatenate([u3_r, y_r3], axis=-1)
        u3_i = jnp.concatenate([u3_i, y_i3], axis=-1)
        u3_r = ws_conv_apply(params['proj3'], u3_r)
        u3_i = ws_conv_apply(params['proj3'], u3_i)
        u3_r = resnet_block_apply(params['real']['d3'], u3_r, t=t)
        u3_i = resnet_block_apply(params['imag']['d3'], u3_i, t=t)

        u2_r = upsample(u3_r)
        u2_i = upsample(u3_i)
        # align to y_r2
        th = y_r2.shape[1] if y_r2.ndim == 4 else y_r2.shape[0]
        tw = y_r2.shape[2] if y_r2.ndim == 4 else y_r2.shape[1]
        u2_r = resize_to(u2_r, th, tw)
        u2_i = resize_to(u2_i, th, tw)
        u2_r = jnp.concatenate([u2_r, y_r2], axis=-1)
        u2_i = jnp.concatenate([u2_i, y_i2], axis=-1)
        u2_r = ws_conv_apply(params['proj2'], u2_r)
        u2_i = ws_conv_apply(params['proj2'], u2_i)
        u2_r = resnet_block_apply(params['real']['d2'], u2_r, t=t)
        u2_i = resnet_block_apply(params['imag']['d2'], u2_i, t=t)

        u1_r = upsample(u2_r)
        u1_i = upsample(u2_i)
        # align to y_r1
        th = y_r1.shape[1] if y_r1.ndim == 4 else y_r1.shape[0]
        tw = y_r1.shape[2] if y_r1.ndim == 4 else y_r1.shape[1]
        u1_r = resize_to(u1_r, th, tw)
        u1_i = resize_to(u1_i, th, tw)
        u1_r = jnp.concatenate([u1_r, y_r1], axis=-1)
        u1_i = jnp.concatenate([u1_i, y_i1], axis=-1)
        u1_r = ws_conv_apply(params['proj1'], u1_r)
        u1_i = ws_conv_apply(params['proj1'], u1_i)
        u1_r = resnet_block_apply(params['real']['d1'], u1_r, t=t)
        u1_i = resnet_block_apply(params['imag']['d1'], u1_i, t=t)

        out_r = ws_conv_apply(params['final_real'], u1_r)
        out_i = ws_conv_apply(params['final_imag'], u1_i)
        out = out_r + 1j * out_i
        if not had_batch:
            out = out[0]
        return out

    return params, apply_fn
