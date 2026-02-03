"""Compatibility shim: re-export diffusion score APIs from `diffusion` package.

Keep a small shim in `ptycho` so existing imports continue to work while the
real implementations live in `diffusion`.
"""

from diffusion.model import score_apply, ComplexUNetV2Placeholder

__all__ = ["score_apply", "ComplexUNetV2Placeholder"]
