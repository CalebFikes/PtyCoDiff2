"""Compatibility shim for the old `ptycho.sampling` module.

The real implementation has been merged into `ptycho.sampler`. This file
keeps a minimal shim so older imports continue to work while emitting a
deprecation message.
"""

import warnings
from importlib import import_module

warnings.warn("ptycho.sampling has been merged into ptycho.sampler; import from ptycho.sampler instead", DeprecationWarning)

# Re-export names from the merged module for backward compatibility
_m = import_module('ptycho.sampler')
posterior_score_measurement = _m.posterior_score_measurement
_measurement_score_complex = _m._measurement_score_complex
generate_ptycho_posterior_samples_ula = _m.generate_ptycho_posterior_samples_ula

__all__ = ["posterior_score_measurement", "_measurement_score_complex", "generate_ptycho_posterior_samples_ula"]
