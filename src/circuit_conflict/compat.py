"""
compat.py — shims for library version drift.

Kept in one place, and each shim reads the REAL value from its new location.
Nothing here fabricates a config value: a wrong rotary setting would produce
silently incorrect activations, which is worse than a model that fails to load.
"""

from __future__ import annotations


def patch_gptneox_rotary() -> bool:
    """
    transformers >= 5.0 moved GPTNeoXConfig.rotary_pct / .rotary_emb_base into
    the `rope_parameters` dict, but transformer-lens 2.16.x still reads the old
    attribute names, so Pythia fails to load with AttributeError.

    Re-expose the legacy names as read-only properties backed by the new dict.
    Returns True if the shim was applied.
    """
    try:
        from transformers import GPTNeoXConfig
    except ImportError:
        return False

    if hasattr(GPTNeoXConfig, "rotary_pct"):
        return False

    def _rope(self, key, default):
        params = getattr(self, "rope_parameters", None) or {}
        return params.get(key, default)

    GPTNeoXConfig.rotary_pct = property(
        lambda self: _rope(self, "partial_rotary_factor", 1.0))
    GPTNeoXConfig.rotary_emb_base = property(
        lambda self: _rope(self, "rope_theta", 10000))
    return True


def apply_all() -> None:
    patch_gptneox_rotary()
