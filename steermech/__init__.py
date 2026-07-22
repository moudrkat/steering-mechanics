"""steermech — heretic-grade auto-calibration for additive steering vectors.

Two axes, two datasets (heretic's structure, generalized to any vector):
- efficacy: does the vector achieve its intent? (per-vector intent file)
- damage:   does steering break normal behavior? (one shared benign set,
            measured as KL divergence, vector-agnostic)
"""
from .client import forced_diff, unembed
from .eval import efficacy, damage, objective, load_intent, load_benign
from .discover import discover_intent

__all__ = ["forced_diff", "unembed", "efficacy", "damage", "objective",
           "load_intent", "load_benign", "discover_intent"]
