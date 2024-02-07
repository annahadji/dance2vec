"""Utils for reproducibility across files."""
import numpy as np

_rngs = {}


def get_rng(name: str, seed: int = None):
    """Get numpy random number generator, for a particular name. Returns existing
    generator if name already exists."""
    rng = np.random.default_rng(seed=seed)
    return _rngs.setdefault(name, rng)


def setup_rng(names, seeds):
    """Set up random numpy generators with certain names and seeds.
    Will overwrite entry if name already exists."""
    for name, seed in zip(names, seeds):
        _rngs[name] = np.random.default_rng(seed=seed)
