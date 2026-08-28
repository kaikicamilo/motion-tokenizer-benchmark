import os
import yaml


class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        # Recursively convert dictionaries to AttributeDict
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttributeDict(value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No attribute named '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`.

    Dicts are merged key by key (override wins on leaf conflicts); any non-dict
    value in override (list, scalar) replaces the base value — lists are not
    merged. Returns a new dict; the arguments are not mutated.
    """
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_raw(config_dir: str, _seen=None):
    """Load a YAML; if it has a `_base_` key, load the base file(s) first and merge
    this one on top. `_base_` is a string or a list of strings, each resolved
    relative to the file that declares it. Guards against inheritance cycles.
    """
    if _seen is None:
        _seen = set()
    abspath = os.path.abspath(config_dir)
    if abspath in _seen:
        raise ValueError(f"config inheritance cycle detected at: {abspath}")
    _seen.add(abspath)

    with open(config_dir, 'r') as f:
        raw = yaml.safe_load(f) or {}

    bases = raw.pop('_base_', None)
    if bases is None:
        return raw

    if isinstance(bases, str):
        bases = [bases]

    here = os.path.dirname(abspath)
    merged = {}
    for b in bases:
        b_path = b if os.path.isabs(b) else os.path.join(here, b)
        base_cfg = _load_raw(b_path, _seen=set(_seen))  # per-branch copy of _seen
        merged = _deep_merge(merged, base_cfg)

    # this file overrides its bases
    merged = _deep_merge(merged, raw)
    return merged


def load_config(config_dir="config/train.yaml"):
    """Drop-in replacement for SnapMoGen's load_config: identical for a YAML without
    `_base_`; with `_base_`, applies inheritance (recursive merge) first.
    """
    config = _load_raw(config_dir)
    config['config_dir'] = config_dir
    return AttributeDict(config)