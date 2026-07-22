"""steermech — mechanistic experiments on steering vectors.

The calibration machinery (brainscope client, efficacy/damage eval, intent
discovery, the Optuna optimizer) moved into the factory where it belongs:
`hidden_directions.calibrate` — calibration is part of making a vector, not
a finding about it. This package keeps the experiment side (figures, plots)
and lazily re-exports the calibration names for backward compatibility, so
`make demo` still needs nothing beyond the stdlib + pillow.

    pip install "hidden-directions[calibrate] @ git+https://github.com/moudrkat/hidden-directions"
"""

_CALIBRATE_NAMES = ("chat", "forced_diff", "unembed", "discover_intent",
                    "write_intent", "combined_objective", "damage", "efficacy",
                    "generate_efficacy", "load_benign", "load_intent",
                    "objective", "score_damage", "score_efficacy", "calibrate")

__all__ = list(_CALIBRATE_NAMES)


def __getattr__(name):
    if name in _CALIBRATE_NAMES:
        try:
            import hidden_directions.calibrate as _c
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "steermech's calibration core now lives in hidden-directions. "
                "Install it:  pip install \"hidden-directions[calibrate] @ "
                "git+https://github.com/moudrkat/hidden-directions\""
            ) from e
        val = getattr(_c, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
