"""steermech tests — the eval-logic tests moved to hidden-directions with
the calibration core; here we test what this repo still owns."""
import pytest


def test_plot_module_is_hd_free():
    """make demo must not require hidden-directions/torch."""
    import steermech.plot  # noqa: F401


def test_calibration_reexport_or_helpful_error():
    import steermech
    try:
        fn = steermech.load_benign
    except ModuleNotFoundError as e:
        assert "hidden-directions[calibrate]" in str(e)
    else:
        assert callable(fn)


def test_unknown_attr_raises_attribute_error():
    import steermech
    with pytest.raises(AttributeError):
        steermech.nonexistent_thing
