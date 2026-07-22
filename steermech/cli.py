"""Console entry points (see pyproject [project.scripts])."""
import runpy
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent / "experiments"


def _run(script):
    sys.argv[0] = str(EXP / script)
    runpy.run_path(str(EXP / script), run_name="__main__")


def calibrate_cli():
    _run("autocalibrate.py")


def discover_cli():
    _run("make_intent.py")


def plot_cli():
    from steermech.plot import main
    main()
