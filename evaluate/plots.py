"""Where the evaluation PDFs go, which galaxies get their own, and the per-galaxy plotting.

Run-level plots go to ``evaluation/plots/run/<name>.pdf``, a galaxy's to
``evaluation/plots/g<id>/<name>.pdf``. A galaxy gets its own when it is the main host, when it reaches
``MAJOR`` (1000) particles in the snapshots a point reads, or when an episode of the point names it.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate.load import MAJOR, pool


def run_plot(out, name):
    """Path of a run-level plot, ``out/plots/run/<name>.pdf`` (its folder created)."""
    folder = out / "plots" / "run"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{name}.pdf"


def galaxy_plot(out, g, name):
    """Path of a galaxy's plot, ``out/plots/g<id>/<name>.pdf`` (its folder created)."""
    folder = out / "plots" / f"g{int(g)}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{name}.pdf"


def chosen(host, peak, named):
    """The galaxies with their own plots: the main host, those whose ``peak`` particle count reaches
    MAJOR, and those ``named`` in an episode."""
    return sorted({int(host)} | {int(g) for g, n in peak.items() if n >= MAJOR} | {int(g) for g in named})


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def per_galaxy(draw, items, workers, desc):
    """``draw(item)`` for every galaxy's item, over processes (each draws and saves its own PDF)."""
    pool(draw, items, workers, _nothing, desc=desc)


def _nothing():
    """A worker needs no setup to draw."""
