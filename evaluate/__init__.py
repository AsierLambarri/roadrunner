"""Evaluation of finished catalogues (roadrunner runs, ACCME results).

Per run, as standalone scripts writing to ``RUN/evaluation/`` (report, log, PDFs under ``plots/``):

    python3 evaluate/run_all.py --path RUN [--particles DIR] [--workers N]    # every per-run point
    python3 evaluate/soundness.py --path RUN ...                               # point 1, technically sound
    python3 evaluate/stability.py --path RUN ...                               # point 2, stable in time
    python3 evaluate/model.py --path RUN ...                                   # goal 3, overlap, calibration

Two or more catalogues (roadrunner runs or ACCME results), the first the reference, written to the first
roadrunner run's ``evaluation/<REF>_vs_<OTHERS>/`` (or ``--out``):

    python3 evaluate/science.py --path REF --vs OTHER [...]                   # goal 5, births and ex-situ
    python3 evaluate/compare.py --path REF --vs OTHER [...]                   # goal 6, where they differ

``python -m evaluate sweep`` (goal 7) gathers the SCOREs of runs into one table.

``soundness`` and ``stability`` give a verdict (FAIL and FLAG episodes), the others only measure.
"""
