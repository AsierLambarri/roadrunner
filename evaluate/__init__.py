"""Evaluation of finished catalogues (roadrunner runs, ACCME results).

Per run, as standalone scripts writing to ``RUN/evaluation/`` (report, log, PDFs under ``plots/``):

    python3 evaluate/run_all.py --path RUN [--particles DIR] [--workers N]    # every per-run point
    python3 evaluate/soundness.py --path RUN ...                               # point 1, technically sound
    python3 evaluate/stability.py --path RUN ...                               # point 2, stable in time
    python3 evaluate/model.py --path RUN ...                                   # goal 3, overlap, calibration

Two or more catalogues, the first the reference (``python -m evaluate -h``): ``science`` (goal 5,
robustness of the science outputs), ``compare`` (goal 6, where they differ); ``sweep`` (goal 7) gathers
the SCOREs of runs into one table.

``soundness`` and ``stability`` give a verdict (FAIL and FLAG episodes), the others only measure.
"""
