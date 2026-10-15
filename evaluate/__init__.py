"""Evaluation of finished catalogues (roadrunner runs, ACCME results).

Per run: ``soundness`` (point 1, technically sound), ``stability`` (point 2,
stable in time), ``model`` (goal 3, overlap and calibration). Two or more
catalogues, the first the reference: ``science`` (goal 5, robustness of the
science outputs), ``compare`` (goal 6, where they differ). ``sweep`` (goal 7)
gathers the SCOREs of runs into one table. Usage: ``python -m evaluate -h``.

Each point writes one ``<point>.txt`` and a few PNGs; ``soundness`` and
``stability`` give a verdict (FAILs, FLAGs), the others only measure.
"""
