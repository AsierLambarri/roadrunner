"""Evaluate one roadrunner run: every point, each as its own process, all writing to RUN/evaluation/.

    python3 evaluate/run_all.py --path RUN [--particles DIR] [--workers N] [--snaps A:B] [--every N]
                                [--feedback] [--dependence] [--light-every N]
                                [--only soundness,stability,model]

Each point is a standalone script (``python3 evaluate/<point>.py --path RUN ...``) that writes its
report ``<point>.txt``, its log ``<point>.log`` and its PDFs under ``plots/``; this script only calls
them one after another, with the same options. A point that fails does not stop the others; the exit
code is the number of points that failed.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

POINTS = ["soundness", "stability", "model"]
HERE = Path(__file__).resolve().parent


def main():
    """Run the chosen points, one process each, with the options that apply to them."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", required=True, help="the roadrunner run directory")
    p.add_argument("--particles", help="directory holding particle_data/ (default: the run's own)")
    p.add_argument("--workers", type=int, help="processes per point (default: each script's)")
    p.add_argument("--snaps", help="first:last snapshot, both included (default: all)")
    p.add_argument("--every", type=int, help="model: every N-th snapshot (default: the script's)")
    p.add_argument("--feedback", action="store_true", help="model: 3.2b, replay the fit without alpha in EM")
    p.add_argument("--dependence", action="store_true",
                   help="model: 3.4, how much of alpha the Gaussians contain")
    p.add_argument("--light-every", type=int,
                   help="model: 3.2b and 3.4 on every N-th snapshot (default: the script's)")
    p.add_argument("--only", help=f"comma-separated points (default: {','.join(POINTS)})")
    args = p.parse_args()

    chosen = args.only.split(",") if args.only else POINTS
    unknown = set(chosen) - set(POINTS)
    if unknown:
        p.error(f"unknown points {sorted(unknown)}; choose from {POINTS}")

    failed = []
    for point in chosen:
        cmd = [sys.executable, str(HERE / f"{point}.py"), "--path", args.path]
        for flag in ("particles", "workers", "snaps"):
            if getattr(args, flag) is not None:
                cmd += [f"--{flag}", str(getattr(args, flag))]
        if point == "model":
            if args.every is not None:
                cmd += ["--every", str(args.every)]
            if args.feedback:
                cmd += ["--feedback"]
            if args.dependence:
                cmd += ["--dependence"]
            if args.light_every is not None:
                cmd += ["--light-every", str(args.light_every)]
        start = time.time()
        code = subprocess.run(cmd).returncode
        print(f"[run_all] {point}: exit {code} after {time.time() - start:.0f} s", flush=True)
        if code:
            failed.append(point)
    print(f"[run_all] done; results in {Path(args.path) / 'evaluation'}"
          + (f"; failed: {failed}" if failed else ""), flush=True)
    sys.exit(len(failed))


if __name__ == "__main__":
    main()
