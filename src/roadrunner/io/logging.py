#############################################################################
#
# package:   roadrunner.io
# file:      logging.py
# brief:     Structured run-log writer with formatted columns.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   19 may 2026 - Created
#            19 may 2026 - Last edit
#
#############################################################################

import numpy as np

from roadrunner._defaults import COL_WIDTH_RUNTIME, COL_WIDTH_INT, COL_WIDTH_FLOAT


def format_runtime(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{sec:06.3f}"


class RunLogger:
    def __init__(self, log_path, logo_path=None):
        self._log_path = log_path
        self._logo_path = logo_path

    def write_header(self, config):
        with open(self._log_path, "w") as f:
            if self._logo_path:
                try:
                    lines = np.loadtxt(
                        self._logo_path, dtype=str, delimiter=",", comments=None
                    )
                    f.write("\n".join(lines[:, 0]) + "\n")
                except Exception:
                    pass
            f.write("\n\n")
            f.write(f"-output_dir: {config.get('output_dir', 'N/A')}\n")
            f.write(f"-halo_model: {config.get('halo_model', 'N/A')}\n")
            f.write(f"-cov_type: {config.get('cov_type', 'N/A')}\n")
            f.write("\n\n")

    def _col_width(self, label, fmt_spec):
        if fmt_spec == "s":
            data_w = COL_WIDTH_RUNTIME
        elif fmt_spec == "d":
            data_w = COL_WIDTH_INT
        else:
            data_w = COL_WIDTH_FLOAT
        return max(len(label), data_w)

    def write_snapshot(self, stats):
        col_defs = [
            ("runtime",    "RUNTIME",    "s"),
            ("snap",       "SNAP",       "d"),
            ("z",          "REDSHIFT",   ".3f"),
            ("load",       "LOAD",       ".3f"),
            ("process",    "PROCESS",    ".3f"),
            ("reduction",  "REDUCTION",  ".3f"),
            ("bound",      "BOUND",      "d"),
            ("groups",     "GROUPS",     "d"),
            ("fragments",  "FRAGS",      "d"),
            ("unassigned", "UNASSIGNED", "d"),
            ("avg_conf",   "AVG_CONF",   ".3f"),
            ("avg_entropy","AVG_ENTROPY",".3f"),
            ("avg_cond",   "AVG_COND",   ".3f"),
            ("avg_retention","AVG_RET",  ".3f"),
        ]

        # Build columns with widths computed from label + format
        cols = []
        for key, label, fmt_spec in col_defs:
            w = self._col_width(label, fmt_spec)
            cols.append((key, label, w, fmt_spec))

        # Header
        header_parts = [f"{label:>{w}}" for _, label, w, _ in cols]
        sep = "   "
        header = sep.join(header_parts)
        total_width = len(header)

        # Data row
        cell_parts = []
        for key, _, w, fmt_spec in cols:
            raw = stats.get(key)
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                cell_parts.append(f"{'--':>{w}}")
            else:
                if isinstance(raw, float) and fmt_spec != "s":
                    raw = min(raw, 99999.999)
                fstr = f">{w}{fmt_spec}" if fmt_spec != "s" else f">{w}s"
                cell_parts.append(f"{raw:{fstr}}")
        row = sep.join(cell_parts)

        with open(self._log_path, "a") as f:
            if not hasattr(self, "_header_written") or not self._header_written:
                f.write(header + "\n")
                f.write("-" * total_width + "\n")
                self._header_written = True
            f.write(row + "\n")

    def write_summary(self):
        with open(self._log_path, "a") as f:
            f.write("\n--- Run complete ---\n")
