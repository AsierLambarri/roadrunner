import numpy as np


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

    def write_snapshot(self, stats):
        # Column definitions: (key, label, width, format_str)
        cols = [
            ("runtime", "RUNTIME", 10, ">10s"),
            ("snap", "SNAP", 6, ">6d"),
            ("z", "REDSHIFT", 10, ">10.3f"),
            ("load", "LOAD", 10, ">10.3f"),
            ("process", "PROCESS", 10, ">10.3f"),
            ("reduction", "REDUCTION", 10, ">10.3f"),
            ("bound", "BOUND", 10, ">10d"),
            ("groups", "GROUPS", 8, ">8d"),
            ("fragments", "FRAGS", 7, ">7d"),
            ("unassigned", "UNASSIGNED", 10, ">10d"),
            ("avg_conf", "AVG_CONF", 10, ">10.4f"),
            ("avg_entropy", "AVG_ENTROPY", 12, ">12.4f"),
            ("avg_cond", "AVG_COND", 10, ">10.2f"),
        ]

        # Build header
        header = "  ".join(label for _, label, _, _ in cols)
        total_width = len(header)

        # Build data row
        cells = []
        for key, _, width, fmt in cols:
            raw = stats.get(key)
            if raw is None:
                cells.append(f"{'--':>{width}}")
            elif key in ("avg_conf", "avg_entropy") and (raw is None or (isinstance(raw, float) and np.isnan(raw))):
                cells.append(f"{'--':>{width}}")
            else:
                cells.append(f"{raw:{fmt}}")

        row = "  ".join(cells)

        with open(self._log_path, "a") as f:
            if not hasattr(self, "_header_written") or not self._header_written:
                f.write(header + "\n")
                f.write("-" * total_width + "\n")
                self._header_written = True
            f.write(row + "\n")

    def write_summary(self):
        with open(self._log_path, "a") as f:
            f.write("\n--- Run complete ---\n")
