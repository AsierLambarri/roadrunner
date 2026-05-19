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
                    text = "\n".join(lines[:, 0]) + "\n"
                    f.write(text)
                except Exception:
                    pass
            f.write(f"\n\n")
            f.write(f"-output_dir: {config.get('output_dir', 'N/A')}\n")
            f.write(f"-halo_model: {config.get('halo_model', 'N/A')}\n")
            f.write(f"-cov_type: {config.get('cov_type', 'N/A')}\n")
            f.write(f"\n\n")

    def write_snapshot(self, stats):
        with open(self._log_path, "a") as f:
            parts = [f"SNAP {stats.get('snap', '?'):>4d}"]
            for k in ("runtime", "load", "process", "bound", "groups"):
                v = stats.get(k)
                if v is not None:
                    parts.append(f"{k}={v}")
            f.write("  ".join(parts) + "\n")

    def write_summary(self):
        with open(self._log_path, "a") as f:
            f.write("\n--- Run complete ---\n")
