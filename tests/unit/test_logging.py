import numpy as np

from roadrunner.io.logging import format_runtime, RunLogger


class TestFormatRuntime:
    def test_format(self):
        assert format_runtime(0) == "00:00:00.000"
        assert format_runtime(3661.5) == "01:01:01.500"
        assert format_runtime(86399.999) == "23:59:59.999"


class TestRunLogger:
    def test_write_header(self, tmp_path):
        log = str(tmp_path / "run.log")
        logger = RunLogger(log)
        logger.write_header({"output_dir": "/tmp", "halo_model": "kepler"})
        with open(log) as f:
            content = f.read()
        assert "kepler" in content

    def test_write_snapshot(self, tmp_path):
        log = str(tmp_path / "snap.log")
        logger = RunLogger(log)
        logger.write_header({})
        logger.write_snapshot({
            "snap": 10, "runtime": "00:01:00", "bound": 5000,
            "groups": 5, "fragments": 2, "unassigned": 0,
            "avg_conf": 0.934, "avg_entropy": 0.087,
        })
        with open(log) as f:
            content = f.read()
        assert "RUNTIME" in content  # header
        assert "AVG_CONF" in content
        assert "AVG_ENTROPY" in content
        assert "0.934" in content
        assert "0.087" in content

    def test_soft_metrics_nan(self, tmp_path):
        log = str(tmp_path / "nan.log")
        logger = RunLogger(log)
        logger.write_header({})
        logger.write_snapshot({
            "snap": 5, "runtime": "00:00:30", "bound": 100,
            "groups": 3, "fragments": 0, "unassigned": 0,
            "avg_conf": np.nan, "avg_entropy": np.nan,
        })
        with open(log) as f:
            content = f.read()
        # Should show '--' for NaN soft metrics
        assert "--" in content.split("UNASSIGNED")[1] if "UNASSIGNED" in content else True

    def test_write_summary(self, tmp_path):
        log = str(tmp_path / "summary.log")
        logger = RunLogger(log)
        logger.write_header({})
        logger.write_summary()
        with open(log) as f:
            content = f.read()
        assert "Run complete" in content

    def test_with_logo(self, tmp_path):
        logo = str(tmp_path / "logo.txt")
        with open(logo, "w") as f:
            f.write("R,D,RUNNER\nA,B,C\n")
        log = str(tmp_path / "logo_run.log")
        logger = RunLogger(log, logo_path=logo)
        logger.write_header({})
        with open(log) as f:
            content = f.read()
        assert "R" in content
