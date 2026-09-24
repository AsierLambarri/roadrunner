import numpy as np

from roadrunner.pipeline.accretion_pipeline import AccretionPipeline


def _resolver():
    """A bare AccretionPipeline instance, enough to call _resolve_snap_indices."""
    return AccretionPipeline.__new__(AccretionPipeline)


class TestResolveSnapIndicesFile:
    def test_single_entry_file(self, tmp_path):
        """A one-line file used to load as a 0-d array and crash on iteration."""
        path = tmp_path / "dynstate.txt"
        path.write_text("5\n")
        assert _resolver()._resolve_snap_indices(str(path), [1, 2, 5]) == {5}

    def test_multi_entry_file(self, tmp_path):
        path = tmp_path / "dynstate.txt"
        path.write_text("1\n3\n7\n")
        assert _resolver()._resolve_snap_indices(str(path), [1, 3, 7]) == {1, 3, 7}

    def test_none_means_all_snapshots(self):
        assert _resolver()._resolve_snap_indices(None, [1, 2, 3]) is None


class TestResolveSnapIndicesList:
    def test_negative_index(self):
        assert _resolver()._resolve_snap_indices([-1], [1, 2, 5]) == {5}

    def test_int_scalar(self):
        assert _resolver()._resolve_snap_indices(5, [1, 2, 5]) == {5}
