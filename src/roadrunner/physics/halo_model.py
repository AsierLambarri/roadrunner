import numpy as np

from roadrunner.physics.constants import G_KM
from roadrunner.physics.potentials import get_potential
from roadrunner._mcf_types import PotentialModel


class HaloModel:
    def __init__(
        self,
        inner: PotentialModel,
        xcen: np.ndarray,
        sub_tree_id: int,
        redshift: float,
        comoving: bool = True,
    ):
        self._inner = inner
        self.xcen = np.asarray(xcen, dtype=np.float64)
        self.sub_tree_id = sub_tree_id
        self.redshift = float(redshift)
        self.comoving = comoving
        self._1plusz  = 1 / (1 + self.redshift) if comoving else 1
        self._boundness: tuple | None = None

    def set_boundness(
        self, indices: np.ndarray, energies: np.ndarray, tdyns: np.ndarray
    ) -> None:
        self._boundness = (indices, energies, tdyns)

    def get_boundness(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        return self._boundness

    @property
    def has_boundness(self) -> bool:
        return self._boundness is not None

    def potential(self, xyz_or_r: np.ndarray) -> np.ndarray:
        if xyz_or_r.ndim == 2:
            r = np.linalg.norm(xyz_or_r - self.xcen, axis=1) * self._1plusz
        else:
            r = xyz_or_r * self._1plusz
        return self._inner.potential(r)

    def dynamical_time(self, xyz_or_r: np.ndarray) -> np.ndarray:
        if xyz_or_r.ndim == 2:
            r = np.linalg.norm(xyz_or_r - self.xcen, axis=1) * self._1plusz
        else:
            r = xyz_or_r * self._1plusz
        return self._inner.dynamical_time(r)

    def tidal_denominator(self, xyz_or_r: np.ndarray) -> np.ndarray:
        if xyz_or_r.ndim == 2:
            r = np.linalg.norm(xyz_or_r - self.xcen, axis=1) * self._1plusz
        else:
            r = xyz_or_r * self._1plusz
        return self._inner.tidal_denominator(r)

    @classmethod
    def from_snapshot_row(cls, row, model="kepler", comoving=True):
        xcen = np.array([
            row["position_x"], row["position_y"], row["position_z"],
        ], dtype=np.float64)
        conc = row["virial_radius"] / row["scale_radius"]
        kwargs = {"M": row["mass"], "G": G_KM}
        if model.lower() == "nfw":
            kwargs["Rs"] = row["scale_radius"] / (1 + row["Redshift"])
            kwargs["c"] = conc
        inner = get_potential(model, **kwargs)
        return cls(
            inner=inner,
            xcen=xcen,
            sub_tree_id=int(row["Sub_tree_id"]),
            redshift=float(row["Redshift"]),
            comoving=comoving,
        )
