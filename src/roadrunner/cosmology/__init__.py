#############################################################################
#
# package:   roadrunner.cosmology
# file:      __init__.py
# brief:     Flat LCDM cosmology and run-time Ishiyama+21 c_vir(M, z).
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   10 Oct 2026 - Created
#
#############################################################################

"""Flat LCDM cosmology and run-time Ishiyama et al. (2021) ``c_vir(M, z)``.

:class:`Cosmology` bundles a flat-LCDM parameter set with a lazily-built
Ishiyama et al. (2021) concentration table; :func:`cosmology` scopes one
instance as the pipeline's active cosmology for a snapshot, and
:func:`current_cosmology` reads it back from anywhere inside that scope
(see :func:`roadrunner.physics.merger_tree.concentration`).
"""

from .model import Cosmology, cosmology, current_cosmology

__all__ = [
    "Cosmology",
    "cosmology",
    "current_cosmology",
]
