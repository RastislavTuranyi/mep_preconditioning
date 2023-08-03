from __future__ import annotations

from typing import TYPE_CHECKING

import ase
from ase.optimize import BFGS

import numpy as np

from Src.common_functions import separate_molecules, _CustomBaseCalculator

if TYPE_CHECKING:
    from typing import Union
    from scipy.sparse import dok_matrix


class BondFormingCalculator(_CustomBaseCalculator):
    def compute_forces(self, molecules: list[ase.Atoms]) -> np.ndarray:
        return np.zeros(0)
