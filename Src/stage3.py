from __future__ import annotations

from typing import TYPE_CHECKING

import ase
import numpy as np

from Src.common_functions import get_all_bond_forming_atoms_in_molecule

if TYPE_CHECKING:
    from typing import Union
    from scipy.sparse import dok_matrix


def rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def reorient_reactants(reactant: ase.Atoms, molecules: list[ase.Atoms], reactivity_matrix: dok_matrix):
    for molecule in molecules:
        g = np.mean(molecule.get_positions(), axis=0)

        bonding_atoms = get_all_bond_forming_atoms_in_molecule(molecule, True, reactivity_matrix)
        gamma = np.mean(reactant.get_positions()[bonding_atoms], axis=0)