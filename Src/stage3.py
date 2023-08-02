from __future__ import annotations

from typing import TYPE_CHECKING

import ase
import numpy as np
from scipy.spatial.transform import Rotation

from Src.common_functions import get_all_bond_forming_atoms_in_molecule, compute_alpha_vector

if TYPE_CHECKING:
    from typing import Union
    from scipy.sparse import dok_matrix


def compute_reactant_rotation(coordinates: np.ndarray,
                              molecule_index: int,
                              molecules: list[ase.Atoms],
                              reactivity_matrix: dok_matrix) -> Union[None, Rotation]:
    molecule = molecules[molecule_index]

    # Calculate gRm and alphaRm
    geometric_centre = np.mean(molecule.get_positions(), axis=0)
    alpha = compute_alpha_vector(coordinates, molecule_index, molecules, True, reactivity_matrix)

    # Calculate gammaRm
    bonding_atoms = get_all_bond_forming_atoms_in_molecule(molecule, True, reactivity_matrix)
    if bonding_atoms.size > 0:
        gamma = np.mean(coordinates[bonding_atoms], axis=0)
    else:
        gamma = alpha

    # The rotation matrix will act on gamma - g to minimize the difference with alpha - g
    rotating_vector = gamma - geometric_centre
    target_vector = alpha - geometric_centre

    rotation, _ = Rotation.align_vectors(target_vector[np.newaxis, :], rotating_vector[np.newaxis, :])

    return rotation


def reorient_reactants(reactant: ase.Atoms,
                       molecules: list[ase.Atoms],
                       reactivity_matrix: dok_matrix) -> None:
    coordinates = reactant.get_positions()
    new_coordinates = np.copy(coordinates)

    for i, molecule in enumerate(molecules):
        rotation = compute_reactant_rotation(coordinates, i, molecules, reactivity_matrix)
        rotation.apply(new_coordinates[molecule.get_tags()])

    reactant.set_positions(new_coordinates)
