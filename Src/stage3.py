from __future__ import annotations

from typing import TYPE_CHECKING

import ase
import numpy as np
from scipy.spatial.transform import Rotation

from Src.common_functions import get_all_bond_forming_atoms_in_molecule, compute_alpha_vector

if TYPE_CHECKING:
    from typing import Union
    from scipy.sparse import dok_matrix


class ParallelVectorsError(Exception):
    pass


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

    # Try creating rotation matrix
    try:
        rotation_matrix = rotation_matrix_from_vectors(rotating_vector, target_vector)
        rotation = Rotation.from_matrix(rotation_matrix)
    except ParallelVectorsError:
        # If the vectors are opposites of each other, create Rotation object using inbuilt method, otherwise nothing
        rotation, _ = Rotation.align_vectors(rotating_vector[np.newaxis, :], target_vector[np.newaxis, :])

    return rotation


def rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    s = np.linalg.norm(v)

    if s == 0:
        raise ParallelVectorsError(f'Rotation matrix could not be computed for the provided vectors because they are'
                                   f'parallel: {vec1=}, {vec2=}.')

    c = np.dot(a, b)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def reorient_reactants(reactant: ase.Atoms, molecules: list[ase.Atoms], reactivity_matrix: dok_matrix):
    coordinates = reactant.get_positions()
    new_coordinates = np.copy(coordinates)

    for i, molecule in enumerate(molecules):
        rotation = compute_reactant_rotation(coordinates, i, molecules, reactivity_matrix)
        if rotation is not None:
            rotation.apply(new_coordinates[molecule.get_tags()])

    reactant.set_positions(new_coordinates)
