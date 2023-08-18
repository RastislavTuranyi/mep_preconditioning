from __future__ import annotations

from itertools import combinations
import logging
from typing import Union

import ase
import numpy as np
from scipy.sparse import dok_matrix
from scipy.spatial.transform import Rotation

from Src.common_functions import _CustomBaseCalculator, optimise_system, estimate_molecular_radius, \
    get_bond_forming_atoms, construct_molecular_reactivity_matrix


def determine_overlaps(size: int,
                       geometric_centres: list[np.ndarray],
                       estimated_radii: list[float]) -> np.ndarray:
    overlaps = np.zeros((size, size), dtype=np.int8)
    indices = list(range(size))

    for mol1, mol2 in combinations(indices, 2):
        centre_distance = np.linalg.norm(geometric_centres[mol2] - geometric_centres[mol1])
        combined_radius = estimated_radii[mol2] + estimated_radii[mol1]

        if centre_distance < 1.5 * combined_radius:
            overlaps[mol1, mol2], overlaps[mol2, mol1] = 1, 1
        # Else keep it 0

    return overlaps


def fix_overlaps(system: ase.Atoms,
                 molecules: list[list[int]],
                 reactivity_matrix: dok_matrix,
                 force_constant: float = 1.0,
                 fmax: float = 1e-5,
                 max_iter: int = 1000,
                 non_convergence_limit: Union[float, None] = 0.001,
                 non_convergence_roof: Union[float, None] = 0.5,
                 trial_constants: Union[None, float, tuple[float], tuple[float, float], tuple[float, float, float],
                                        list[float], np.ndarray] = 10.0):
    system.calc = HardSphereCalculator(molecules, reactivity_matrix, force_constant, atoms=system)

    coordinates = optimise_system(system, system.calc, molecules, force_constant, fmax, max_iter, non_convergence_limit,
                                  non_convergence_roof, trial_constants)

    if coordinates is not None:
        system.set_positions(coordinates)
    else:
        coordinates = system.get_positions()

    geometric_centres = [np.mean(coordinates[mol], axis=0) for mol in molecules]
    molecular_radii = [estimate_molecular_radius(coordinates[mol], centre) for mol, centre in
                       zip(molecules, geometric_centres)]
    logging.debug(f'{determine_overlaps(len(molecules), geometric_centres, molecular_radii)}')


def overlay_non_reacting_molecules(static_system: ase.Atoms,
                                   moving_system: ase.Atoms,
                                   moving_molecules: list[list[int]],
                                   reactivity_matrix: dok_matrix,
                                   max_iter: int = 100):
    static_coordinates = static_system.get_positions()
    moving_coordinates = moving_system.get_positions()

    molecular_reactivity_matrix = construct_molecular_reactivity_matrix(moving_molecules, reactivity_matrix)

    non_reacting_molecules = []
    for molecule, reactivity in zip(moving_molecules, molecular_reactivity_matrix):
        if np.sum(reactivity) == 0:
            non_reacting_molecules.append(molecule)

    logging.debug(f'{non_reacting_molecules=}')

    for non_reacting_molecule in non_reacting_molecules:
        static_geometric_centre = np.mean(static_coordinates[non_reacting_molecule, :], axis=0)

        previous_centre = np.zeros(3)
        for i in range(max_iter):
            new_centre = np.mean(moving_coordinates[non_reacting_molecule, :], axis=0)
            if np.allclose(previous_centre, new_centre):
                logging.debug(f'{previous_centre=}    {new_centre=}')
                break

            moving_coordinates[non_reacting_molecule, :] += static_geometric_centre - new_centre

            # Find Kabsch rotation and rotate the molecule accordingly
            rotation, rssd = Rotation.align_vectors(static_coordinates[non_reacting_molecule],
                                                    moving_coordinates[non_reacting_molecule])
            moving_coordinates[non_reacting_molecule, :] = rotation.apply(moving_coordinates[non_reacting_molecule, :])

            previous_centre = new_centre

    moving_system.set_positions(moving_coordinates)


class HardSphereCalculator(_CustomBaseCalculator):
    def __init__(self,
                 molecules: list[list[int]],
                 reactivity_matrix: dok_matrix,
                 force_constant: float = 1.0,
                 label=None,
                 atoms=None,
                 directory='.',
                 **kwargs):
        self.reactivity_matrix = reactivity_matrix

        super().__init__(molecules, force_constant, label=label, atoms=atoms, directory=directory, **kwargs)

    def compute_forces(self) -> np.ndarray:
        coordinates = self.atoms.get_positions()

        geometric_centres = [np.mean(coordinates[mol], axis=0) for mol in self.molecules]
        molecular_radii = [estimate_molecular_radius(coordinates[mol], centre) for mol, centre in
                           zip(self.molecules, geometric_centres)]

        n_mol = len(self.molecules)
        overlaps = determine_overlaps(n_mol, geometric_centres, molecular_radii)

        forces = np.zeros((n_mol, 3), dtype=np.float64)
        for i, affected_mol in enumerate(self.molecules):
            n_atoms = len(affected_mol)
            n = 3 * n_atoms * np.sum(overlaps[i, :])

            pairwise_forces = []
            for j, other_mol in enumerate(self.molecules):
                if overlaps[i, j] == 0 or i == j:
                    continue

                shared_atoms_affected, shared_atoms_other = get_bond_forming_atoms(affected_mol, other_mol, True,
                                                                                   self.reactivity_matrix, True,
                                                                                   [1, -1])

                centre_diff = np.mean(coordinates[other_mol], axis=0) - np.mean(coordinates[affected_mol], axis=0)
                distance = np.linalg.norm(centre_diff)
                try:
                    translation_vector = np.mean(coordinates[shared_atoms_affected], axis=0) - \
                                         np.mean(coordinates[shared_atoms_other], axis=0)
                    multiplier = 1.
                except IndexError:
                    multiplier = 50.
                    translation_vector = centre_diff

                length = np.linalg.norm(translation_vector)
                phi = self.force_constant * (distance - 1.5 * (molecular_radii[i] + molecular_radii[j])) / n
                pairwise_forces.append(phi * multiplier * translation_vector / length)

            forces[i, :] = n_atoms * np.sum(np.array(pairwise_forces), axis=0)

        return - forces
