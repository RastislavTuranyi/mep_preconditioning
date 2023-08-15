from __future__ import annotations

from itertools import combinations
import logging
from typing import Union

import ase
import numpy as np

from Src.common_functions import _CustomBaseCalculator, optimise_system, estimate_molecular_radius


def determine_overlaps(size: int,
                       geometric_centres: list[np.ndarray],
                       estimated_radii: list[float]) -> np.ndarray:
    overlaps = np.zeros((size, size), dtype=np.int8)
    indices = list(range(size))

    for mol1, mol2 in combinations(indices, 2):
        centre_distance = np.linalg.norm(geometric_centres[mol2] - geometric_centres[mol1])
        combined_radius = estimated_radii[mol2] + estimated_radii[mol1]

        if centre_distance < combined_radius:
            overlaps[mol1, mol2], overlaps[mol2, mol1] = 1, 1
        # Else keep it 0

    return overlaps


def fix_overlaps(system: ase.Atoms,
                 molecules: list[list[int]],
                 force_constant: float = 1.0,
                 fmax: float = 1e-5,
                 max_iter: int = 1000,
                 non_convergence_limit: Union[float, None] = 0.001,
                 non_convergence_roof: Union[float, None] = 0.5,
                 trial_constants: Union[None, float, tuple[float], tuple[float, float], tuple[float, float, float],
                                        list[float], np.ndarray] = 10.0):
    system.calc = HardSphereCalculator(molecules, force_constant, atoms=system)

    coordinates = optimise_system(system, system.calc, molecules, force_constant, fmax, max_iter, non_convergence_limit,
                                  non_convergence_roof, trial_constants)

    if coordinates is not None:
        system.set_positions(coordinates)


class HardSphereCalculator(_CustomBaseCalculator):
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
            for j, _ in enumerate(self.molecules):
                if overlaps[i, j] == 0 or i == j:
                    continue

                centre_diff = geometric_centres[i] - geometric_centres[j]
                distance = np.linalg.norm(centre_diff)
                phi = self.force_constant * (distance - (molecular_radii[i] + molecular_radii[j])) / n
                pairwise_forces.append(phi * centre_diff / distance)

            forces[i, :] = n_atoms * np.sum(np.array(pairwise_forces), axis=0)

        return - forces
