from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import ase

import numpy as np

from Src.common_functions import separate_molecules, _CustomBaseCalculator, optimise_system

if TYPE_CHECKING:
    from typing import Union
    from numpy.typing import ArrayLike
    from scipy.sparse import dok_matrix


def determine_overlaps(molecules: list[ase.Atoms],
                       geometric_centres: list[np.ndarray],
                       estimated_radii: list[float]) -> np.ndarray:
    size = len(molecules)
    overlaps = np.zeros((size, size), dtype=np.int8)
    indices = list(range(size))

    for mol1, mol2 in combinations(indices, 2):
        centre_distance = np.linalg.norm(geometric_centres[mol2] - geometric_centres[mol1])
        combined_radius = estimated_radii[mol2] + estimated_radii[mol1]

        if centre_distance < combined_radius:
            overlaps[mol1, mol2], overlaps[mol2, mol1] = 1, 1
        # Else keep it 0

    return overlaps


def estimate_molecular_radius(molecule: ase.Atoms, geometric_centre: np.ndarray) -> float:
    distances = np.zeros(len(molecule))
    for i, atom in enumerate(molecule.get_positions()):
        distances[i] = np.linalg.norm(atom - geometric_centre)

    mean = np.mean(distances)
    std = np.std(distances)

    return mean + 2 * std


def fix_overlaps(system: ase.Atoms,
                 molecules: list[list[int]],
                 force_constant: float = 1.0,
                 fmax: float = 1e-5,
                 max_iter: int = 1000,
                 non_convergence_limit: Union[float, None] = 0.001,
                 non_convergence_roof: Union[float, None] = 0.5,
                 trial_constants: Union[None, float, tuple[float], tuple[float, float], tuple[float, float, float],
                                        list[float], np.ndarray] = 10.0):
    system.calc = HardSphereCalculator(molecules, force_constant)

    coordinates = optimise_system(system, system.calc, molecules, force_constant, fmax, max_iter, non_convergence_limit,
                                  non_convergence_roof, trial_constants)

    if coordinates is not None:
        system.set_positions(coordinates)


class HardSphereCalculator(_CustomBaseCalculator):
    def compute_forces(self, molecules: list[ase.Atoms]) -> np.ndarray:
        geometric_centres = [np.mean(mol.get_positions(), axis=0) for mol in molecules]
        molecular_radii = [estimate_molecular_radius(mol, centre) for mol, centre in zip(molecules, geometric_centres)]

        overlaps = determine_overlaps(molecules, geometric_centres, molecular_radii)

        forces = np.zeros((len(molecules), 3), dtype=np.float64)
        for i, affected_mol in enumerate(molecules):
            n_atoms = len(affected_mol)
            n = 3 * n_atoms * np.sum(overlaps[i, :])

            pairwise_forces = []
            for j, other_mol in enumerate(molecules):
                if overlaps[i, j] == 0 or i == j:
                    continue

                centre_diff = geometric_centres[i] - geometric_centres[j]
                distance = np.linalg.norm(centre_diff)
                phi = self.force_constant * (distance - (molecular_radii[i] + molecular_radii[j])) / n
                pairwise_forces.append(phi * centre_diff / distance)

            forces[i, :] = n_atoms * np.sum(np.array(pairwise_forces), axis=0)

        return forces
