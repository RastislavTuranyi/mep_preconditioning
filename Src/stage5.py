from __future__ import annotations

from itertools import product
import logging
from typing import TYPE_CHECKING

import ase
from ase.data import covalent_radii
import numpy as np

from Src.common_functions import *
from Src.common_functions import _CustomBaseCalculator

if TYPE_CHECKING:
    from typing import Union
    from numpy.typing import ArrayLike
    from scipy.sparse import dok_matrix


class AtomAtomHardSphereCalculator(_CustomBaseCalculator):
    def __init__(self,
                 molecules: list[list[int]],
                 reactants: bool,
                 reactivity_matrix: dok_matrix,
                 force_constant: float = 2.0,
                 label=None,
                 atoms=None,
                 directory='.',
                 **kwargs):
        self.reactivity_matrix = reactivity_matrix
        self.reactants = reactants

        super().__init__(molecules, force_constant, label=label, atoms=atoms, directory=directory, **kwargs)

    def compute_forces(self, molecules: list[ase.Atoms]) -> np.ndarray:
        coordinates = self.atoms.get_positions()
        forces = np.zeros((len(molecules), 3), dtype=np.float64)

        for i, molecule in enumerate(self.molecules):
            geometric_centre = np.mean(coordinates[molecule], axis=0)
            translational_vector, rotational_vector = self.compute_vectors(coordinates, i, molecules, geometric_centre)

            for atom in molecule:
                diff = coordinates[atom] - geometric_centre
                forces[i, :] += self.force_constant * (np.cross(-rotational_vector, diff) + rotational_vector)

        return forces

    def compute_vectors(self, coordinates, index, molecules: list[ase.Atoms], geometric_centre):
        translational_vector, rotational_vector = np.zeros(3), np.zeros(3)
        n = 0.0
        mol1 = molecules[index]

        for i, mol2 in enumerate(molecules):
            if i == index:
                continue

            bond_forming_atoms = get_bond_forming_atoms(mol1, mol2, self.reactants, self.reactivity_matrix)
            for a in mol1:
                phi = 1.5 if a in bond_forming_atoms else 2
                for b in mol2:
                    threshold = np.mean([covalent_radii[mol1.get_atomic_numbers()[a]],
                                         covalent_radii[mol2.get_atomic_numbers()[b]]])

                    diff_ba = coordinates[b] - coordinates[a]
                    diff_ag = coordinates[a] - geometric_centre
                    y = threshold * diff_ba / np.linalg.norm(diff_ba) - diff_ba

                    h = 1 if np.linalg.norm(diff_ba) < phi * threshold else 0
                    n += h

                    translational_vector += h * abs(np.dot(y, diff_ag)) * y / np.linalg.norm(y)
                    rotational_vector += h * np.cross(y, diff_ag)

        n *= len(mol1) ** 2
        return translational_vector / n, rotational_vector / n


class HardSphereCalculator(_CustomBaseCalculator):
    def __init__(self,
                 reactant_molecules: list[list[int]],
                 product_molecules: list[list[int]],
                 product: ase.Atoms,
                 reactivity_matrix: dok_matrix,
                 force_constant: float = 50.0,
                 label=None,
                 atoms=None,
                 directory='.',
                 **kwargs):
        self.reactivity_matrix = reactivity_matrix
        self.product_molecules = product_molecules
        self.product_coordinates = product.get_positions()

        super().__init__(reactant_molecules, force_constant, label=label, atoms=atoms, directory=directory, **kwargs)

    def compute_forces(self, molecules: list[ase.Atoms]) -> np.ndarray:
        reactant_geometric_centres = [np.mean(mol.get_positions(), axis=0) for mol in molecules]
        product_geometric_centres = [np.mean(self.product_coordinates[mol], axis=0) for mol in self.product_molecules]

        reactant_molecular_radii = [estimate_molecular_radius(mol, centre) for mol, centre in
                                    zip(molecules, reactant_geometric_centres)]
        product_molecular_radii = [estimate_molecular_radius(self.product_coordinates[mol], centre) for mol, centre in
                                    zip(self.product_molecules, product_geometric_centres)]

        overlaps = self.determine_overlaps(reactant_geometric_centres, product_geometric_centres,
                                           reactant_molecular_radii, product_molecular_radii)

        forces = np.zeros((len(molecules), 3), dtype=np.float64)
        for i, reactant_mol in enumerate(self.molecules):
            n_atoms = len(reactant_mol)
            n = 3 * n_atoms * np.sum(overlaps[i, :])

            pairwise_forces = []
            for j, product_mol in enumerate(self.product_molecules):
                if overlaps[i, j] == 0 or i == j:
                    continue

                shared = get_shared_atoms(reactant_mol, product_mol)
                if shared.size > 0:
                    continue

                centre_diff = product_geometric_centres[i] - reactant_geometric_centres[j]
                distance = np.linalg.norm(centre_diff)
                phi = self.force_constant * (distance - (reactant_molecular_radii[i] + product_molecular_radii[j])) / n
                pairwise_forces.append(phi * centre_diff / distance)

            forces[i, :] = n_atoms * np.sum(np.array(pairwise_forces), axis=0)

        return forces

    @staticmethod
    def determine_overlaps(reactant_geometric_centres: list[np.ndarray],
                           product_geometric_centres: list[np.ndarray],
                           reactant_estimated_radii: list[float],
                           product_estimated_radii: list[float]) -> np.ndarray:
        reactant_size = len(reactant_geometric_centres)
        product_size = len(product_geometric_centres)
        overlaps = np.zeros((reactant_size, product_size), dtype=np.int8)

        reactant_indices = list(range(reactant_size))
        product_indices = list(range(product_size))

        for mol1, mol2 in product(reactant_indices, product_indices):
            centre_distance = np.linalg.norm(reactant_geometric_centres[mol1] - product_geometric_centres[mol2])
            combined_radius = reactant_estimated_radii[mol1] + product_estimated_radii[mol2]

            if centre_distance < combined_radius:
                overlaps[mol1, mol2], overlaps[mol2, mol1] = 1, 1
            # Else keep it 0

        return overlaps
