from __future__ import annotations

from typing import TYPE_CHECKING

import ase
import numpy as np

from Src.common_functions import *
from Src.common_functions import _CustomBaseCalculator
from Src.stage2 import HardSphereCalculator

if TYPE_CHECKING:
    from typing import Union
    from scipy.sparse import dok_matrix


def reposition_reactants(reactant: ase.Atoms,
                         reactant_molecules: list[list[int]],
                         product: ase.Atoms,
                         product_molecules: list[list[int]],
                         reactivity_matrix: dok_matrix,
                         bond_forming_force_constant: float = 1.0,
                         correlated_placement_force_constant: float = 1.0,
                         hard_sphere_force_constant: float = 50.0,
                         fmax: float = 1e-5,
                         max_iter: int = 1000,
                         non_convergence_limit: Union[float, None] = 0.001,
                         non_convergence_roof: Union[float, None] = 0.5,
                         trial_constants: tuple[Union[
                             None, float, tuple[float], tuple[float, float], tuple[float, float, float],
                             list[float], np.ndarray]] = (5.0, 5.0, 100.0)
                         ) -> None:
    calculators = [BondFormingCalculator(reactant_molecules, reactivity_matrix, bond_forming_force_constant),
                   CorrelatedPlacementCalculator(reactant_molecules, product_molecules, product, reactivity_matrix,
                                                 correlated_placement_force_constant),
                   HardSphereCalculator(reactant_molecules, hard_sphere_force_constant)]

    force_constants = [bond_forming_force_constant, correlated_placement_force_constant, hard_sphere_force_constant]

    for calculator, force_constant, trial_constants in zip(calculators, force_constants, trial_constants):
        reactant.calc = calculator

        coordinates = optimise_system(reactant, calculator, reactant_molecules, force_constant, fmax, max_iter,
                                      non_convergence_limit, non_convergence_roof, trial_constants)

        if coordinates is not None:
            reactant.set_positions(coordinates)


class BondFormingCalculator(_CustomBaseCalculator):
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

    def compute_forces(self, molecules: list[ase.Atoms]) -> np.ndarray:
        coordinates = self.atoms.get_positions()
        forces = np.zeros((len(molecules), 3), dtype=np.float64)

        for i, molecule in enumerate(self.molecules):
            geometric_centre = np.mean(coordinates[molecule], axis=0)
            translational_vector, rotational_vector = self.compute_vectors(coordinates, molecule, i, geometric_centre)

            for atom in molecule:
                diff = coordinates[atom] - geometric_centre
                forces[i, :] += self.force_constant * (np.cross(-rotational_vector, diff) + rotational_vector)

        return forces

    def compute_vectors(self,
                        coordinates: np.ndarray,
                        molecule: list[int],
                        molecule_index: int,
                        geometric_centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the translational and rotational vectors for molecule n.

        :param coordinates:
        :param molecule:
        :param molecule_index:
        :param geometric_centre:
        :return:
        """
        translational_vector, rotational_vector = np.zeros(3), np.zeros(3)
        n = 0.0

        for i, other_mol in enumerate(self.molecules):
            if i == molecule_index:
                continue

            # Get atoms that will form bonds and that belong to each of these molecules
            bond_forming_atoms = get_bond_forming_atoms(molecule, other_mol, True, self.reactivity_matrix,
                                                        return_both=True)
            bond_forming_atoms_in_molecule = bond_forming_atoms[0]
            bond_forming_atoms_in_other_mol = bond_forming_atoms[1]

            n += len(bond_forming_atoms_in_molecule) * len(bond_forming_atoms_in_other_mol)

            for a in coordinates[bond_forming_atoms_in_molecule]:
                for b in coordinates[bond_forming_atoms_in_other_mol]:
                    diff_ba = b - a
                    diff_ag = a - geometric_centre

                    translational_vector += abs(np.dot(diff_ba, diff_ag)) * diff_ba / np.linalg.norm(diff_ba)
                    rotational_vector += np.cross(diff_ba, diff_ag)

        n = 3 * len(molecule) * n

        return translational_vector / n, rotational_vector / n


class CorrelatedPlacementCalculator(_CustomBaseCalculator):
    def __init__(self,
                 reactant_molecules: list[list[int]],
                 product_molecules: list[list[int]],
                 product: ase.Atoms,
                 reactivity_matrix: dok_matrix,
                 force_constant: float = 1.0,
                 label=None,
                 atoms=None,
                 directory='.',
                 **kwargs):
        self.reactivity_matrix = reactivity_matrix
        self.product_molecules = product_molecules
        self.product_coordinates = product.get_positions()

        super().__init__(reactant_molecules, force_constant, label=label, atoms=atoms, directory=directory, **kwargs)

    def compute_forces(self, molecules: list[ase.Atoms]) -> np.ndarray:
        coordinates = self.atoms.get_positions()
        forces = np.zeros((len(molecules), 3), dtype=np.float64)

        for i, molecule in enumerate(self.molecules):
            geometric_centre = np.mean(coordinates[molecule], axis=0)
            translational_vector, rotational_vector = self.compute_vectors(coordinates, molecule, geometric_centre)

            for atom in molecule:
                diff = coordinates[atom] - geometric_centre
                forces[i, :] += self.force_constant * (np.cross(-rotational_vector, diff) + rotational_vector)

        return forces

    def compute_vectors(self,
                        reactant_coordinates: np.ndarray,
                        reactant_mol: list[int],
                        geometric_centre: np.ndarray):
        translational_vector, rotational_vector = np.zeros(3), np.zeros(3)
        n = 0.0

        for product_mol in self.product_molecules:
            shared_atoms = get_shared_atoms(reactant_mol, product_mol)
            reactive_atoms = get_reactive_atoms(shared_atoms, self.reactivity_matrix)

            n += len(shared_atoms) ** 2

            for a in shared_atoms:
                phi = 1. if a in reactive_atoms else 0.5

                for b in shared_atoms:
                    diff_ba = self.product_coordinates[b] - reactant_coordinates[a]
                    diff_ag = reactant_coordinates[a] - geometric_centre

                    translational_vector += phi * abs(np.dot(diff_ba, diff_ag)) * diff_ba / np.linalg.norm(diff_ba)
                    rotational_vector += phi * np.cross(diff_ba, diff_ag)

        n *= 3 * len(reactant_mol)

        return translational_vector / n, rotational_vector / n
