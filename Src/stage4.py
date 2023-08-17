from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import ase
import numpy as np

from Src.common_functions import *
from Src.common_functions import _CustomBaseCalculator
from Src.optimise import DualBFGS
from Src.stage2 import HardSphereCalculator

if TYPE_CHECKING:
    from typing import Union
    from scipy.sparse import dok_matrix


np.seterr(all='raise')


def reposition_reactants(reactant: ase.Atoms,
                         reactant_molecules: list[list[int]],
                         product: ase.Atoms,
                         product_molecules: list[list[int]],
                         reactivity_matrix: dok_matrix,
                         bond_forming_force_constant: float = 1.0,
                         correlated_placement_force_constant: float = 1.0,
                         hard_sphere_force_constant: float = 50.0,
                         fmax: float = 1e-5,
                         max_iter: int = 500,
                         non_convergence_limit: Union[float, None] = 0.001,
                         non_convergence_roof: Union[float, None] = 0.5,
                         trial_constants: tuple[Union[
                                                    None, float, tuple[float], tuple[float, float], tuple[
                                                        float, float, float],
                                                    list[float], np.ndarray], ...] = (5.0, 5.0, 100.0)
                         ) -> None:
    reactant.calc = TestCalculator(reactant_molecules, product_molecules, product, reactivity_matrix, calc_reactant=True)
    product.calc = TestCalculator(product_molecules, reactant_molecules, reactant, reactivity_matrix, calc_reactant=False)

    # dyn = DualBFGS(reactant, product, maxstep=0.05)
    # result = dyn.run(fmax=fmax, steps=max_iter)
    # print(result)

    for i in range(max_iter):
        reactant.calc.atoms = reactant
        product.calc.atoms = product

        reactant.calc.product = product
        product.calc.product = reactant

        reactant.calc.calculate(reactant)
        product.calc.calculate(product)
        reactant_forces = reactant.calc.results['forces']
        product_forces = product.calc.results['forces']

        rmax = np.max(np.sum(reactant_forces ** 2, axis=1))
        pmax = np.max(np.sum(product_forces ** 2, axis=1))
        max_force = np.sqrt(max([rmax, pmax]))

        logging.info(f'{i}   {rmax}    {pmax}    {max_force}')

        if max_force < fmax:
            break

        reactant_coordinates = reactant.get_positions()
        #for force, molecule in zip(reactant_forces, reactant_molecules):
        reactant_coordinates -= 0.05 * reactant_forces

        product_coordinates = product.get_positions()
        #for force, molecule in zip(product_forces, product_molecules):
        product_coordinates -= 0.05 * product_forces

        reactant.set_positions(reactant_coordinates)
        product.set_positions(product_coordinates)


class TestCalculator(_CustomBaseCalculator):
    def __init__(self,
                 reactant_molecules: list[list[int]],
                 product_molecules: list[list[int]],
                 product: ase.Atoms,
                 reactivity_matrix: dok_matrix,
                 bond_forming_force_constant: float = 1.0,
                 correlated_placement_force_constant: float = 1.0,
                 hard_sphere_force_constant: float = 50.0,
                 calc_reactant: bool = True,
                 label=None,
                 atoms=None,
                 directory='.',
                 **kwargs):
        self.product = product
        self.bfc = BondFormingCalculator(reactant_molecules, reactivity_matrix, bond_forming_force_constant, calc_reactant)
        self.cpc = CorrelatedPlacementCalculator(reactant_molecules, product_molecules, product, reactivity_matrix,
                                                 correlated_placement_force_constant, calc_reactant=calc_reactant)
        self.hsc = HardSphereCalculator(reactant_molecules, hard_sphere_force_constant)

        super().__init__(reactant_molecules, 0, label=label, atoms=atoms, directory=directory, **kwargs)

    def compute_forces(self) -> np.ndarray:
        self.bfc.atoms = self.atoms.copy()
        forces = self.bfc.compute_forces()

        self.cpc.atoms = self.atoms.copy()
        self.cpc.product_coordinates = self.product.get_positions()
        forces += self.cpc.compute_forces()

        self.hsc.atoms = self.atoms.copy()
        forces += self.hsc.compute_forces()

        return forces

    def calculate(self, atoms=None, properties=None, system_changes=None) -> None:
        self.bfc.atoms = self.atoms.copy()
        submit_forces = self.bfc.compute_forces()

        self.cpc.atoms = self.atoms.copy()
        self.cpc.product_coordinates = self.product.get_positions()
        submit_forces += self.cpc.compute_forces()

        self.hsc.atoms = self.atoms.copy()
        forces = self.hsc.compute_forces()

        for mol, force in zip(self.molecules, forces):
            for index in mol:
                submit_forces[index, :] += force

        shape = np.shape(submit_forces)

        projection = self.compute_projection()
        f = submit_forces.flatten()

        f = np.matmul(projection, f)
        submit_forces = f.reshape(shape)

        self.results['forces'] = submit_forces


class BondFormingCalculator(_CustomBaseCalculator):
    def __init__(self,
                 molecules: list[list[int]],
                 reactivity_matrix: dok_matrix,
                 force_constant: float = 1.0,
                 calc_reactant: bool = True,
                 label=None,
                 atoms=None,
                 directory='.',
                 **kwargs):
        self.reactivity_matrix = reactivity_matrix
        self.calc_reactant = calc_reactant

        super().__init__(molecules, force_constant, label=label, atoms=atoms, directory=directory, **kwargs)

    def compute_forces(self) -> np.ndarray:
        coordinates = self.atoms.get_positions()
        forces = np.zeros((len(self.atoms), 3), dtype=np.float64)

        for i, molecule in enumerate(self.molecules):
            geometric_centre = np.mean(coordinates[molecule], axis=0)
            translational_vector, rotational_vector = self.compute_vectors(coordinates, molecule, i, geometric_centre)

            for atom in molecule:
                diff = coordinates[atom] - geometric_centre
                forces[atom, :] += self.force_constant * (np.cross(rotational_vector, diff) - rotational_vector)

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
            bond_forming_atoms = get_bond_forming_atoms(molecule, other_mol, self.calc_reactant, self.reactivity_matrix,
                                                        return_both=True)
            bond_forming_atoms_in_molecule = bond_forming_atoms[0]
            bond_forming_atoms_in_other_mol = bond_forming_atoms[1]

            if bond_forming_atoms_in_molecule.size == 0 or bond_forming_atoms_in_other_mol.size == 0:
                continue

            n += len(bond_forming_atoms_in_molecule) * len(bond_forming_atoms_in_other_mol)

            for a in coordinates[bond_forming_atoms_in_molecule]:
                for b in coordinates[bond_forming_atoms_in_other_mol]:
                    diff_ba = b - a
                    diff_ag = a - geometric_centre

                    translational_vector += abs(np.dot(diff_ba, diff_ag)) * diff_ba / np.linalg.norm(diff_ba)
                    rotational_vector += np.cross(diff_ba, diff_ag)

        n = 3 * len(molecule) * n

        try:
            return translational_vector / n, rotational_vector / n
        except FloatingPointError:
            return translational_vector, rotational_vector


class CorrelatedPlacementCalculator(_CustomBaseCalculator):
    def __init__(self,
                 reactant_molecules: list[list[int]],
                 product_molecules: list[list[int]],
                 product: ase.Atoms,
                 reactivity_matrix: dok_matrix,
                 force_constant: float = 1.0,
                 calc_reactant = True,
                 label=None,
                 atoms=None,
                 directory='.',
                 **kwargs):
        self.reactivity_matrix = reactivity_matrix
        self.product_molecules = product_molecules
        self.product_coordinates = product.get_positions()
        self.calc_reactant = calc_reactant

        super().__init__(reactant_molecules, force_constant, label=label, atoms=atoms, directory=directory, **kwargs)

    def compute_forces(self) -> np.ndarray:
        coordinates = self.atoms.get_positions()
        forces = np.zeros((len(self.atoms), 3), dtype=np.float64)

        for i, molecule in enumerate(self.molecules):
            geometric_centre = np.mean(coordinates[molecule], axis=0)
            translational_vector, rotational_vector = self.compute_vectors(coordinates, molecule, geometric_centre)

            for atom in molecule:
                diff = coordinates[atom] - geometric_centre
                forces[atom, :] += self.force_constant * (np.cross(rotational_vector, diff) - rotational_vector)

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

        try:
            return translational_vector / n, rotational_vector / n
        except FloatingPointError:
            return translational_vector, rotational_vector
