from __future__ import annotations

from itertools import product
import logging
from typing import TYPE_CHECKING

import ase
from ase.data import covalent_radii
import numpy as np

from Src.common_functions import *
from Src.common_functions import _CustomBaseCalculator
from Src.stage4 import BondFormingCalculator, CorrelatedPlacementCalculator

if TYPE_CHECKING:
    from typing import Union
    from numpy.typing import ArrayLike
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
                         max_iter: int = 500,
                         non_convergence_limit: Union[float, None] = 0.001,
                         non_convergence_roof: Union[float, None] = 0.5,
                         trial_constants: tuple[Union[
                                                    None, float, tuple[float], tuple[float, float], tuple[
                                                        float, float, float],
                                                    list[float], np.ndarray], ...] = (5.0, 5.0, 100.0)
                         ) -> None:
    reactant.calc = TestCalculator5(reactant_molecules, product_molecules, product, reactivity_matrix,
                                    calc_reactant=True)
    product.calc = TestCalculator5(product_molecules, reactant_molecules, reactant, reactivity_matrix,
                                   calc_reactant=False)

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


class TestCalculator5(_CustomBaseCalculator):
    def __init__(self,
                 reactant_molecules: list[list[int]],
                 product_molecules: list[list[int]],
                 product: ase.Atoms,
                 reactivity_matrix: dok_matrix,
                 bond_forming_force_constant: float = 1.0,
                 correlated_placement_force_constant: float = 3.0,
                 hard_sphere_force_constant: float = 50.0,
                 atom_atom_hard_sphere_force_constant: float = 2.0,
                 calc_reactant: bool = True,
                 label=None,
                 atoms=None,
                 directory='.',
                 **kwargs):
        self.product = product
        self.bfc = BondFormingCalculator(reactant_molecules, reactivity_matrix, bond_forming_force_constant, calc_reactant)
        self.cpc = CorrelatedPlacementCalculator(reactant_molecules, product_molecules, product, reactivity_matrix,
                                                 correlated_placement_force_constant, calc_reactant=calc_reactant)
        self.hsc = HardSphereCalculator(reactant_molecules, product_molecules, product,reactivity_matrix,
                                        hard_sphere_force_constant)
        self.aahsc = AtomAtomHardSphereCalculator(reactant_molecules, calc_reactant, reactivity_matrix,
                                                  atom_atom_hard_sphere_force_constant)

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
        self.hsc.product_coordinates = self.product.get_positions()
        forces = self.hsc.compute_forces()

        self.aahsc.atoms = self.atoms.copy()
        submit_forces += self.aahsc.compute_forces()

        for mol, force in zip(self.molecules, forces):
            for index in mol:
                submit_forces[index, :] += force

        shape = np.shape(submit_forces)

        projection = self.compute_projection()
        f = submit_forces.flatten()

        f = np.matmul(projection, f)
        submit_forces = f.reshape(shape)

        self.results['forces'] = submit_forces


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

    def compute_forces(self) -> np.ndarray:
        coordinates = self.atoms.get_positions()
        forces = np.zeros((len(self.atoms), 3), dtype=np.float64)

        for i, molecule in enumerate(self.molecules):
            geometric_centre = np.mean(coordinates[molecule], axis=0)
            translational_vector, rotational_vector = self.compute_vectors(coordinates, i, molecule, geometric_centre)

            for atom in molecule:
                diff = coordinates[atom] - geometric_centre
                forces[atom, :] += self.force_constant * (np.cross(rotational_vector, diff) - translational_vector)

        return forces

    def compute_vectors(self, coordinates, index, mol1: list[int], geometric_centre):
        translational_vector, rotational_vector = np.zeros(3), np.zeros(3)
        n = 0.0
        atomic_numbers = self.atoms.get_atomic_numbers()

        for i, mol2 in enumerate(self.molecules):
            if i == index:
                continue

            bond_forming_atoms = get_bond_forming_atoms(mol1, mol2, self.reactants, self.reactivity_matrix)
            for a in mol1:
                phi = 1.5 if a in bond_forming_atoms else 2
                for b in mol2:
                    threshold = np.mean([covalent_radii[atomic_numbers[a]], covalent_radii[atomic_numbers[b]]])

                    diff_ba = coordinates[b] - coordinates[a]
                    diff_ag = coordinates[a] - geometric_centre
                    y = threshold * diff_ba / np.linalg.norm(diff_ba) - diff_ba

                    h = 1 if np.linalg.norm(diff_ba) < phi * threshold else 0
                    n += h

                    translational_vector += h * abs(np.dot(y, diff_ag)) * y / np.linalg.norm(y)
                    rotational_vector += h * np.cross(y, diff_ag)

        n *= len(mol1) ** 2
        try:
            return translational_vector / n, rotational_vector / n
        except FloatingPointError:
            return translational_vector, rotational_vector


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

    def compute_forces(self) -> np.ndarray:
        coordinates = self.atoms.get_positions()

        reactant_geometric_centres = [np.mean(coordinates[mol], axis=0) for mol in self.molecules]
        product_geometric_centres = [np.mean(self.product_coordinates[mol], axis=0) for mol in self.product_molecules]

        reactant_molecular_radii = [estimate_molecular_radius(coordinates[mol], centre) for mol, centre in
                                    zip(self.molecules, reactant_geometric_centres)]
        product_molecular_radii = [estimate_molecular_radius(self.product_coordinates[mol], centre) for mol, centre in
                                   zip(self.product_molecules, product_geometric_centres)]

        overlaps = self.determine_overlaps(reactant_geometric_centres, product_geometric_centres,
                                           reactant_molecular_radii, product_molecular_radii)

        forces = np.zeros((len(self.molecules), 3), dtype=np.float64)
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

        return - forces

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
                overlaps[mol1, mol2] = 1
            # Else keep it 0

        return overlaps
