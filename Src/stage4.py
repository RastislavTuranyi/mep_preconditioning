from __future__ import annotations

from typing import TYPE_CHECKING

import ase
from ase.optimize import BFGS

import numpy as np

from Src.common_functions import separate_molecules, _CustomBaseCalculator, get_bond_forming_atoms

if TYPE_CHECKING:
    from typing import Union
    from scipy.sparse import dok_matrix


def reposition_reactants(reactant: ase.Atoms,
                         molecules: list[list[int]],
                         reactivity_matrix: dok_matrix,
                         force_constant: float,
                         fmax: float = 1e-5,
                         max_iter: int = 1000,
                         non_convergence_limit: Union[float, None] = 0.001,
                         non_convergence_roof: Union[float, None] = 0.5,
                         trial_constants: Union[
                             None, float, tuple[float], tuple[float, float], tuple[float, float, float],
                             list[float], np.ndarray] = 10.0
                         ) -> None:
    pass


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

        super().__init__(molecules, force_constant, restart=None, label=label, atoms=atoms, directory=directory,
                         **kwargs)

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
