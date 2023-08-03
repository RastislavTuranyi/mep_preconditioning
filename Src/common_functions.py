from __future__ import annotations

import abc
from abc import ABC
from typing import TYPE_CHECKING

import ase
from ase.calculators.calculator import Calculator
from ase.build import connected_indices
from ase.geometry.analysis import Analysis
import numpy as np

if TYPE_CHECKING:
    from typing import Union
    from numpy.typing import ArrayLike
    from scipy.sparse import dok_matrix


def compute_alpha_vector(coordinates: np.ndarray,
                         target: int,
                         molecules: list[ase.Atoms],
                         reactant: bool,
                         reactivity_matrix: dok_matrix) -> np.ndarray:
    r"""
    .. math::
        \alpha_{Rm} = \frac{1}{N^R} \sum_{n≠m}^{N^R}{\frac{1}{|A^{Rnm}|}\sum_{a∈A^{Rnm}}{\xrightarrow{r_{Rn}^a}}}
    :param coordinates:
    :param target:
    :param molecules:
    :param reactant:
    :param reactivity_matrix:
    :return:
    """
    alpha = np.zeros(3)
    target_mol = molecules[target]
    n_mol = len(molecules)

    for j, mol2 in enumerate(molecules):
        if target == j:
            continue

        atoms = get_bond_forming_atoms(target_mol, mol2, reactant, reactivity_matrix)
        if atoms.size > 0:
            alpha += np.mean(coordinates[atoms, :], axis=0)

    return alpha / n_mol


def get_all_bond_forming_atoms_in_molecule(molecule: ase.Atoms,
                                           reactants: bool,
                                           reactivity_matrix: dok_matrix) -> np.ndarray:
    # TODO: Write test
    search = 1 if reactants else -1
    atoms = molecule.get_tags()

    bonding_atoms = []
    for key, val in reactivity_matrix.items():
        if val == search and (key[0] in atoms or key[1] in atoms):
            bonding_atoms.append(key[0])

    return np.array(bonding_atoms)


def get_bond_forming_atoms(molecule1: ase.Atoms,
                           molecule2: ase.Atoms,
                           reactants: bool,
                           reactivity_matrix: dok_matrix) -> np.ndarray:
    search = 1 if reactants else -1

    atoms1, atoms2 = molecule1.get_tags(), molecule2.get_tags()

    bonding_atoms = []
    for key, val in reactivity_matrix.items():
        if val == search:
            if key[0] in atoms1 and key[1] in atoms2:
                bonding_atoms.append(key[0])
            elif key[1] in atoms1 and key[0] in atoms2:
                bonding_atoms.append(key[1])

    return np.array(bonding_atoms)


def get_reactive_atoms(shared_atoms: np.ndarray,
                       reactivity_matrix: dok_matrix) -> np.ndarray:
    reactive_atoms = []
    for atom in shared_atoms:
        row = reactivity_matrix[atom, :]
        if row.count_nonzero() > 0:
            reactive_atoms.append(atom)

    return np.array(reactive_atoms)


def get_reactivity_matrix(reactant: ase.Atoms, product: ase.Atoms) -> dok_matrix:
    reactant_connectivity = Analysis(reactant).adjacency_matrix[0]
    product_connectivity = Analysis(product).adjacency_matrix[0]

    return (product_connectivity - reactant_connectivity).todok()


# noinspection PyTypeChecker
def get_shared_atoms(reactant_molecule: Union[ase.Atoms, list[int]],
                     product_molecule: Union[ase.Atoms, list[int]]) -> np.ndarray:
    try:
        intersection = np.intersect1d(reactant_molecule.get_tags(), product_molecule.get_tags())
    except AttributeError:
        intersection =np.intersect1d(reactant_molecule, product_molecule)
    return intersection


def separate_molecules(system: ase.Atoms, molecules: Union[None, list[list[int]]] = None) -> list[ase.Atoms]:
    if molecules is None:
        return _separate_molecules_using_connectivity(system)
    else:
        return _separate_molecules_using_list(system, molecules)


def _separate_molecules_using_connectivity(system: ase.Atoms) -> list[ase.Atoms]:
    indices = list(range(len(system)))

    separated = []
    while indices:
        my_indices = connected_indices(system, indices[0])
        separated.append(ase.Atoms(cell=system.cell, pbc=system.pbc))

        for i in my_indices:
            separated[-1].append(system[i])
            del indices[indices.index(i)]

        separated[-1].set_tags(my_indices)

    return separated


def _separate_molecules_using_list(system: ase.Atoms,
                                   molecules: list[list[int]]) -> list[ase.Atoms]:
    separated = []
    for molecule in molecules:
        separated.append(ase.Atoms(cell=system.cell, pbc=system.pbc))
        for index in molecule:
            separated[-1].append(system[index])

        separated[-1].set_tags(molecule)

    return separated


class _CustomBaseCalculator(Calculator, ABC):
    implemented_properties = ['forces', 'energy']

    def __init__(self, molecules: list[list[int]], force_constant: float = 1.0, label=None, atoms=None,
                 directory='.', **kwargs):
        self.force_constant = force_constant
        self.molecules = molecules

        super().__init__(restart=None, label=label, atoms=atoms, directory=directory, **kwargs)

    def calculate(self, atoms=None, properties=None, system_changes=None) -> None:
        self.atoms = atoms.copy()
        molecules = separate_molecules(self.atoms, self.molecules)
        forces = self.compute_forces(molecules)

        submit_forces = np.zeros((len(self.atoms), 3))

        for mol, force in zip(molecules, forces):
            for index in mol.get_tags():
                submit_forces[index, :] = force

        self.results['forces'] = submit_forces
        self.results['energy'] = 0.0

    @abc.abstractmethod
    def compute_forces(self, molecules: list[ase.Atoms]) -> np.ndarray:
        pass
