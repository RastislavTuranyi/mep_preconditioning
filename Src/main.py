from __future__ import annotations

from typing import TYPE_CHECKING

import ase
from ase.build import separate, connected_indices
from ase.geometry.analysis import Analysis
import ase.io

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from scipy.sparse import dok_matrix


class InputError(Exception):
    pass


def main(start=None, end=None, both=None):
    if both is not None:
        reactant = ase.io.read(both, 0)
        product = ase.io.read(both, 1)
    elif start is not None and end is not None:
        reactant = ase.io.read(start)
        product = ase.io.read(end)
    else:
        raise InputError()

    reactant_molecules = separate_molecules(reactant)
    product_molecules = separate_molecules(product)

    reactivity_matrix = get_reactivity_matrix(reactant, product)

    initial_positioning(reactant, product, reactant_molecules, product_molecules, reactivity_matrix)


def initial_positioning(reactant: ase.Atoms,
                        product: ase.Atoms,
                        reactant_molecules: list[ase.Atoms],
                        product_molecules: list[ase.Atoms],
                        reactivity_matrix: dok_matrix):
    # Translate reactant molecules so that their geometric centres are at origin
    for i, reactant_mol in enumerate(reactant_molecules):
        reactant_mol.translate(np.mean(reactant_mol.get_positions(), axis=0))


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


def get_reactive_atoms(reactant_molecule: ase.Atoms,
                       product_molecule: ase.Atoms,
                       reactivity_matrix: dok_matrix) -> np.ndarray:
    shared_atoms = get_shared_atoms(reactant_molecule, product_molecule)

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


def get_indices(arr: np.ndarray, values):
    sorter = np.argsort(arr)
    return sorter[np.searchsorted(arr, values, sorter=sorter)]


# noinspection PyTypeChecker
def get_shared_atoms(reactant_molecule: ase.Atoms, product_molecule: ase.Atoms) -> np.ndarray:
    intersection = np.intersect1d(reactant_molecule.get_tags(), product_molecule.get_tags())
    return intersection


def separate_molecules(system: ase.Atoms):
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
