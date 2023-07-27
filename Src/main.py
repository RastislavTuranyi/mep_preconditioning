from __future__ import annotations

from itertools import combinations
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
    coordinates = reactant.get_positions()

    shared_atom_geometric_centres = []
    reactive_atom_geometric_centres = []

    for reactant_mol in reactant_molecules:
        # Translate reactant molecules so that their geometric centres are (approximately) at origin
        reactant_mol.translate(-np.mean(reactant_mol.get_positions(), axis=0))

        # Calculate geometric centres of reactive (B) and shared atoms (C)
        for product_mol in product_molecules:
            shared = get_shared_atoms(reactant_mol, product_mol)
            reactive = get_reactive_atoms(shared, reactivity_matrix)

            if shared.size > 0:
                shared_atom_geometric_centres.append(np.mean(coordinates[shared, :], axis=0))
            if reactive.size > 0:
                reactive_atom_geometric_centres.append(np.mean(coordinates[reactive, :], axis=0))

    # Calculate the geometric centres of bond-forming atoms (A)
    bonding_atom_geometric_centres = []
    for rmol1, rmol2 in combinations(reactant_molecules, 2):
        atoms = get_bond_forming_atoms(rmol1, rmol2, True, reactivity_matrix)
        bonding_atom_geometric_centres.append(np.mean(coordinates[atoms, :], axis=0))

    # Calculate the vectors which will be used to reposition the reactants and products
    n_reactants = len(reactant_molecules)
    beta = np.sum(np.array(reactive_atom_geometric_centres), axis=0) / n_reactants
    sigma = np.sum(np.array(shared_atom_geometric_centres), axis=0) / n_reactants

    reactant_repositioning_vector = np.sum(np.array(bonding_atom_geometric_centres), axis=0) / n_reactants
    product_repositioning_vector = (beta + sigma) / 2

    # Translate all reactants and products using the computed vectors
    reactant.translate(reactant_repositioning_vector)
    for reactant_mol in reactant_molecules:
        reactant_mol.translate(reactant_repositioning_vector)

    product.translate(product_repositioning_vector)
    for product_mol in product_molecules:
        product_mol.translate(product_repositioning_vector)


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
