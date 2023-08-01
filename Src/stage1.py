from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import ase
import numpy as np

from Src.common_functions import *

if TYPE_CHECKING:
    from typing import Union
    from numpy.typing import ArrayLike
    from scipy.sparse import dok_matrix


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

        # Calculate geometric centres of reactive (B, beta) and shared atoms (C, sigma)
        for product_mol in product_molecules:
            shared = get_shared_atoms(reactant_mol, product_mol)
            reactive = get_reactive_atoms(shared, reactivity_matrix)

            if shared.size > 0:
                shared_atom_geometric_centres.append(np.mean(coordinates[shared, :], axis=0))
            if reactive.size > 0:
                reactive_atom_geometric_centres.append(np.mean(coordinates[reactive, :], axis=0))

    # Calculate the geometric centres of bond-forming atoms (A, alpha)
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