from __future__ import annotations

from typing import TYPE_CHECKING

import ase
import numpy as np

from Src.common_functions import *

if TYPE_CHECKING:
    from typing import Union
    from numpy.typing import ArrayLike
    from scipy.sparse import dok_matrix


def reposition_reactants(reactant: ase.Atoms,
                         reactant_molecules: list[ase.Atoms],
                         reactivity_matrix: dok_matrix) -> None:
    coordinates = reactant.get_positions()
    # Translate reactant molecules so that their geometric centres are (approximately) at origin
    for reactant_mol in reactant_molecules:
        reactant_mol.translate(-np.mean(reactant_mol.get_positions(), axis=0))

    # Calculate the geometric centres of bond-forming atoms (A, alpha)
    for i, mol1 in enumerate(reactant_molecules):
        alpha = compute_alpha_vector(coordinates, i, reactant_molecules, True, reactivity_matrix)
        mol1.translate(alpha)
        coordinates[mol1.get_tags()] += alpha

    reactant.set_positions(coordinates)


def reposition_products(product: ase.Atoms,
                        reactant: ase.Atoms,
                        reactant_molecules: list[ase.Atoms],
                        product_molecules: list[ase.Atoms],
                        reactivity_matrix: dok_matrix) -> None:
    n_reactants = len(reactant_molecules)
    reactant_coordinates = reactant.get_positions()
    product_coordinates = product.get_positions()

    for product_mol in product_molecules:
        beta, sigma = np.zeros(3), np.zeros(3)
        # Calculate beta and sigma for each reaction molecule
        for reactant_mol in reactant_molecules:
            shared = get_shared_atoms(reactant_mol, product_mol)
            reactive = get_reactive_atoms(shared, reactivity_matrix)

            if shared.size > 0:
                beta += np.mean(reactant_coordinates[shared, :], axis=0)
            if reactive.size > 0:
                sigma += np.mean(reactant_coordinates[reactive, :], axis=0)

        # Compute displacement vector from current position to the destination
        destination = (beta / n_reactants + sigma / n_reactants) / 2
        displacement = destination - np.mean(product_mol.get_positions(), axis=0)

        # Move the molecule itself as well as the entire system
        product_mol.translate(displacement)
        product_coordinates[product_mol.get_tags()] += displacement

    product.set_positions(product_coordinates)
