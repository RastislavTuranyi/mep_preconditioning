from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import ase
import numpy as np
from scipy.spatial.transform import Rotation

from Src.common_functions import *

if TYPE_CHECKING:
    from typing import Union
    from numpy.typing import ArrayLike
    from scipy.sparse import dok_matrix


def find_largest_molecule(reactant, product):
    num_atom_reactant = np.array([len(mol) for mol in reactant])
    num_atom_product = np.array([len(mol) for mol in product])

    reactant_max_arg, product_max_arg = np.argmax(num_atom_reactant), np.argmax(num_atom_product)
    reactant_max, product_max = num_atom_reactant[reactant_max_arg], num_atom_product[product_max_arg]

    if reactant_max >= product_max:
        return reactant_max_arg, True
    else:
        return product_max_arg, False


def reposition_largest_molecule_system(system: ase.Atoms, indices: list[list[int]], largest_molecule_index: int,
                                       reactivity_matrix: dok_matrix):
    coordinates = system.get_positions()
    largest_molecule = indices[largest_molecule_index]

    # Move largest molecule to origin
    geometric_centre = np.mean(coordinates[largest_molecule], axis=0)
    coordinates[largest_molecule, :] += - geometric_centre

    non_reacting_molecules = []
    structure_vectors = []
    for i, molecule in enumerate(indices):
        if i == largest_molecule_index:
            continue

        # Find atoms in both molecules that will react
        reactive_atoms_largest, reactive_atoms_other = get_bond_forming_atoms(largest_molecule, molecule, True,
                                                                              reactivity_matrix, True, [1, -1])

        # Skip molecules that do not react with the largest molecule
        if len(reactive_atoms_largest) == 0 and len(reactive_atoms_other) == 0:
            non_reacting_molecules.append(molecule)
            continue

        # Translate the molecule so that the first bond-forming pair of atoms in this pair of molecules is in the
        # same location
        translation_vector = coordinates[reactive_atoms_largest[0], :] - coordinates[reactive_atoms_largest[0], :]
        coordinates[molecule] += translation_vector

        # If the molecules are going to form bonds in more than one place, rotate the smaller molecule so that the
        # atom that forms the second bond aligns with its corresponding atom in the largest molecule
        try:
            rotation, _ = Rotation.align_vectors(coordinates[reactive_atoms_largest[1]][np.newaxis, :],
                                                 coordinates[reactive_atoms_other[1]][np.newaxis, :])
            coordinates[molecule, :] = rotation.apply(coordinates[molecule, :])
        except IndexError:
            # pass

            # Rotate the other molecule so that it points to opposite direction as the largest molecule
            structure_vector_largest = np.mean(coordinates[largest_molecule, :] -
                                               np.mean(coordinates[reactive_atoms_largest], axis=0), axis=0)
            structure_vector_other = np.mean(coordinates[molecule, :] -
                                             np.mean(coordinates[reactive_atoms_other], axis=0), axis=0)

            rotation, _ = Rotation.align_vectors(- structure_vector_largest[np.newaxis, :],
                                                 structure_vector_other[np.newaxis, :])
            coordinates[molecule, :] = rotation.apply(coordinates[molecule, :])

            structure_vectors.append(structure_vector_largest)

    # Move molecules that do not participate far away
    if non_reacting_molecules:
        largest_distance = np.sqrt(np.max(np.sum((coordinates[largest_molecule, :] - geometric_centre) ** 2, axis=1)))
        mean_structure_vector = np.mean(structure_vectors, axis=0)
        length = np.linalg.norm(mean_structure_vector)

        for molecule in non_reacting_molecules:
            size = np.sqrt(np.max(np.sum((coordinates[molecule, :] - geometric_centre) ** 2, axis=1)))
            destination = 2 * (largest_distance + size) / length * mean_structure_vector
            coordinates[molecule, :] += destination - coordinates[molecule, :]

    system.set_positions(coordinates)


def reposition_other_system(main_system: ase.Atoms, other_system: ase.Atoms, main_molecules: list[list[int]],
                            other_molecules: list[list[int]], max_iter: int = 100, max_rssd: float = 1.05):
    main_coordinates = main_system.get_positions()
    other_coordinates = other_system.get_positions()

    for other_mol in other_molecules:
        shared_atoms, main_mol = find_most_similar_molecule(other_mol, main_molecules)
        logging.debug(f'Shared atoms = {shared_atoms}, main mol = {main_mol}')

        # Treat finding the optimal rotation as an optimisation problem
        for i in range(max_iter):
            # Translate the molecule to the geometric centre of the shared atoms in the main image
            try:
                logging.debug(f'Molecule at {np.mean(other_coordinates[other_mol], axis=0)} moved to ')
                other_coordinates[other_mol, :] += np.mean(main_coordinates[shared_atoms], axis=0) - \
                                                   np.mean(other_coordinates[shared_atoms], axis=0)
                logging.debug(f'{np.mean(other_coordinates[other_mol], axis=0)}  (destination='
                              f'{np.mean(main_coordinates[shared_atoms], axis=0)})\n')
            except TypeError:
                raise Exception()

            # Rotate molecule so that the shared atoms in the two images are aligned well
            rotation, rssd = Rotation.align_vectors(main_coordinates[shared_atoms], other_coordinates[shared_atoms])
            other_coordinates[other_mol, :] = rotation.apply(other_coordinates[other_mol, :])

            logging.debug(f'Molecule rotated; current rotation diff = '
                          f'{Rotation.align_vectors(main_coordinates[shared_atoms], other_coordinates[shared_atoms])}')
            logging.debug(f'New geometric centre={np.mean(other_coordinates[other_mol], axis=0)}  (destination='
                          f'{np.mean(main_coordinates[shared_atoms], axis=0)})\n')

            if rssd < max_rssd:
                break

    other_system.set_positions(other_coordinates)


def find_most_similar_molecule(target_molecule, other_system_molecules):
    most_shared_number = 0
    most_shared_list = []
    index = None

    for i, molecule in enumerate(other_system_molecules):
        shared_atoms = get_shared_atoms(target_molecule, molecule)
        if len(shared_atoms) > most_shared_number:
            most_shared_list = shared_atoms
            most_shared_number = len(shared_atoms)
            index = i

    return most_shared_list, other_system_molecules[index]
