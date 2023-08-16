from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import ase
import numpy as np
from scipy.spatial.transform import Rotation

from Src.common_functions import *
from Src.common_functions import _CustomBaseCalculator
from Src.stage2 import determine_overlaps

if TYPE_CHECKING:
    from typing import Union
    from numpy.typing import ArrayLike
    from scipy.sparse import dok_matrix


def compute_structure_vector(coordinates: np.ndarray,
                             molecule: list[int],
                             reactive_atoms: list[int]) -> np.ndarray:
    """
    Computes the mean structure vector of a molecule; mean structure vector is
    :param coordinates:
    :param molecule:
    :param reactive_atoms:
    :return:
    """
    return np.mean(coordinates[molecule, :] - np.mean(coordinates[reactive_atoms], axis=0), axis=0)


def construct_molecular_reactivity_matrix(molecules: list[list[int]], reactivity_matrix: dok_matrix):
    result = np.zeros((len(molecules), len(molecules)))

    for (key1, key2), val in reactivity_matrix.items():
        for i, molecule in enumerate(molecules):
            if key1 in molecule:
                index1 = i
            if key2 in molecule:
                index2 = i

        try:
            result[index1, index2], result[index2, index1] = val, val
        except NameError:
            raise Exception()

    return result


def find_largest_molecule(reactant, product):
    num_atom_reactant = np.array([len(mol) for mol in reactant])
    num_atom_product = np.array([len(mol) for mol in product])

    reactant_max_arg, product_max_arg = np.argmax(num_atom_reactant), np.argmax(num_atom_product)
    reactant_max, product_max = num_atom_reactant[reactant_max_arg], num_atom_product[product_max_arg]

    if reactant_max >= product_max:
        return reactant_max_arg, True
    else:
        return product_max_arg, False


def reposition_largest_molecule_system(system: ase.Atoms,
                                       indices: list[list[int]],
                                       largest_molecule_index: int,
                                       reactivity_matrix: dok_matrix,
                                       fmax: float = 1e-5,
                                       max_iter: int = 1000,
                                       non_convergence_limit: Union[float, None] = 0.001,
                                       non_convergence_roof: Union[float, None] = 0.5):
    coordinates = system.get_positions()
    largest_molecule = indices[largest_molecule_index]

    # Move largest molecule to origin
    geometric_centre = np.mean(coordinates[largest_molecule], axis=0)
    coordinates[largest_molecule, :] -= geometric_centre

    logging.debug(f'Largest molecule moved to origin: {np.mean(coordinates[largest_molecule], axis=0)}')

    non_reacting_molecules = []
    for i, molecule in enumerate(indices):
        if i == largest_molecule_index:
            continue

        # Find atoms in both molecules that will react
        reactive_atoms_largest, reactive_atoms_other = get_bond_forming_atoms(largest_molecule, molecule, True,
                                                                              reactivity_matrix, True, [1, -1])
        logging.debug(f'Molecule {molecule} reacts with the largest molecule at {reactive_atoms_other} '
                      f'({reactive_atoms_largest=})')

        # Skip molecules that do not react with the largest molecule
        if len(reactive_atoms_largest) == 0 and len(reactive_atoms_other) == 0:
            non_reacting_molecules.append(molecule)
            continue

        # Translate the molecule so that the first bond-forming pair of atoms in this pair of molecules is in the
        # same location
        translation_vector = coordinates[reactive_atoms_largest[0], :] - coordinates[reactive_atoms_other[0], :]
        coordinates[molecule] += translation_vector
        logging.debug(f'Molecule translated by {translation_vector} to {np.mean(coordinates[molecule], axis=0)}')

        # If the molecules are going to form bonds in more than one place, rotate the smaller molecule so that the
        # atom that forms the second bond aligns with its corresponding atom in the largest molecule
        try:
            rotation, _ = Rotation.align_vectors(coordinates[reactive_atoms_largest[1]][np.newaxis, :],
                                                 coordinates[reactive_atoms_other[1]][np.newaxis, :])
            coordinates[molecule, :] = rotation.apply(coordinates[molecule, :])
        except IndexError:
            # Rotate the other molecule so that it points to opposite direction as the largest molecule
            # Calculate the geometric centre of
            structure_vector_largest = compute_structure_vector(coordinates, largest_molecule, reactive_atoms_largest)
            structure_vector_other = compute_structure_vector(coordinates, molecule, reactive_atoms_other)

            rotation, _ = Rotation.align_vectors(- structure_vector_largest[np.newaxis, :],
                                                 structure_vector_other[np.newaxis, :])
            coordinates[molecule, :] = rotation.apply(coordinates[molecule, :])

    system.set_positions(coordinates)

    # Move molecules that do not participate far away
    if non_reacting_molecules:
        logging.debug(f'Non-reacting molecules found: {non_reacting_molecules}\nStructure vectors: {structure_vectors}')

        # Move each molecule to the origin
        for molecule in non_reacting_molecules:
            coordinates[molecule, :] -= np.mean(coordinates[molecule], axis=0)
        system.set_positions(coordinates)

        system.calc = HardSphereNonReactiveCalculator(indices, reactivity_matrix)
        dyn = ConstrainedBFGS(system, non_convergence_limit, non_convergence_roof)
        dyn.run(fmax=fmax, steps=max_iter)


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


class HardSphereNonReactiveCalculator(_CustomBaseCalculator):
    def __init__(self,
                 molecules: list[list[int]],
                 reactivity_matrix: dok_matrix,
                 force_constant: float = 500.0,
                 label=None,
                 atoms=None,
                 directory='.',
                 **kwargs):
        self.reactivity_matrix = reactivity_matrix
        self.molecular_reactivity_matrix = construct_molecular_reactivity_matrix(molecules, reactivity_matrix)

        super().__init__(molecules, force_constant, label=label, atoms=atoms, directory=directory, **kwargs)

    def compute_forces(self) -> np.ndarray:
        coordinates = self.atoms.get_positions()

        geometric_centres = [np.mean(coordinates[mol], axis=0) for mol in self.molecules]
        molecular_radii = [estimate_molecular_radius(coordinates[mol], centre) for mol, centre in
                           zip(self.molecules, geometric_centres)]

        n_mol = len(self.molecules)
        overlaps = determine_overlaps(n_mol, geometric_centres, molecular_radii)

        forces = np.zeros((n_mol, 3), dtype=np.float64)
        for i, affected_mol in enumerate(self.molecules):
            n_atoms = len(affected_mol)
            n = 3 * n_atoms * np.sum(overlaps[i, :])

            pairwise_forces = []
            for j, other_mol in enumerate(self.molecules):
                if overlaps[i, j] == 0 or self.molecular_reactivity_matrix[i, j] != 0 or i == j:
                    continue

                centre_diff = np.mean(coordinates[affected_mol], axis=0) - np.mean(coordinates[other_mol], axis=0)

                distance = np.linalg.norm(centre_diff)
                phi = self.force_constant * (distance - (molecular_radii[i] + molecular_radii[j])) / n
                pairwise_forces.append(phi * centre_diff / distance)

            forces[i, :] = n_atoms * np.sum(np.array(pairwise_forces), axis=0)

        return - forces
