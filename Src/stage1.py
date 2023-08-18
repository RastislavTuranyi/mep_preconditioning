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


def reposition_everything(main_system: ase.Atoms,
                          other_system: ase.Atoms,
                          main_molecules: list[list[int]],
                          other_molecules: list[list[int]],
                          largest_molecule_index: int,
                          reactivity_matrix: dok_matrix,
                          max_iter: int = 100,
                          fmax: float = 1e-5,
                          non_convergence_limit: Union[float, None] = 0.001,
                          non_convergence_roof: Union[float, None] = 0.5
                          ):
    from copy import deepcopy
    main_coordinates = main_system.get_positions()
    other_coordinates = other_system.get_positions()

    largest_molecule = main_molecules[largest_molecule_index]

    # Move largest molecule to origin
    geometric_centre = np.mean(main_coordinates[largest_molecule], axis=0)
    main_coordinates[largest_molecule, :] -= geometric_centre

    logging.debug(f'Largest molecule moved to origin: {np.mean(main_coordinates[largest_molecule], axis=0)}')

    set_atoms_main = deepcopy(list(largest_molecule))
    set_atoms_other = []

    optimised_main_molecules, optimised_other_molecules = [], []

    main_molecules_copy = deepcopy(main_molecules)
    optimised_main_molecules.append(main_molecules_copy.pop(largest_molecule_index))
    other_molecules_copy = deepcopy(other_molecules)

    n_atoms = len(main_system)
    unoptimised_main, unoptimised_other = [], []
    while len(set_atoms_main) < n_atoms or len(set_atoms_other) < n_atoms:
        # Choose which system to optimise a molecule from (the one that has the fewer optimised molecules)
        if len(set_atoms_main) < len(set_atoms_other):
            logging.debug('Optimising molecule from the MAIN system ...')
            molecules, coordinates, set_atoms = main_molecules_copy, main_coordinates, set_atoms_main
            target_molecules, target_coordinates = optimised_other_molecules, other_coordinates
            optimised_molecules = optimised_main_molecules
        else:
            logging.debug('Optimising molecule from the OTHER system ...')
            molecules, coordinates, set_atoms = other_molecules_copy, other_coordinates, set_atoms_other
            target_molecules, target_coordinates = optimised_main_molecules, main_coordinates
            optimised_molecules = optimised_other_molecules

        # Choose molecule to optimise (the largest one)
        index, largest = 0, 0
        for i, mol in enumerate(molecules):
            if len(mol) > largest:
                index, largest = i, len(mol)
        molecule = molecules[index]
        logging.debug(f'Optimising molecule {molecule}')

        previous_centre = np.zeros(3)
        for i in range(max_iter):
            try:
                shared_atoms, closest_molecule = find_most_similar_molecule(molecule, target_molecules)
            except TypeError:
                unoptimised = unoptimised_main if len(set_atoms_main) < len(set_atoms_other) else unoptimised_other
                unoptimised.append((molecule, index))
                molecules.pop(index)
                break

            new_centre = np.mean(coordinates[shared_atoms, :], axis=0)
            if np.allclose(previous_centre, new_centre):
                logging.debug(f'{previous_centre=}    {new_centre=}')
                optimised_molecules.append(molecules.pop(index))
                break

            # Move the molecule to the geometric centre of the most similar molecule in the complementary system
            endpoint = np.mean(target_coordinates[shared_atoms, :], axis=0)
            coordinates[molecule, :] += endpoint - new_centre

            # Find Kabsch rotation and rotate the molecule accordingly
            rotation, rssd = Rotation.align_vectors(target_coordinates[shared_atoms], coordinates[shared_atoms])
            coordinates[molecule, :] = rotation.apply(coordinates[molecule, :])

            previous_centre = new_centre
        else:
            optimised_molecules.append(molecules.pop(index))

        set_atoms.extend(molecule)

    main_system.set_positions(main_coordinates)
    other_system.set_positions(other_coordinates)

    # Move molecules that do not participate far away
    if unoptimised_main or unoptimised_other:
        logging.debug(f'Non-reacting molecules found: {unoptimised_main}, {unoptimised_other}')

        for coordinates, unoptimised, system, molecules in zip([main_coordinates, other_coordinates],
                                                               [unoptimised_main, unoptimised_other],
                                                               [main_system, other_system],
                                                               [main_molecules, other_molecules]):
            # Move each molecule to the origin
            for molecule, _ in unoptimised:
                coordinates[molecule, :] -= np.mean(coordinates[molecule], axis=0)
            system.set_positions(coordinates)

            system.calc = HardSphereNonReactiveCalculator(molecules, reactivity_matrix, [i for _, i in unoptimised])
            dyn = ConstrainedBFGS(system, non_convergence_limit, non_convergence_roof)
            dyn.run(fmax=fmax, steps=max_iter)
    else:
        logging.debug('All molecules participated in the reaction')


def find_most_similar_molecule(target_molecule: list[int],
                               other_system_molecules: list[list[int]]) -> tuple[np.ndarray, list[int]]:
    most_shared_number = 0
    most_shared_list = []
    index = None

    for i, molecule in enumerate(other_system_molecules):
        shared_atoms = get_shared_atoms(target_molecule, molecule)
        if len(shared_atoms) > most_shared_number:
            most_shared_list = shared_atoms
            most_shared_number = len(shared_atoms)
            index = i

    return most_shared_list, list(other_system_molecules[index])


class HardSphereNonReactiveCalculator(_CustomBaseCalculator):
    def __init__(self,
                 molecules: list[list[int]],
                 reactivity_matrix: dok_matrix,
                 unoptimised: list[int],
                 force_constant: float = 500.0,
                 label=None,
                 atoms=None,
                 directory='.',
                 **kwargs):
        self.reactivity_matrix = reactivity_matrix
        self.molecular_reactivity_matrix = construct_molecular_reactivity_matrix(molecules, reactivity_matrix)
        self.unoptimised = unoptimised

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
            if i not in self.unoptimised:
                continue  # Don't move molecules that have been optimised

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
