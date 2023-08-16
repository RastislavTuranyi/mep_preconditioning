from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import ase
from ase.build import separate, connected_indices
import ase.io

import numpy as np

from Src.common_functions import separate_molecules, get_reactivity_matrix
import Src.stage1 as stage1
import Src.stage2 as stage2

if TYPE_CHECKING:
    from typing import Union
    from numpy.typing import ArrayLike
    from scipy.sparse import dok_matrix

np.seterr(all='raise')


class InputError(Exception):
    pass


def precondition_path_ends(start=None,
                           end=None,
                           both=None,
                           output: list[str] = ('preconditioned.xyz',),
                           stepwise_output: bool = False,
                           max_iter: int = 1000,
                           max_rssd: float = 1.05,
                           force_constant: float = 1.0,
                           fmax: float = 1e-5,
                           non_convergence_limit: Union[float, None] = 0.001,
                           non_convergence_roof: Union[float, None] = 0.5,
                           trial_constants: Union[
                               None, float, tuple[float], tuple[float, float], tuple[float, float, float],
                               list[float], np.ndarray] = 10.0):
    logging.basicConfig(level=logging.DEBUG)

    logging.info('Reading input')
    reactant, product, reactant_molecules, product_molecules, \
        reactant_indices, product_indices = read_input(start, end, both)

    logging.info(f'{len(reactant_indices)} molecules were found in the reactant system.')
    logging.debug(f'   > {reactant_indices}')

    logging.info(f'{len(product_indices)} molecules were found in the product system.')
    logging.debug(f'   > {product_indices}')

    reactivity_matrix = get_reactivity_matrix(reactant, product)
    logging.debug(f'Reactivity matrix obtained: {repr(reactivity_matrix.items())}')

    logging.info('** Starting STAGE 1 **')
    index, is_reactant = stage1.find_largest_molecule(reactant_indices, product_indices)
    logging.info(f'The largest molecule is in the {"REACTANT" if is_reactant else "PRODUCT"} system: {index}')

    if is_reactant:
        main_system, other_system = reactant, product
        main_indices, other_indices = reactant_indices, product_indices
    else:
        main_system, other_system = product, reactant
        main_indices, other_indices = product_indices, reactant_indices

    logging.info(' * Working on the system containing the largest molecule * ')
    stage1.reposition_largest_molecule_system(main_system, main_indices, index, reactivity_matrix)
    logging.info(' * Working on the other system * ')
    stage1.reposition_other_system(main_system, other_system, main_indices, other_indices, max_iter, max_rssd)

    if stepwise_output:
        ase.io.write('stage1_new.xyz', [reactant, product])

    logging.info('*** Starting STAGE 2 ***')
    stage2.fix_overlaps(reactant, reactant_indices, reactivity_matrix, force_constant, fmax, max_iter,
                        non_convergence_roof, non_convergence_limit, trial_constants)
    stage2.fix_overlaps(product, product_indices, reactivity_matrix, force_constant, fmax, max_iter,
                        non_convergence_roof, non_convergence_limit, trial_constants)

    if stepwise_output:
        ase.io.write('stage2.xyz', [reactant, product])

    if len(output) == 1:
        logging.info(f'Writing both reactant and product into ONE file: {output[0]}')
        ase.io.write(output[0], [reactant, product])
    else:
        logging.info(f'Writing reactants and products into SEPARATE files: {output[0]}, {output[1]}')
        ase.io.write(output[0], reactant)
        ase.io.write(output[1], product)

    logging.info('Finished')


def read_input(start=None, end=None, both=None):
    if both is not None:
        try:
            result = ase.io.read(both)
            reactant, product = result[0], result[-1]
        except IndexError:
            raise InputError()

        logging.info('One input file provided - first and last images read successfully')

        product_molecules = separate_molecules(product)
        reactant_molecules = separate_molecules(reactant)

        reactant_indices = [mol.get_tags() for mol in reactant_molecules]
        product_indices = [mol.get_tags() for mol in product_molecules]

    elif start is not None and end is not None:
        logging.info('First and last images provided in separate files')
        start = [start] if isinstance(start, str) else start
        end = [end] if isinstance(end, str) else end

        if len(start) == 1:
            reactant = ase.io.read(start[0])
            reactant_molecules = separate_molecules(reactant)
            reactant_indices = [mol.get_tags() for mol in reactant_molecules]
            logging.info('Reactant read successfully from one file')
        else:
            reactant, reactant_molecules, reactant_indices = assemble_system(start)
            logging.info(f'Reactant read successfully from {len(start)} files; total {len(reactant)} atoms found')

        if len(end) == 1:
            product = ase.io.read(end[0])
            product_molecules = separate_molecules(product)
            product_indices = [mol.get_tags() for mol in product_molecules]
            logging.info('Product read successfully from one file')
        else:
            product, product_molecules, product_indices = assemble_system(end)
            logging.info(f'Product read successfully from {len(end)} files; total {len(product)} atoms found')

    else:
        raise InputError()

    return reactant, product, reactant_molecules, product_molecules, reactant_indices, product_indices


def assemble_system(filenames: list[str]) -> tuple[ase.Atoms, list[ase.Atoms], list[list[int]]]:
    molecules = []
    numbers = []
    coordinates = []
    indices = []
    last_index = 0

    cell = np.zeros((3, 3))
    pbc = np.array([False, False, False])

    for filename in filenames:
        molecule = ase.io.read(filename, 0)

        n_atoms = len(molecule)
        molecule_indices = list(range(last_index, last_index + n_atoms))
        last_index += n_atoms
        molecule.set_tags(molecule_indices)

        molecules.append(molecule)
        indices.append(molecule_indices)
        numbers.extend(list(molecule.get_atomic_numbers()))
        coordinates.extend(list(molecule.get_positions()))

    system = ase.Atoms.fromdict({'numbers': np.array(numbers), 'positions': np.array(coordinates),
                                 'cell': cell, 'pbc': pbc})

    return system, molecules, indices
