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
import Src.stage3 as stage3
import Src.stage4 as stage4

if TYPE_CHECKING:
    from typing import Union
    from numpy.typing import ArrayLike
    from scipy.sparse import dok_matrix


np.seterr(all='raise')

STEPWISE_OUTPUT = True


class InputError(Exception):
    pass


def main(start=None, end=None, both=None):
    if both is not None:
        reactant = ase.io.read(both, 0)
        product = ase.io.read(both, -1)
        ase.io.write('start.xyz', [reactant, product])
    elif start is not None and end is not None:
        reactant = ase.io.read(start)
        product = ase.io.read(end)
    else:
        raise InputError()
    logging.basicConfig(level=logging.DEBUG)

    reactant_molecules = separate_molecules(reactant)
    reactant_indices = [mol.get_tags() for mol in reactant_molecules]
    logging.info(f'{len(reactant_indices)} molecules were found in the reactant system.')
    logging.debug(f'   > {reactant_indices}')

    product_molecules = separate_molecules(product)
    product_indices = [mol.get_tags() for mol in product_molecules]
    logging.info(f'{len(product_indices)} molecules were found in the product system.')
    logging.debug(f'   > {product_indices}')

    reactivity_matrix = get_reactivity_matrix(reactant, product)
    logging.debug(f'Reactivity matrix obtained: {repr(reactivity_matrix.items())}')

    logging.info('** Starting STAGE 1 **')
    stage1.reposition_reactants(reactant, reactant_indices, reactivity_matrix)
    stage1.reposition_products(reactant, product, reactant_indices, product_indices, reactivity_matrix)

    if STEPWISE_OUTPUT:
        ase.io.write('stage1.xyz', [reactant, product])

    logging.info('** Starting STAGE 2 **')
    stage2.fix_overlaps(reactant, reactant_indices)
    stage2.fix_overlaps(product, product_indices)

    if STEPWISE_OUTPUT:
        ase.io.write('stage2.xyz', [reactant, product])

    reactant_molecules = separate_molecules(reactant, reactant_indices)
    product_molecules = separate_molecules(product, product_indices)

    logging.info('** Starting STAGE 3 **')
    stage3.reorient_reactants(reactant, reactant_molecules, reactivity_matrix)
    stage3.reorient_products(product, product_indices, reactant, reactant_indices)

    if STEPWISE_OUTPUT:
        ase.io.write('stage3.xyz', [reactant, product])
        return

    stage4.reposition_reactants(reactant, reactant_indices, product, product_indices, reactivity_matrix)

    if STEPWISE_OUTPUT:
        ase.io.write('stage4.xyz', [reactant, product])
