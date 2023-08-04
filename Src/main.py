from __future__ import annotations

from typing import TYPE_CHECKING

import ase
from ase.build import separate, connected_indices
import ase.io

import numpy as np

from Src.common_functions import separate_molecules, get_reactivity_matrix
import Src.stage1 as stage1
from Src.stage2 import fix_overlaps
from Src.stage3 import reorient_reactants, reorient_products
import Src.stage4 as stage4

if TYPE_CHECKING:
    from typing import Union
    from numpy.typing import ArrayLike
    from scipy.sparse import dok_matrix


class InputError(Exception):
    pass


def main(start=None, end=None, both=None):
    if both is not None:
        reactant = ase.io.read(both, 0)
        product = ase.io.read(both, -1)
    elif start is not None and end is not None:
        reactant = ase.io.read(start)
        product = ase.io.read(end)
    else:
        raise InputError()

    reactant_molecules = separate_molecules(reactant)
    reactant_indices = [mol.get_tags() for mol in reactant_molecules]
    product_molecules = separate_molecules(product)
    product_indices = [mol.get_tags() for mol in product_molecules]

    reactivity_matrix = get_reactivity_matrix(reactant, product)

    stage1.reposition_reactants(reactant, reactant_molecules, reactivity_matrix)
    stage1.reposition_products(reactant, product, reactant_molecules, product_molecules, reactivity_matrix)

    fix_overlaps(reactant, reactant_indices)
    fix_overlaps(product, product_indices)

    reactant_molecules = separate_molecules(reactant, reactant_indices)
    product_molecules = separate_molecules(product, product_indices)

    reorient_reactants(reactant, reactant_molecules, reactivity_matrix)
    reorient_products(product, product_indices, reactant, reactant_indices)

    stage4.reposition_reactants(reactant, reactant_indices, product, product_indices, reactivity_matrix)

    ase.io.write('start.xyz', reactant)
    ase.io.write('end.xyz', product)
