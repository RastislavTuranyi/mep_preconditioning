from __future__ import annotations

from typing import TYPE_CHECKING

import ase
from ase.build import separate, connected_indices
import ase.io

import numpy as np

from Src.common_functions import separate_molecules
from Src.stage1 import *
from Src.stage2 import fix_overlaps

if TYPE_CHECKING:
    from typing import Union
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

    reposition_reactants(reactant, reactant_molecules, reactivity_matrix)
    reposition_products(reactant, product, reactant_molecules, product_molecules, reactivity_matrix)

    fix_overlaps(reactant, [mol.get_tags() for mol in reactant_molecules])
    fix_overlaps(product, [mol.get_tags() for mol in reactant_molecules])

    ase.io.write('start.xyz', reactant)
    ase.io.write('end.xyz', product)
