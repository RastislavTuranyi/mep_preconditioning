import numpy as np
from numpy.testing import assert_allclose
import pytest

from Src.common_functions import separate_molecules, get_reactivity_matrix
from Src.stage1 import *

from Tests.common_fixtures import ester_hydrolysis_reaction, set_up_separate_molecules


def test_reposition_reactants(set_up_separate_molecules):
    reactant, product, reactant_indices, _, _, _, reactivity_matrix = set_up_separate_molecules

    product_expected = product.get_positions()
    reactant_expected = np.array([[-1.55600045, -0.01439364, -0.40663091],
                                  [-1.23201045, -1.04254364, -0.23333091],
                                  [-0.70949045, 0.64189636, -1.48285091],
                                  [-2.60432045, -0.03587364, -0.71336091],
                                  [-1.48642045, 0.54308636, 0.52962909],
                                  [-1.20709045, 1.08634636, -2.49626091],
                                  [0.62632955, 0.72891636, -1.25726091],
                                  [1.26464955, 0.20698636, -0.06497091],
                                  [2.33653955, 0.41217636, -0.12793091],
                                  [0.85680955, 0.70057636, 0.82120909],
                                  [1.10953955, -0.87355364, -0.00536091],
                                  [-0.091485, -0.0342225, -0.7057725],
                                  [0.030495, 0.0114075, 0.2352575]])

    reposition_reactants(reactant, reactant_indices, reactivity_matrix)
    print(repr(reactant.get_positions()))
    assert_allclose(reactant.get_positions(), reactant_expected, rtol=0, atol=10e-9)
    assert_allclose(product.get_positions(), product_expected)


def test_reposition_products(set_up_separate_molecules):
    reactant, product, reactant_indices, product_indices, _, _, reactivity_matrix = set_up_separate_molecules

    reactant_expected = reactant.get_positions()
    product_expected = np.array([[-1.10267583, 1.38115208, 1.69827458],
                                 [-0.75933583, 0.35489208, 1.84614458],
                                 [-0.21757583, 2.09084208, 0.69906458],
                                 [-2.13544583, 1.35610208, 1.34202458],
                                 [-1.07999583, 1.89746208, 2.66073458],
                                 [-0.60885583, 2.49192208, -0.37692542],
                                 [1.631778, 0.7532195, 0.196242],
                                 [0.401618, 0.8707895, 0.850522],
                                 [0.067398, -0.1213105, 1.224302],
                                 [0.520928, 1.5476095, 1.721452],
                                 [-0.357142, 1.3121595, 0.168492],
                                 [1.01821417, 2.23061208, 1.10013458],
                                 [1.09251417, 1.83391208, 1.98472458]])

    reposition_products(reactant, product, reactant_indices, product_indices, reactivity_matrix)

    assert_allclose(product.get_positions(), product_expected)
    assert np.all(reactant.get_positions() == reactant_expected)
