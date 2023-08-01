import numpy as np
from numpy.testing import assert_allclose
import pytest

from Src.common_functions import separate_molecules, get_reactivity_matrix
from Src.stage1 import *

from Tests.common_fixtures import ester_hydrolysis_reaction


def test_initial_positioning(ester_hydrolysis_reaction):
    reactant, product = ester_hydrolysis_reaction

    reactant_molecules = separate_molecules(reactant)
    reactivity_matrix = get_reactivity_matrix(reactant, product)

    reposition_reactants(reactant, reactant_molecules, reactivity_matrix)

    reactant_expected = np.array([[-1.05141, 5.188985, 4.03545],
                                  [-0.72742, 4.160835, 4.20875],
                                  [-0.2049, 5.845275, 2.95923],
                                  [-2.09973, 5.167505, 3.72872],
                                  [-0.98183, 5.746465, 4.97171],
                                  [-0.7025, 6.289725, 1.94582],
                                  [1.13092, 5.932295, 3.18482],
                                  [1.76924, 5.410365, 4.37711],
                                  [2.84113, 5.615555, 4.31415],
                                  [1.3614, 5.903955, 5.26329],
                                  [1.61413, 4.329825, 4.43672],
                                  [-1.391955, -0.828885, 0.567825],
                                  [-1.269975, -0.783255, 1.508855]])
    print(repr(reactant.get_positions()))
    assert_allclose(reactant.get_positions(), reactant_expected)


def test_reposition_products(ester_hydrolysis_reaction):
    reactant, product = ester_hydrolysis_reaction

    reactant_molecules = separate_molecules(reactant)
    product_molecules = separate_molecules(product)

    reactivity_matrix = get_reactivity_matrix(reactant, product)

    reposition_products(reactant, product, reactant_molecules, product_molecules, reactivity_matrix)

    product_expected = np.array([[-4.01608, 0.23907, 0.06919],
                                 [-3.67274, -0.78719, 0.21706],
                                 [-3.13098, 0.94876, -0.93002],
                                 [-5.04885, 0.21402, -0.28706],
                                 [-3.9934, 0.75538, 1.03165],
                                 [-3.52226, 1.34984, -2.00601],
                                 [2.13267, 2.28495, 1.85001],
                                 [0.90251, 2.40252, 2.50429],
                                 [0.56829, 1.41042, 2.87807],
                                 [1.02182, 3.07934, 3.37522],
                                 [0.14375, 2.84389, 1.82226],
                                 [-1.89519, 1.08853, -0.52895],
                                 [-1.82089, 0.69183, 0.35564]])
    print(repr(product.get_positions()))
    print()
    assert_allclose(product.get_positions(), product_expected)
