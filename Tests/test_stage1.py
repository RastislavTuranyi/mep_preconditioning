import numpy as np
import pytest

from Src.common_functions import separate_molecules, get_reactivity_matrix
from Src.stage1 import initial_positioning

from Tests.common_fixtures import ester_hydrolysis_reaction


def test_initial_positioning(ester_hydrolysis_reaction):
    reactant, product = ester_hydrolysis_reaction

    reactant_molecules = separate_molecules(reactant)
    product_molecules = separate_molecules(product)

    reactivity_matrix = get_reactivity_matrix(reactant, product)

    initial_positioning(reactant, product, reactant_molecules, product_molecules, reactivity_matrix)

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
                                  [-0.99627, 1.395835, 1.36496],
                                  [-0.87429, 1.441465, 2.30599]])

    product_expected = np.array([[-4.03730858, 2.81617558, 2.27066408],
                                 [-3.69396858, 1.78991558, 2.41853408],
                                 [-3.15220858, 3.52586558, 1.27145408],
                                 [-5.07007858, 2.79112558, 1.91441408],
                                 [-4.01462858, 3.33248558, 3.23312408],
                                 [-3.54348858, 3.92694558, 0.19546408],
                                 [2.11144142, 4.86205558, 4.05148408],
                                 [0.88128142, 4.97962558, 4.70576408],
                                 [0.54706142, 3.98752558, 5.07954408],
                                 [1.00059142, 5.65644558, 5.57669408],
                                 [0.12252142, 5.42099558, 4.02373408],
                                 [-1.91641858, 3.66563558, 1.67252408],
                                 [-1.84211858, 3.26893558, 2.55711408]])

    assert np.allclose(reactant.get_positions(), reactant_expected)
    assert np.allclose(product.get_positions(), product_expected)
