import ase
from ase.build import separate

import numpy as np
import pytest

from Src.main import *


@pytest.fixture()
def ester_hydrolysis_reaction():
    numbers = np.array([6, 1, 6, 1, 1, 8, 8, 6, 1, 1, 1, 8, 1])
    positions = np.array([[-0.98311, 3.24056, 3.04904],
                          [-0.65912, 2.21241, 3.22234],
                          [-0.1366, 3.89685, 1.97282],
                          [-2.03143, 3.21908, 2.74231],
                          [-0.91353, 3.79804, 3.9853],
                          [-0.6342, 4.3413, 0.95941],
                          [1.19922, 3.98387, 2.19841],
                          [1.83754, 3.46194, 3.3907],
                          [2.90943, 3.66713, 3.32774],
                          [1.4297, 3.95553, 4.27688],
                          [1.68243, 2.3814, 3.45031],
                          [-0.92797, -0.55259, 0.37855],
                          [-0.80599, -0.50696, 1.31958]])
    cell = np.zeros((3, 3))
    pbc = np.array([False, False, False])

    reactant = ase.Atoms.fromdict({'numbers': numbers, 'positions': positions, 'cell': cell, 'pbc': pbc})

    positions = np.array([[-4.01608, 0.23907, 0.06919],
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

    product = ase.Atoms.fromdict({'numbers': numbers, 'positions': positions, 'cell': cell, 'pbc': pbc})

    return reactant, product


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


def test_get_indices():
    arr = np.array(list(range(10)))
    values = [1, 5, 9]

    expected = np.array([1, 5, 9])
    result = get_indices(arr, values)

    assert np.all(result == expected)


def test_get_bond_forming_atoms(ester_hydrolysis_reaction):
    reactant, product = ester_hydrolysis_reaction

    ester, oh = separate_molecules(reactant)
    acid, ome = separate_molecules(product)

    matrix = get_reactivity_matrix(reactant, product)
    print(type(matrix), matrix)
    ester_result = get_bond_forming_atoms(ester, oh, True, matrix)
    oh_result = get_bond_forming_atoms(oh, ester, True, matrix)

    acid_result = get_bond_forming_atoms(acid, ome, False, matrix)
    ome_result = get_bond_forming_atoms(ome, acid, False, matrix)

    ester_expected = np.array([2])
    oh_expected = np.array([11])
    acid_expected = np.array([2])
    ome_expected = np.array([6])

    assert np.all(ester_result == ester_expected)
    assert np.all(oh_result == oh_expected)
    assert np.all(acid_result == acid_expected)
    assert np.all(ome_result == ome_expected)


def test_get_reactive_atoms(ester_hydrolysis_reaction):
    reactant, product = ester_hydrolysis_reaction

    matrix = get_reactivity_matrix(reactant, product)
    shared = np.array([0, 1, 2, 3, 4, 5])
    result = get_reactive_atoms(shared, matrix)
    expected = np.array([2])

    assert np.all(result == expected)


def test_get_reactivity_matrix(ester_hydrolysis_reaction):
    reactant, product = ester_hydrolysis_reaction
    matrix = get_reactivity_matrix(reactant, product)

    expected = np.zeros((13, 13), dtype=np.int8)
    expected[2, 6] = -1
    expected[2, 11] = 1

    assert np.all(matrix.todense() == expected)


def test_get_shared_atoms(ester_hydrolysis_reaction):
    reactant, product = ester_hydrolysis_reaction

    ester = separate_molecules(reactant)[0]
    acid = separate_molecules(product)[0]

    result = get_shared_atoms(ester, acid)
    expected = np.array([0, 1, 2, 3, 4, 5])

    assert np.all(result == expected)


def test_separate_molecules(ester_hydrolysis_reaction):
    reactant, product = ester_hydrolysis_reaction

    reactant_expected = separate(reactant)
    reactant_expected[0].set_tags([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    reactant_expected[1].set_tags([11, 12])

    product_expected = separate(product)
    product_expected[0].set_tags([0, 1, 2, 3, 4, 5, 11, 12])
    product_expected[1].set_tags([6, 7, 8, 9, 10])

    reactant_result = separate_molecules(reactant)
    product_result = separate_molecules(product)

    assert reactant_result == reactant_expected
    assert product_result == product_expected
