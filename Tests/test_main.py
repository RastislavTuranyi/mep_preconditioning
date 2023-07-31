import ase
from ase.build import separate
from ase.io import write

import numpy as np
from numpy.testing import assert_allclose

import pytest

from Src.main import *
from Src.main import _separate_molecules_using_connectivity, _separate_molecules_using_list


def restructure(*args):
    result = []

    for vals in zip(*args):
        result.append(vals)

    return result


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


@pytest.fixture()
def set_up_separate_molecules(ester_hydrolysis_reaction) \
        -> tuple[ase.Atoms, ase.Atoms, list[list[int]], list[list[int]], list[ase.Atoms], list[ase.Atoms]]:
    reactant, product = ester_hydrolysis_reaction

    reactant_indices = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12]]
    product_indices = [[0, 1, 2, 3, 4, 5, 11, 12], [6, 7, 8, 9, 10]]

    reactant_expected = separate(reactant)
    reactant_expected[0].set_tags(reactant_indices[0])
    reactant_expected[1].set_tags(reactant_indices[1])

    product_expected = separate(product)
    product_expected[0].set_tags(product_indices[0])
    product_expected[1].set_tags(product_indices[1])

    return reactant, product, reactant_indices, product_indices, reactant_expected, product_expected


@pytest.fixture()
def overlapping_system():
    numbers = np.array([6, 1, 6, 1, 1, 1, 1, 6, 1, 6, 1, 1, 1, 1, 8, 1, 1, 7, 1, 1, 1, 6, 1, 1, 6, 6, 6, 1, 1, 6, 1, 1,
                        1, 1, 1, 1, 6, 1, 1, 6, 1, 1, 1, 1, 8, 1, 1, 8, 1, 1])

    positions = np.array([[-2.76484e+00, 5.82650e-01, -0.00000e+00],
                          [-2.27972e+00, -5.84300e-02, 7.06100e-01],
                          [-2.11361e+00, 7.55120e-01, -8.31300e-01],
                          [-3.66578e+00, 1.18380e-01, -3.43020e-01],
                          [-3.00023e+00, 1.51553e+00, 4.68220e-01],
                          [-1.87821e+00, -1.77750e-01, -1.29952e+00],
                          [-2.59872e+00, 1.39621e+00, -1.53740e+00],
                          [-1.21267e+00, 1.21939e+00, -4.88280e-01],
                          [-7.27560e-01, 5.78300e-01, 2.17820e-01],
                          [-5.61450e-01, 1.39186e+00, -1.31958e+00],
                          [-1.44807e+00, 2.15226e+00, -2.00600e-02],
                          [-3.26050e-01, 4.58980e-01, -1.78780e+00],
                          [-1.04656e+00, 2.03294e+00, -2.02568e+00],
                          [3.39490e-01, 1.85613e+00, -9.76550e-01],
                          [-1.54803e+00, -2.75300e-01, -5.01430e-01],
                          [-2.25853e+00, -8.73850e-01, -7.00000e-01],
                          [-1.42605e+00, -2.29670e-01, 4.39600e-01],
                          [3.07850e-01, 7.10640e-01, -1.01636e+00],
                          [2.18670e-01, 8.51100e-02, -1.82978e+00],
                          [-6.03280e-01, 7.89770e-01, -5.42560e-01],
                          [1.00308e+00, 3.25170e-01, -3.61400e-01],
                          [1.41980e-01, -6.36970e-01, -5.19270e-01],
                          [6.06320e-01, 2.68400e-02, 1.79770e-01],
                          [-8.39890e-01, -2.80650e-01, -7.51360e-01],
                          [7.27040e-01, -6.78730e-01, -1.41418e+00],
                          [7.44600e-02, -1.61533e+00, -9.13100e-02],
                          [-5.10600e-01, -1.57357e+00, 8.03600e-01],
                          [-3.89870e-01, -2.27913e+00, -7.90350e-01],
                          [1.05633e+00, -1.97165e+00, 1.40780e-01],
                          [7.94560e-01, 2.99640e-01, -1.84214e+00],
                          [1.70891e+00, -1.03504e+00, -1.18209e+00],
                          [2.62700e-01, -1.34253e+00, -2.11322e+00],
                          [1.25889e+00, 9.63440e-01, -1.14310e+00],
                          [1.37962e+00, 2.57880e-01, -2.73705e+00],
                          [-1.87310e-01, 6.55950e-01, -2.07423e+00],
                          [-1.49247e+00, -1.21726e+00, 5.71510e-01],
                          [-5.78120e-01, -2.55193e+00, 1.23156e+00],
                          [-4.62600e-02, -9.09770e-01, 1.50264e+00],
                          [-1.04245e+00, -3.21574e+00, 5.32520e-01],
                          [4.03750e-01, -2.90825e+00, 1.46365e+00],
                          [-1.16318e+00, -2.51018e+00, 2.12647e+00],
                          [9.88810e-01, -2.95001e+00, 5.68740e-01],
                          [8.68090e-01, -2.24445e+00, 2.16269e+00],
                          [3.36230e-01, -3.88661e+00, 1.89160e+00],
                          [2.68608e+00, -2.00980e+00, -3.09040e-01],
                          [2.39294e+00, -2.88787e+00, -9.55800e-02],
                          [2.20669e+00, -1.37956e+00, 2.15830e-01],
                          [5.24394e+00, -6.51780e-01, -3.14968e+00],
                          [5.93690e+00, -4.69000e-03, -3.08987e+00],
                          [4.64022e+00, -4.06260e-01, -3.84087e+00]])

    cell = np.zeros((3, 3))
    pbc = np.array([False, False, False])

    return ase.Atoms.fromdict({'numbers': numbers, 'positions': positions, 'cell': cell, 'pbc': pbc})


@pytest.fixture()
def overlapping_system_after_overlaps_fixed():
    return np.array([[-1.04637981e+00, 2.26783462e+00, 2.52096548e-01],
                     [-5.61259807e-01, 1.62675462e+00, 9.58196548e-01],
                     [-3.95149807e-01, 2.44030462e+00, -5.79203452e-01],
                     [-1.94731981e+00, 1.80356462e+00, -9.09234523e-02],
                     [-1.28176981e+00, 3.20071462e+00, 7.20316548e-01],
                     [-1.59749807e-01, 1.50743462e+00, -1.04742345e+00],
                     [-8.80259807e-01, 3.08139462e+00, -1.28530345e+00],
                     [5.05790193e-01, 2.90457462e+00, -2.36183452e-01],
                     [9.90900193e-01, 2.26348462e+00, 4.69916548e-01],
                     [1.15701019e+00, 3.07704462e+00, -1.06748345e+00],
                     [2.70390193e-01, 3.83744462e+00, 2.32036548e-01],
                     [1.39241019e+00, 2.14416462e+00, -1.53570345e+00],
                     [6.71900193e-01, 3.71812462e+00, -1.77358345e+00],
                     [2.05795019e+00, 3.54131462e+00, -7.24453452e-01],
                     [7.00034733e-01, -9.58461788e-01, -3.91298894e-01],
                     [-1.04652674e-02, -1.55701179e+00, -5.89868894e-01],
                     [8.22014733e-01, -9.12831788e-01, 5.49731106e-01],
                     [2.31558622e+00, -2.76604179e+00, -1.09400255e+00],
                     [2.22640622e+00, -3.39157179e+00, -1.90742255e+00],
                     [1.40445622e+00, -2.68691179e+00, -6.20202551e-01],
                     [3.01081622e+00, -3.15151179e+00, -4.39042551e-01],
                     [3.71037343e+00, -3.13157914e-01, 2.81127067e+00],
                     [4.17471343e+00, 3.50652086e-01, 3.51031067e+00],
                     [2.72850343e+00, 4.31620864e-02, 2.57918067e+00],
                     [4.29543343e+00, -3.54917914e-01, 1.91636067e+00],
                     [3.64285343e+00, -1.29151791e+00, 3.23923067e+00],
                     [3.05779343e+00, -1.24975791e+00, 4.13414067e+00],
                     [3.17852343e+00, -1.95531791e+00, 2.54019067e+00],
                     [4.62472343e+00, -1.64783791e+00, 3.47132067e+00],
                     [4.36295343e+00, 6.23452086e-01, 1.48840067e+00],
                     [5.27730343e+00, -7.11227914e-01, 2.14845067e+00],
                     [3.83109343e+00, -1.01871791e+00, 1.21732067e+00],
                     [4.82728343e+00, 1.28725209e+00, 2.18744067e+00],
                     [4.94801343e+00, 5.81692086e-01, 5.93490667e-01],
                     [3.38108343e+00, 9.79762086e-01, 1.25631067e+00],
                     [2.07592343e+00, -8.93447914e-01, 3.90205067e+00],
                     [2.99027343e+00, -2.22811791e+00, 4.56210067e+00],
                     [3.52213343e+00, -5.85957914e-01, 4.83318067e+00],
                     [2.52594343e+00, -2.89192791e+00, 3.86306067e+00],
                     [3.97214343e+00, -2.58443791e+00, 4.79419067e+00],
                     [2.40521343e+00, -2.18636791e+00, 5.45701067e+00],
                     [4.55720343e+00, -2.62619791e+00, 3.89928067e+00],
                     [4.43648343e+00, -1.92063791e+00, 5.49323067e+00],
                     [3.90462343e+00, -3.56279791e+00, 5.22214067e+00],
                     [3.48545899e-01, 8.80258335e-01, -4.75137895e+00],
                     [5.54058994e-02, 2.18833488e-03, -4.53791895e+00],
                     [-1.30844101e-01, 1.51049833e+00, -4.22650895e+00],
                     [3.26492816e+00, -4.38327069e-02, -1.93118092e+00],
                     [3.95788816e+00, 6.03257293e-01, -1.87137092e+00],
                     [2.66120816e+00, 2.01687293e-01, -2.62237092e+00]])


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


def generate_molecules():
    cell = np.zeros((3, 3))
    pbc = np.array([False, False, False])

    equidistant = ase.Atoms.fromdict({'numbers': [78, 1, 1, 1, 1],
                                      'positions': [[0, 0, 0],
                                                    [1, 0, 0],
                                                    [-1, 0, 0],
                                                    [0, 1, 0],
                                                    [0, -1, 0]],
                                      'cell': cell, 'pbc': pbc})

    distorted = ase.Atoms.fromdict({'numbers': [78, 1, 1, 1, 1],
                                    'positions': [[0, 0, 0],
                                                  [2, 0, 0],
                                                  [-2, 0, 0],
                                                  [0, 1, 0],
                                                  [0, -1, 0]],
                                    'cell': cell, 'pbc': pbc})

    return equidistant, distorted


def test_determine_overlaps():
    molecules = [ase.Atoms() for _ in range(5)]
    centres = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, -10, 0]), np.array([-2.5, 0, 0]),
               np.array([-5, 0, 0])]
    radii = [1.0, 1.0, 0.5, 2.0, 0.5]

    result = determine_overlaps(molecules, centres, radii)
    expected = np.array([[0, 1, 0, 1, 0],
                         [1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]], dtype=np.int8)

    assert np.all(result == expected)


@pytest.mark.parametrize(['atoms', 'expected'], restructure(generate_molecules(), (1.6, 2.6966629547095766)))
def test_estimate_molecular_radius(atoms, expected):
    result = estimate_molecular_radius(atoms, np.mean(atoms.get_positions(), axis=0))

    assert result == expected


def test_fix_overlaps_zero(ester_hydrolysis_reaction):
    reactant, product = ester_hydrolysis_reaction

    reactant_expected, product_expected = reactant.get_positions(), product.get_positions()

    fix_overlaps(reactant, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12]])
    fix_overlaps(product, [[0, 1, 2, 3, 4, 5, 11, 12], [6, 7, 8, 9, 10]])

    assert np.allclose(reactant.get_positions(), reactant_expected)
    assert np.allclose(product.get_positions(), product_expected)


def test_fix_overlaps_nonzero(overlapping_system, overlapping_system_after_overlaps_fixed):
    fix_overlaps(overlapping_system, [list(range(14)), [14, 15, 16], [17, 18, 19, 20],
                                      list(range(21, 44)), [44, 45, 46], [47, 48, 49]])
    expected = overlapping_system_after_overlaps_fixed
    assert_allclose(overlapping_system.get_positions(), expected)


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


def test_separate_molecules_connectivity(set_up_separate_molecules):
    reactant, product, reactant_indices, product_indices, \
    reactant_expected, product_expected = set_up_separate_molecules

    reactant_result = separate_molecules(reactant, None)
    product_result = separate_molecules(product, None)

    reactant_tags = [list(mol.get_tags()) for mol in reactant_result]
    product_tags = [list(mol.get_tags()) for mol in product_result]

    assert reactant_result == reactant_expected
    assert product_result == product_expected

    assert reactant_tags == reactant_indices
    assert product_tags == product_indices


def test_separate_molecules_list(set_up_separate_molecules):
    reactant, product, reactant_indices, product_indices, \
    reactant_expected, product_expected = set_up_separate_molecules

    reactant_result = separate_molecules(reactant, reactant_indices)
    product_result = separate_molecules(product, product_indices)

    reactant_tags = [list(mol.get_tags()) for mol in reactant_result]
    product_tags = [list(mol.get_tags()) for mol in product_result]

    assert reactant_result == reactant_expected
    assert product_result == product_expected

    assert reactant_tags == reactant_indices
    assert product_tags == product_indices


def test_separate_molecules_using_connectivity(set_up_separate_molecules):
    reactant, product, reactant_indices, product_indices, \
    reactant_expected, product_expected = set_up_separate_molecules

    reactant_result = _separate_molecules_using_connectivity(reactant)
    product_result = _separate_molecules_using_connectivity(product)

    reactant_tags = [list(mol.get_tags()) for mol in reactant_result]
    product_tags = [list(mol.get_tags()) for mol in product_result]

    assert reactant_result == reactant_expected
    assert product_result == product_expected

    assert reactant_tags == reactant_indices
    assert product_tags == product_indices


def test_separate_molecules_using_list(set_up_separate_molecules):
    reactant, product, reactant_indices, product_indices, \
    reactant_expected, product_expected = set_up_separate_molecules

    reactant_result = _separate_molecules_using_list(reactant, reactant_indices)
    product_result = _separate_molecules_using_list(product, product_indices)

    reactant_tags = [list(mol.get_tags()) for mol in reactant_result]
    product_tags = [list(mol.get_tags()) for mol in product_result]

    assert reactant_result == reactant_expected
    assert product_result == product_expected

    assert reactant_tags == reactant_indices
    assert product_tags == product_indices


def test_simple_optimise_structure_zero(ester_hydrolysis_reaction):
    reactant, product = ester_hydrolysis_reaction

    reactant_expected, product_expected = reactant.get_positions(), product.get_positions()

    simple_optimise_structure(reactant, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12]])
    simple_optimise_structure(product, [[0, 1, 2, 3, 4, 5, 11, 12], [6, 7, 8, 9, 10]])

    assert np.allclose(reactant.get_positions(), reactant_expected)
    assert np.allclose(product.get_positions(), product_expected)


def test_simple_optimise_structure_nonzero_failure(overlapping_system):
    initial_coordinates = overlapping_system.get_positions()
    result = simple_optimise_structure(overlapping_system, [list(range(14)), [14, 15, 16], [17, 18, 19, 20],
                                                            list(range(21, 44)), [44, 45, 46], [47, 48, 49]])
    print(overlapping_system.get_positions())

    assert_allclose(overlapping_system.get_positions(), initial_coordinates)
    assert result is None


def test_simple_optimise_structure_nonzero_success(overlapping_system, overlapping_system_after_overlaps_fixed):
    initial_coordinates = overlapping_system.get_positions()
    result = simple_optimise_structure(overlapping_system,
                                       [list(range(14)), [14, 15, 16], [17, 18, 19, 20],
                                        list(range(21, 44)), [44, 45, 46], [47, 48, 49]],
                                       force_constant=4.0)
    expected = overlapping_system_after_overlaps_fixed
    print(overlapping_system.get_positions())

    assert_allclose(overlapping_system.get_positions(), initial_coordinates)
    assert_allclose(result, expected)


def test_constrained_bfgs_init(ester_hydrolysis_reaction):
    dyn = ConstrainedBFGS(ester_hydrolysis_reaction[0])

    # New, custom fields
    assert dyn.non_convergence_limit == 0.001
    assert dyn.non_convergence_roof == 0.5
    assert dyn._total_fmax == 0.0
    assert dyn._total_iteration == 0.0

    # Fields inherited from ASE
    assert dyn.atoms == ester_hydrolysis_reaction[0]
    assert dyn.maxstep == 0.2
    assert dyn.alpha == 70.0
    assert dyn.restart is None
    assert dyn.fmax is None


@pytest.mark.parametrize(['force_change', 'n_iter', 'expected'],
                         [([1], 1, (False, 25, 1)),
                          ([1, 2, 3, 4], 4, (False, 750, 4)),
                          ([1, 0.5, 0.0001], 3, (True, 31.25000025, 3))],
                         ids=['one_iter', 'no_convergence', 'converged'])
def test_constrained_bfgs_converged(force_change, n_iter, expected):
    dyn = ConstrainedBFGS(ase.Atoms())
    dyn.fmax = 0.05

    forces = np.array([[1, 1, 1], [0, 0, 0], [-5, 0, 0]])
    for _, change in zip(range(n_iter), force_change):
        converged = dyn.converged(forces * change)

    assert converged == expected[0]
    assert dyn._total_fmax == expected[1]
    assert dyn._total_iteration == expected[2]


def test_constrained_bfgs_converged_exception():
    dyn = ConstrainedBFGS(ase.Atoms())
    dyn.fmax = 0.05

    forces = np.array([[1, 1, 1], [0, 0, 0], [-5, 0, 0]])
    dyn.converged(forces)

    with pytest.raises(OptimisationNotConvergingError):
        dyn.converged(forces)


def test_hard_sphere_calculator_init():
    calc1 = HardSphereCalculator([[1, 2], [3, 4, 5]], 5.0)
    calc2 = HardSphereCalculator([[1, 2], [3, 4, 5]])

    # Custom to the HardSphereCalculator
    assert calc1.molecules == [[1, 2], [3, 4, 5]]
    assert calc1.force_constant == 5.0
    assert calc2.force_constant == 1.0

    # Default ASE
    assert calc1.results == {}
    assert calc1.atoms is None
    assert calc1.directory == '.'
    assert calc1.prefix is None
    assert calc1.label is None


def test_hard_sphere_calculator_calculate_zero(ester_hydrolysis_reaction):
    reactant, product = ester_hydrolysis_reaction

    reactant_calc = HardSphereCalculator([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12]])
    product_calc = HardSphereCalculator([[0, 1, 2, 3, 4, 5, 11, 12], [6, 7, 8, 9, 10]])

    reactant_calc.calculate(atoms=reactant)
    product_calc.calculate(atoms=product)

    expected = np.zeros((13, 3))

    assert np.all(reactant_calc.results['forces'] == expected)
    assert np.all(product_calc.results['forces'] == expected)

    assert reactant_calc.results['energy'] == 0.0
    assert product_calc.results['energy'] == 0.0


def test_hard_sphere_calculator_calculate_nonzero(overlapping_system):
    molecule_atom_indices = [list(range(14)), [14, 15, 16], [17, 18, 19, 20],
                             list(range(21, 44)), [44, 45, 46], [47, 48, 49]]
    calc = HardSphereCalculator(molecule_atom_indices)
    calc.calculate(atoms=overlapping_system)

    expected = [[3.85357690e-01, -5.19467374e-01, 9.35067787e-02]] * 14
    expected.extend([[2.67685739e-01, 1.13410846e-01, -4.38364673e-02]] * 3)
    expected.extend([[-2.27744444e-01, -2.38339808e-01, 1.70179298e-01]] * 4)
    expected.extend([[-1.46625980e-01, 4.28006866e-01, -1.64768804e-01]] * 23)
    expected.extend([[-6.89393035e-01, 2.21161542e-01, -4.73610277e-04]] * 3)
    expected.extend([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00]] * 3)
    expected = np.array(expected)

    assert np.allclose(calc.results['forces'], expected)
    assert calc.results['energy'] == 0.0


def test_compute_hard_sphere_forces_zero(ester_hydrolysis_reaction):
    reactant, product = ester_hydrolysis_reaction

    reactant.set_positions(np.array([[-1.05141, 5.188985, 4.03545],
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
                                     [-0.87429, 1.441465, 2.30599]]))

    product.set_positions(np.array([[-4.03730858, 2.81617558, 2.27066408],
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
                                    [-1.84211858, 3.26893558, 2.55711408]]))

    reactant_molecules = separate(reactant)
    product_molecules = separate(product)

    calc = HardSphereCalculator([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12]])
    reactant_result = calc.compute_hard_sphere_forces(reactant_molecules)

    calc.molecules = [[0, 1, 2, 3, 4, 5, 11, 12], [6, 7, 8, 9, 10]]
    product_result = calc.compute_hard_sphere_forces(product_molecules)

    expected = np.zeros((2, 3))

    assert np.all(reactant_result == expected)
    assert np.all(product_result == expected)


def test_compute_hard_sphere_forces_clustered(overlapping_system):
    molecule_atom_indices = [list(range(14)), [14, 15, 16], [17, 18, 19, 20],
                             list(range(21, 44)), [44, 45, 46], [47, 48, 49]]
    separated = separate_molecules(overlapping_system, molecule_atom_indices)
    calc = HardSphereCalculator(molecule_atom_indices)

    result = calc.compute_hard_sphere_forces(separated)
    expected = np.array([[3.85357690e-01, -5.19467374e-01, 9.35067787e-02],
                         [2.67685739e-01, 1.13410846e-01, -4.38364673e-02],
                         [-2.27744444e-01, -2.38339808e-01, 1.70179298e-01],
                         [-1.46625980e-01, 4.28006866e-01, -1.64768804e-01],
                         [-6.89393035e-01, 2.21161542e-01, -4.73610277e-04],
                         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

    assert np.allclose(result, expected)
