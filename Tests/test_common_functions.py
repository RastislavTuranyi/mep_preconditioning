import pytest

import ase
from ase.build import separate
import numpy as np
from numpy.testing import assert_allclose

from Src.common_functions import *
from Src.common_functions import _separate_molecules_using_connectivity, _separate_molecules_using_list

from Tests.common_fixtures import *


def prepare_parameters_for_test_alpha_vector_simple() -> list[tuple[np.ndarray, np.ndarray]]:
    reactant, product, _, _, reactant_molecules, product_molecules, _ = set_up_separate_molecules_wrapped()
    matrix = get_reactivity_matrix(reactant, product)

    reactant_alpha1 = compute_alpha_vector(reactant.get_positions(), 0, reactant_molecules, True, matrix)
    reactant_alpha2 = compute_alpha_vector(reactant.get_positions(), 1, reactant_molecules, True, matrix)

    product_alpha1 = compute_alpha_vector(product.get_positions(), 0, product_molecules, False, matrix)
    product_alpha2 = compute_alpha_vector(product.get_positions(), 1, product_molecules, False, matrix)

    expected = [np.array([-0.0683, 1.948425, 0.98641]), np.array([-0.463985, -0.276295, 0.189275]),
                np.array([-1.56549,  0.47438, -0.46501]), np.array([1.066335, 1.142475, 0.925005])]

    return [(reactant_alpha1, expected[0]), (reactant_alpha2, expected[1]), (product_alpha1, expected[2]),
            (product_alpha2, expected[3])]


@pytest.mark.parametrize(['result', 'expected'],
                         prepare_parameters_for_test_alpha_vector_simple(),
                         ids=['reactant1', 'reactant2', 'product1', 'product2'])
def test_compute_alpha_vector_simple(result, expected):
    print(repr(result))
    assert_allclose(result, expected)


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


vals = [(0, 0), (0, 1), (1, 0), (1, 1)]


@pytest.mark.parametrize(['mol1', 'mol2', 'indices', 'expected'],
                         [(0, 0, False, [0, 1, 2, 3, 4, 5]),
                          (0, 1, False, [6, 7, 8, 9, 10]),
                          (1, 0, False, [11, 12]),
                          (1, 1, False, []),
                          (0, 0, True, [0, 1, 2, 3, 4, 5]),
                          (0, 1, True, [6, 7, 8, 9, 10]),
                          (1, 0, True, [11, 12]),
                          (1, 1, True, [])],
                         ids=[f'mols({i},{j})' for i, j in vals] + [f'indices({i},{j})' for i, j in vals])
def test_get_shared_atoms(set_up_separate_molecules, mol1, mol2, indices, expected):
    _, _, reactant_indices, product_indices, reactant_molecules, product_molecules, _ = set_up_separate_molecules

    reactant = reactant_indices if indices else reactant_molecules
    product = product_indices if indices else product_molecules

    result = get_shared_atoms(reactant[mol1], product[mol2])

    assert np.all(result == expected)


def test_separate_molecules_connectivity(set_up_separate_molecules):
    reactant, product, reactant_indices, product_indices, \
        reactant_expected, product_expected, _ = set_up_separate_molecules

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
        reactant_expected, product_expected, _ = set_up_separate_molecules

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
        reactant_expected, product_expected, _ = set_up_separate_molecules

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
        reactant_expected, product_expected, _ = set_up_separate_molecules

    reactant_result = _separate_molecules_using_list(reactant, reactant_indices)
    product_result = _separate_molecules_using_list(product, product_indices)

    reactant_tags = [list(mol.get_tags()) for mol in reactant_result]
    product_tags = [list(mol.get_tags()) for mol in product_result]

    assert reactant_result == reactant_expected
    assert product_result == product_expected

    assert reactant_tags == reactant_indices
    assert product_tags == product_indices
