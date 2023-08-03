import pytest

import ase
from ase.build import separate
import numpy as np
from numpy.testing import assert_allclose

from Src.common_functions import *
from Src.common_functions import _separate_molecules_using_connectivity, _separate_molecules_using_list
from Src.stage2 import HardSphereCalculator

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


@pytest.mark.parametrize(['force_constant', 'trial_constants', 'expected'],
                         [(1.0, 10.0, np.array(list(range(1, 10)), dtype=float)),
                          (1.0, 0.0, np.array([], dtype=np.int32)),
                          (1.0, (10.0,), np.array(list(range(1, 10)), dtype=float)),
                          (1.0, (2.0, 10.0), np.array(list(range(2, 10)), dtype=float)),
                          (1.0, (2.0, 10.0, 2.0), np.array([2., 4., 6., 8.])),
                          (20.0, (1., 2., 3., 4.), np.array([1., 2., 3., 4.])),
                          (20.0, [1., 2., 3., 4.], np.array([1., 2., 3., 4.])),
                          (20.0, np.array([1., 2., 3., 4.]), np.array([1., 2., 3., 4.])),
                          (20.0, [1.], np.array([1.]))],
                         ids=['float', 'invalid_float', '1-tuple', '2-tuple', '3-tuple', 'multi-tuple', 'list',
                              'array', 'short_list'])
def test_optimise_system_create_trial_constants(force_constant, trial_constants, expected):
    result = optimise_system_create_trial_constants(force_constant, trial_constants)
    assert np.all(result == expected)


def test_simple_optimise_structure_zero(ester_hydrolysis_reaction):
    reactant, product = ester_hydrolysis_reaction
    reactant_expected, product_expected = reactant.get_positions(), product.get_positions()

    calc = HardSphereCalculator([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12]], 1.0)
    simple_optimise_structure(reactant, calc, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12]])

    calc = HardSphereCalculator([[0, 1, 2, 3, 4, 5, 11, 12], [6, 7, 8, 9, 10]], 1.0)
    simple_optimise_structure(product, calc, [[0, 1, 2, 3, 4, 5, 11, 12], [6, 7, 8, 9, 10]])

    assert np.allclose(reactant.get_positions(), reactant_expected)
    assert np.allclose(product.get_positions(), product_expected)


def test_simple_optimise_structure_nonzero_failure(overlapping_system):
    initial_coordinates = overlapping_system.get_positions()

    molecules = [list(range(14)), [14, 15, 16], [17, 18, 19, 20], list(range(21, 44)), [44, 45, 46], [47, 48, 49]]
    calc = HardSphereCalculator(molecules, 1.0)
    result = simple_optimise_structure(overlapping_system, calc, molecules)

    assert_allclose(overlapping_system.get_positions(), initial_coordinates)
    assert result is None


def test_simple_optimise_structure_nonzero_success(overlapping_system, overlapping_system_after_overlaps_fixed_alone):
    initial_coordinates = overlapping_system.get_positions()

    # TODO: Mock HardSphereCalculator
    molecules = [list(range(14)), [14, 15, 16], [17, 18, 19, 20], list(range(21, 44)), [44, 45, 46], [47, 48, 49]]
    calc = HardSphereCalculator(molecules, 4.0)
    result = simple_optimise_structure(overlapping_system, calc, molecules)

    expected = overlapping_system_after_overlaps_fixed_alone

    assert_allclose(overlapping_system.get_positions(), initial_coordinates)
    assert_allclose(result, expected)


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


@pytest.fixture()
def overlapping_system_after_overlaps_fixed_alone():
    return np.array([[3.46572839e+00, -3.96741288e+00, 4.98924636e-01],
                     [3.95084839e+00, -4.60849288e+00, 1.20502464e+00],
                     [4.11695839e+00, -3.79494288e+00, -3.32375364e-01],
                     [2.56478839e+00, -4.43168288e+00, 1.55904636e-01],
                     [3.23033839e+00, -3.03453288e+00, 9.67144636e-01],
                     [4.35235839e+00, -4.72781288e+00, -8.00595364e-01],
                     [3.63184839e+00, -3.15385288e+00, -1.03847536e+00],
                     [5.01789839e+00, -3.33067288e+00, 1.06446361e-02],
                     [5.50300839e+00, -3.97176288e+00, 7.16744636e-01],
                     [5.66911839e+00, -3.15820288e+00, -8.20655364e-01],
                     [4.78249839e+00, -2.39780288e+00, 4.78864636e-01],
                     [5.90451839e+00, -4.09108288e+00, -1.28887536e+00],
                     [5.18400839e+00, -2.51712288e+00, -1.52675536e+00],
                     [6.57005839e+00, -2.69393288e+00, -4.77625364e-01],
                     [2.08015141e-01, -2.36868029e-01, -1.93611489e+00],
                     [-5.02484859e-01, -8.35418029e-01, -2.13468489e+00],
                     [3.29995141e-01, -1.91238029e-01, -9.95084887e-01],
                     [-4.16536791e+00, -9.07581110e-03, -3.99842960e+00],
                     [-4.25454791e+00, -6.34605811e-01, -4.81184960e+00],
                     [-5.07649791e+00, 7.00541889e-02, -3.52462960e+00],
                     [-3.47013791e+00, -3.94545811e-01, -3.34346960e+00],
                     [-1.27338814e+00, 5.06312287e+00, -8.76557150e-01],
                     [-8.09048138e-01, 5.72693287e+00, -1.77517150e-01],
                     [-2.25525814e+00, 5.41944287e+00, -1.10864715e+00],
                     [-6.88328138e-01, 5.02136287e+00, -1.77146715e+00],
                     [-1.34090814e+00, 4.08476287e+00, -4.48597150e-01],
                     [-1.92596814e+00, 4.12652287e+00, 4.46312850e-01],
                     [-1.80523814e+00, 3.42096287e+00, -1.14763715e+00],
                     [-3.59038138e-01, 3.72844287e+00, -2.16507150e-01],
                     [-6.20808138e-01, 5.99973287e+00, -2.19942715e+00],
                     [2.93541862e-01, 4.66505287e+00, -1.53937715e+00],
                     [-1.15266814e+00, 4.35756287e+00, -2.47050715e+00],
                     [-1.56478138e-01, 6.66353287e+00, -1.50038715e+00],
                     [-3.57481376e-02, 5.95797287e+00, -3.09433715e+00],
                     [-1.60267814e+00, 6.35604287e+00, -2.43151715e+00],
                     [-2.90783814e+00, 4.48283287e+00, 2.14222850e-01],
                     [-1.99348814e+00, 3.14816287e+00, 8.74272850e-01],
                     [-1.46162814e+00, 4.79032287e+00, 1.14535285e+00],
                     [-2.45781814e+00, 2.48435287e+00, 1.75232850e-01],
                     [-1.01161814e+00, 2.79184287e+00, 1.10636285e+00],
                     [-2.57854814e+00, 3.18991287e+00, 1.76918285e+00],
                     [-4.26558138e-01, 2.75008287e+00, 2.11452850e-01],
                     [-5.47278138e-01, 3.45564287e+00, 1.80540285e+00],
                     [-1.07913814e+00, 1.81348287e+00, 1.53431285e+00],
                     [-2.89095554e+00, -4.98821654e-02, 1.69329767e+00],
                     [-3.18409554e+00, -9.27952165e-01, 1.90675767e+00],
                     [-3.37034554e+00, 5.80357835e-01, 2.21816767e+00],
                     [5.24394000e+00, -6.51780000e-01, -3.14968000e+00],
                     [5.93690000e+00, -4.69000000e-03, -3.08987000e+00],
                     [4.64022000e+00, -4.06260000e-01, -3.84087000e+00]])
