import ase
from ase.build import separate

import numpy as np
from numpy.testing import assert_allclose
import pytest

from Src.stage2 import *
from Src.common_functions import ConvergenceError

from Tests.common_fixtures import restructure, ester_hydrolysis_reaction, overlapping_system


@pytest.fixture()
def overlapping_system_after_overlaps_fixed_bfgs():
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
    fix_overlaps(product, [[0, 1, 2, 3, 4, 5, 11, 12], [6, 7, 8, 9, 10]], non_convergence_roof=None)

    assert_allclose(reactant.get_positions(), reactant_expected)
    assert_allclose(product.get_positions(), product_expected)


def test_fix_overlaps_nonzero(overlapping_system, overlapping_system_after_overlaps_fixed_bfgs):
    fix_overlaps(overlapping_system, [list(range(14)), [14, 15, 16], [17, 18, 19, 20],
                                      list(range(21, 44)), [44, 45, 46], [47, 48, 49]])
    expected = overlapping_system_after_overlaps_fixed_bfgs
    assert_allclose(overlapping_system.get_positions(), expected)


def test_fix_overlaps_error_no_trial_constants(overlapping_system):
    with pytest.raises(ConvergenceError):
        fix_overlaps(overlapping_system, [list(range(14)), [14, 15, 16], [17, 18, 19, 20],
                                          list(range(21, 44)), [44, 45, 46], [47, 48, 49]],
                     trial_constants=None)


def test_fix_overlaps_error_no_convergence(overlapping_system):
    with pytest.raises(ConvergenceError):
        fix_overlaps(overlapping_system, [list(range(14)), [14, 15, 16], [17, 18, 19, 20],
                                          list(range(21, 44)), [44, 45, 46], [47, 48, 49]],
                     trial_constants=(1., 3.))


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


def test_compute_forces_zero(ester_hydrolysis_reaction):
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
    reactant_result = calc.compute_forces(reactant_molecules)

    calc.molecules = [[0, 1, 2, 3, 4, 5, 11, 12], [6, 7, 8, 9, 10]]
    product_result = calc.compute_forces(product_molecules)

    expected = np.zeros((2, 3))

    assert np.all(reactant_result == expected)
    assert np.all(product_result == expected)


def test_compute_forces_clustered(overlapping_system):
    molecule_atom_indices = [list(range(14)), [14, 15, 16], [17, 18, 19, 20],
                             list(range(21, 44)), [44, 45, 46], [47, 48, 49]]
    separated = separate_molecules(overlapping_system, molecule_atom_indices)
    calc = HardSphereCalculator(molecule_atom_indices)

    result = calc.compute_forces(separated)
    expected = np.array([[3.85357690e-01, -5.19467374e-01, 9.35067787e-02],
                         [2.67685739e-01, 1.13410846e-01, -4.38364673e-02],
                         [-2.27744444e-01, -2.38339808e-01, 1.70179298e-01],
                         [-1.46625980e-01, 4.28006866e-01, -1.64768804e-01],
                         [-6.89393035e-01, 2.21161542e-01, -4.73610277e-04],
                         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

    assert np.allclose(result, expected)
