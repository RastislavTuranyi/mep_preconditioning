import ase

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.sparse import dok_matrix

from Src.stage4 import *
from Src.common_functions import ConvergenceError

from Tests.common_fixtures import *


def test_reposition_reactants_ester_hydrolysis(set_up_separate_molecules):
    reactant, product, reactant_indices, product_indices, _, _, reactivity_matrix = set_up_separate_molecules
    expected_product = product.get_positions()

    reposition_reactants(reactant, reactant_indices, product, product_indices, reactivity_matrix, trial_constants=(10., 10., 100.))

    print(repr(reactant.get_positions()))

    assert_allclose(product.get_positions(), expected_product)
    assert False


class TestBondFormingCalculator:
    def test_init(self):
        calc1 = BondFormingCalculator([[1, 2], [3, 4, 5]], dok_matrix(np.array([[1, 0], [1, 0]])), 5.0)
        calc2 = BondFormingCalculator([[1, 2], [3, 4, 5]], dok_matrix(np.array([[1, 0], [1, 0]])))

        # Custom to the BondFormingCalculator
        assert calc1.molecules == [[1, 2], [3, 4, 5]]
        assert np.all(calc1.reactivity_matrix.toarray() == np.array([[1, 0], [1, 0]]))
        assert calc1.force_constant == 5.0
        assert calc2.force_constant == 1.0

        # Default ASE
        assert calc1.results == {}
        assert calc1.atoms is None
        assert calc1.directory == '.'
        assert calc1.prefix is None
        assert calc1.label is None

    def test_compute_forces_ester_hydrolysis(self, set_up_separate_molecules):
        reactant, _, reactant_indices, _, reactant_molecules, _, reactivity_matrix = set_up_separate_molecules

        calc = BondFormingCalculator(reactant_indices, reactivity_matrix, 1.0)
        calc.atoms = reactant
        forces = calc.compute_forces(reactant_molecules)

        expected = np.array([[1.69360273, -0.00941431, -0.81440283], [-0.68571833, 0.09170564, 0.08443875]])

        assert_allclose(forces, expected, rtol=0, atol=1e-7)

    def test_compute_forces_one_molecule(self, one_molecule_breakdown):
        reactant, _, reactant_indices, _, _, _, reactivity_matrix = one_molecule_breakdown

        calc = BondFormingCalculator(reactant_indices, reactivity_matrix, 1.0)
        calc.atoms = reactant
        forces = calc.compute_forces([ase.Atoms()])

        expected = np.array([[0., 0., 0.]])

        assert_allclose(forces, expected, rtol=0, atol=1e-7)

    def test_compute_forces_complex_system(self, overlapping_system_reactive):
        reactant, reactant_indices, reactivity_matrix = overlapping_system_reactive

        calc = BondFormingCalculator(reactant_indices, reactivity_matrix, 1.0)
        calc.atoms = reactant
        forces = calc.compute_forces([ase.Atoms() for _ in range(6)])

        expected = np.array([[-0.05139756, 0.09309322, -0.09272658],
                             [-0.00962594, -0.02851363, 0.12039113],
                             [-0.00537344, -0.04389142, -0.13490337],
                             [-0.88858011, -0.9154377, -1.48577367],
                             [0., 0., 0.],
                             [0., 0., 0.]])

        assert_allclose(forces, expected, rtol=0, atol=1e-7)

    @pytest.mark.parametrize(['mol', 'expected_trans', 'expected_rot'],
                             [(0, np.array([-0.00023171, -0.00130276, -0.00046679]),
                               np.array([0.15396388, -0.00085585, -0.07403662])),
                              (1, np.array([0.02476786, 0.13925611, 0.04989658]),
                               np.array([-0.34285917, 0.04585282, 0.04221937]))],
                             ids=['molecule1', 'molecule2'])
    def test_compute_vectors_ester_hydrolysis(self, set_up_separate_molecules, mol, expected_trans, expected_rot):
        reactant, _, reactant_indices, _, reactant_molecules, _, reactivity_matrix = set_up_separate_molecules

        calc = BondFormingCalculator(reactant_indices, reactivity_matrix, 1.0)

        coordinates = reactant.get_positions()
        translational_vector, rotational_vector = calc.compute_vectors(coordinates, reactant_indices[mol], mol,
                                                                       np.mean(coordinates[reactant_indices[mol]],
                                                                               axis=0))

        assert_allclose(translational_vector, expected_trans, rtol=0, atol=1e-7)
        assert_allclose(rotational_vector, expected_rot, rtol=0, atol=1e-7)

    def test_compute_vectors_one_molecule(self, one_molecule_breakdown):
        reactant, _, reactant_indices, _, _, _, reactivity_matrix = one_molecule_breakdown

        calc = BondFormingCalculator(reactant_indices, reactivity_matrix, 1.0)

        coordinates = reactant.get_positions()
        translational_vector, rotational_vector = calc.compute_vectors(coordinates, reactant_indices[0], 0,
                                                                       np.mean(coordinates[reactant_indices[0]],
                                                                               axis=0))

        assert_allclose(translational_vector, np.zeros(3), rtol=0, atol=1e-7)
        assert_allclose(rotational_vector, np.zeros(3), rtol=0, atol=1e-7)

    @pytest.mark.parametrize(['mol', 'expected_trans', 'expected_rot'],
                             [(0, np.array([0.0259728, -0.01670866, -0.0002968]),
                               np.array([-0.00367125, 0.00664952, -0.00662333])),
                              (1, np.array([-0.0201771, 0.0391812, 0.00737736]),
                               np.array([-0.00320865, -0.00950454, 0.04013038])),
                              (2, np.array([-0.00052866, 0.01127499, 0.00023538]),
                               np.array([-0.00134336, -0.01097285, -0.03372584])),
                              (3, np.array([-0.02446602, 0.02615294, -0.00148163]),
                               np.array([-0.03863392, -0.03980164, -0.06459886])),
                              (4, np.array([0., 0., 0.]),
                               np.array([0., 0., 0.])),
                              (5, np.array([0., 0., 0.]),
                               np.array([0., 0., 0.]))],
                             ids=[f'mol{i}' for i in range(6)]
                             )
    def test_compute_vectors_large_system(self, overlapping_system_reactive, mol, expected_trans, expected_rot):
        reactant, reactant_indices, reactivity_matrix = overlapping_system_reactive

        calc = BondFormingCalculator(reactant_indices, reactivity_matrix, 1.0)

        coordinates = reactant.get_positions()
        translational_vector, rotational_vector = calc.compute_vectors(coordinates, reactant_indices[mol], mol,
                                                                       np.mean(coordinates[reactant_indices[mol]],
                                                                               axis=0))

        assert_allclose(translational_vector, expected_trans, rtol=0, atol=1e-7)
        assert_allclose(rotational_vector, expected_rot, rtol=0, atol=1e-7)


class TestCorrelatedPlacementCalculator:
    def test_init(self):
        calc1 = CorrelatedPlacementCalculator([[1, 2], [3, 4, 5]], [[4, 5], [1, 2, 3]], ase.Atoms(),
                                              dok_matrix(np.array([[1, 0], [1, 0]])), 5.0)
        calc2 = CorrelatedPlacementCalculator([[1, 2], [3, 4, 5]], [[4, 5], [1, 2, 3]], ase.Atoms(),
                                              dok_matrix(np.array([[1, 0], [1, 0]])))

        # Custom to the BondFormingCalculator
        assert calc1.molecules == [[1, 2], [3, 4, 5]]
        assert calc1.product_molecules == [[4, 5], [1, 2, 3]]
        assert np.all(calc1.product_coordinates == np.zeros((0, 3)))
        assert np.all(calc1.reactivity_matrix.toarray() == np.array([[1, 0], [1, 0]]))
        assert calc1.force_constant == 5.0
        assert calc2.force_constant == 1.0

        # Default ASE
        assert calc1.results == {}
        assert calc1.atoms is None
        assert calc1.directory == '.'
        assert calc1.prefix is None
        assert calc1.label is None

    def test_compute_forces_ester_hydrolysis(self, set_up_separate_molecules):
        reactant, product, reactant_indices, product_indices, reactant_molecules, _, reactivity_matrix = \
            set_up_separate_molecules

        calc = CorrelatedPlacementCalculator(reactant_indices, product_indices, product, reactivity_matrix, 1.0)
        calc.atoms = reactant
        forces = calc.compute_forces(reactant_molecules)

        expected = np.array([[1.309970e-01, 1.624020e-01, -3.022588e-01], [1.387779e-17, 4.625929e-18, 1.156482e-18]])

        assert_allclose(forces, expected, rtol=0, atol=1e-7)

    def test_compute_forces_one_molecule(self, one_molecule_breakdown):
        reactant, product, reactant_indices, product_indices, _, _, reactivity_matrix = one_molecule_breakdown

        calc = CorrelatedPlacementCalculator(reactant_indices, product_indices, product, reactivity_matrix, 1.0)
        calc.atoms = reactant
        forces = calc.compute_forces([ase.Atoms()])

        expected = np.array([[0.04926947, -0.03127423, 0.11394098]])

        assert_allclose(forces, expected, rtol=0, atol=1e-7)

    @pytest.mark.parametrize(['mol', 'expected_trans', 'expected_rot'],
                             [(0, np.array([-0.0290572, -0.02685846, -0.02710734]),
                               np.array([0.01190882, 0.01476381, -0.02747807])),
                              (1, np.array([-0.0179672, 0.02569202, -0.02230917]),
                               np.array([6.93889390e-18, 2.31296463e-18, 5.78241159e-19]))],
                             ids=['molecule1', 'molecule2'])
    def test_compute_vectors_ester_hydrolysis(self, set_up_separate_molecules, mol, expected_trans, expected_rot):
        reactant, product, reactant_indices, product_indices, _, _, reactivity_matrix = set_up_separate_molecules

        calc = CorrelatedPlacementCalculator(reactant_indices, product_indices, product, reactivity_matrix, 1.0)

        coordinates = reactant.get_positions()
        translational_vector, rotational_vector = calc.compute_vectors(coordinates, reactant_indices[mol],
                                                                       np.mean(coordinates[reactant_indices[mol]],
                                                                               axis=0))

        assert_allclose(translational_vector, expected_trans, rtol=0, atol=1e-7)
        assert_allclose(rotational_vector, expected_rot, rtol=0, atol=1e-7)

    def test_compute_vectors_one_molecule(self, one_molecule_breakdown):
        reactant, product, reactant_indices, product_indices, _, _, reactivity_matrix = one_molecule_breakdown

        calc = CorrelatedPlacementCalculator(reactant_indices, product_indices, product, reactivity_matrix, 1.0)

        coordinates = reactant.get_positions()
        translational_vector, rotational_vector = calc.compute_vectors(coordinates, reactant_indices[0],
                                                                       np.mean(coordinates[reactant_indices[0]],
                                                                               axis=0))

        assert_allclose(translational_vector, np.array([-0.00926627, -0.00860095, 0.00120452]), rtol=0, atol=1e-7)
        assert_allclose(rotational_vector, np.array([0.00205289, -0.00130309, 0.00474754]), rtol=0, atol=1e-7)
