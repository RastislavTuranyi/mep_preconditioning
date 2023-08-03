import ase
from ase.build import separate
import pytest
import numpy as np
from scipy.sparse import dok_matrix


def restructure(*args):
    result = []

    for vals in zip(*args):
        result.append(vals)

    return result


@pytest.fixture()
def ester_hydrolysis_reaction():
    return ester_hydrolysis_wrapped()


def ester_hydrolysis_wrapped():
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
def set_up_separate_molecules(ester_hydrolysis_reaction) \
        -> tuple[ase.Atoms, ase.Atoms, list[list[int]], list[list[int]], list[ase.Atoms], list[ase.Atoms], dok_matrix]:
    # TODO: Look for a less cursed way to do this
    return set_up_separate_molecules_wrapped(ester_hydrolysis_reaction)


def set_up_separate_molecules_wrapped(ester_hydrolysis_reaction=None) \
        -> tuple[ase.Atoms, ase.Atoms, list[list[int]], list[list[int]], list[ase.Atoms], list[ase.Atoms], dok_matrix]:
    if ester_hydrolysis_reaction is None:
        reactant, product = ester_hydrolysis_wrapped()
    else:
        reactant, product = ester_hydrolysis_reaction

    reactant_indices = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12]]
    product_indices = [[0, 1, 2, 3, 4, 5, 11, 12], [6, 7, 8, 9, 10]]

    reactant_expected = separate(reactant)
    reactant_expected[0].set_tags(reactant_indices[0])
    reactant_expected[1].set_tags(reactant_indices[1])

    product_expected = separate(product)
    product_expected[0].set_tags(product_indices[0])
    product_expected[1].set_tags(product_indices[1])

    reactivity_matrix = np.zeros((13, 13), dtype=np.int8)
    reactivity_matrix[2, 6] = -1
    reactivity_matrix[2, 11] = 1
    reactivity_matrix = dok_matrix(reactivity_matrix)

    return reactant, product, reactant_indices, product_indices, reactant_expected, product_expected, reactivity_matrix
