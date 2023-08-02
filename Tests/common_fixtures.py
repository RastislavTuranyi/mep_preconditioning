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
