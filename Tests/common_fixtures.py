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
def one_molecule_breakdown():
    numbers = np.array([8, 1, 7, 6, 6, 8, 6, 6, 6, 7, 6, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 1])
    positions = np.array([[-1.2072015, 3.30745609, 0.23500988],
                          [-1.07946655, 4.31619115, 2.379028],
                          [-3.51277448, 1.7412719, 1.38177868],
                          [-2.20142814, 1.43050236, 1.42474572],
                          [-1.51402415, 2.81869732, 1.28231616],
                          [-1.38347744, 3.39725203, 2.46639759],
                          [-1.70585696, 0.46600224, 0.31753105],
                          [-0.24666626, 0.07932497, 0.52005358],
                          [0.22829198, -0.86412901, -0.58282265],
                          [1.63318177, -1.15535894, -0.40540378],
                          [2.33848654, -2.03545905, -1.18287991],
                          [1.82661315, -2.3053925, -2.43712381],
                          [-1.90904282, 1.04218908, 2.41988118],
                          [-1.83939348, 0.97154292, -0.63951363],
                          [-2.3457762, -0.41717598, 0.33788642],
                          [-0.12327135, -0.41052014, 1.48959191],
                          [0.37908561, 0.97401352, 0.50637915],
                          [0.0814004, -0.37198351, -1.55107366],
                          [-0.37809854, -1.7824925, -0.57411687],
                          [1.9596544, -1.16980496, 0.54991451],
                          [1.21163614, -1.62471897, -2.85173981],
                          [2.47160049, -2.72846852, -3.08575342],
                          [3.39792526, -2.53838423, -0.68682543],
                          [3.91860212, -3.14055525, -1.31326088]])

    cell = np.zeros((3, 3))
    pbc = np.array([False, False, False])

    reactant = ase.Atoms.fromdict({'numbers': numbers, 'positions': positions, 'cell': cell, 'pbc': pbc})

    positions = np.array([[2.52048597e+00, 1.06922971e+01, 2.46801414e-01],
                          [2.40507869e+00, 9.60354097e+00, 2.59887598e+00],
                          [-4.82930106e+00, -3.00789398e+00, 1.10371577e+00],
                          [-3.65878459e+00, -3.33573868e+00, 1.00534056e+00],
                          [2.77678582e+00, 9.62189029e+00, 6.94090387e-01],
                          [2.72942352e+00, 9.05890887e+00, 1.84167869e+00],
                          [-2.68819670e+00, -2.93014043e+00, -9.16604770e-02],
                          [-1.43984438e+00, -2.28168252e+00, 4.99529968e-01],
                          [-4.46419950e-01, -1.89528046e+00, -5.93276792e-01],
                          [6.90627504e-01, -1.21636910e+00, -9.18352084e-03],
                          [1.85213099e+00, -9.70707881e-01, -6.98462306e-01],
                          [1.73924947e+00, -9.01391246e-01, -2.07803693e+00],
                          [-3.18017039e+00, -3.99288105e+00, 1.74934034e+00],
                          [-3.19618194e+00, -2.24331678e+00, -7.69670122e-01],
                          [-2.41707711e+00, -3.83402922e+00, -6.43631955e-01],
                          [-9.57387787e-01, -2.97453392e+00, 1.19418372e+00],
                          [-1.72171962e+00, -1.38522505e+00, 1.05579759e+00],
                          [-9.43789726e-01, -1.21991274e+00, -1.30116622e+00],
                          [-1.35225879e-01, -2.79421437e+00, -1.14698967e+00],
                          [8.85810251e-01, -1.44660311e+00, 9.54350390e-01],
                          [8.39532259e-01, -6.53926924e-01, -2.45652668e+00],
                          [2.51446355e+00, -4.60278734e-01, -2.54810608e+00],
                          [2.92364061e+00, -8.41302167e-01, -2.76439478e-02],
                          [3.73687048e+00, -5.91208899e-01, -5.79350110e-01]])

    product = ase.Atoms.fromdict({'numbers': numbers, 'positions': positions, 'cell': cell, 'pbc': pbc})

    reactant_indices = [list(range(24))]
    product_indices = [[0, 1, 4, 5], [2, 3] + list(range(6, 24))]

    reactant_molecules = separate(reactant)
    reactant_molecules[0].set_tags(reactant_indices[0])

    product_molecules = separate(product)
    product_molecules[0].set_tags(product_indices[0])
    product_molecules[1].set_tags(product_indices[1])

    reactivity_matrix = np.zeros((24, 24))
    reactivity_matrix[4, 5], reactivity_matrix[5, 4] = -1, -1

    return reactant, product, reactant_indices, product_indices, reactant_molecules, product_molecules, \
                dok_matrix(reactivity_matrix)


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
def overlapping_system_reactive(overlapping_system):
    indices = [list(range(14)), [14, 15, 16], [17, 18, 19, 20], list(range(21, 44)), [44, 45, 46], [47, 48, 49]]
    reactivity_matrix = np.zeros((50, 50))
    reactivity_matrix[17, 13], reactivity_matrix[13, 17] = 1, 1
    reactivity_matrix[9, 13], reactivity_matrix[13, 9] = -1, -1
    reactivity_matrix[9, 30], reactivity_matrix[30, 9] = 1, 1
    reactivity_matrix[48, 49], reactivity_matrix[49, 48] = -1, -1
    reactivity_matrix[15, 3], reactivity_matrix[3, 15] = 1, 1
    reactivity_matrix[15, 1], reactivity_matrix[1, 15] = 1, 1
    reactivity_matrix[15, 16], reactivity_matrix[16, 15] = -1, -1
    reactivity_matrix[15, 17], reactivity_matrix[17, 15] = -1, -1
    reactivity_matrix[16, 2], reactivity_matrix[2, 16] = 1, 1
    reactivity_matrix[17, 6], reactivity_matrix[6, 17] = 1, 1
    reactivity_matrix[1, 2], reactivity_matrix[2, 1] = -1, -1
    reactivity_matrix[3, 6], reactivity_matrix[6, 3] = -1, -1

    return overlapping_system, indices, dok_matrix(reactivity_matrix)


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
