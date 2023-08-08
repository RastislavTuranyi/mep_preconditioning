from contextlib import nullcontext as does_not_raise

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial.transform import Rotation

from Src.stage3 import *

from Tests.common_fixtures import *


@pytest.mark.parametrize(['index', 'which', 'expected'],
                         [(0, 0, Rotation.from_quat([-0.41616115, -0.08740613, 0.27850356, 0.86116539])),
                          (1, 0, Rotation.from_quat([0.22196907, 0.83375788, -0.43964306, 0.24958267])),
                          (0, 1, Rotation.from_quat([0.35479002, -0.37783375, -0.53250004, 0.66918563])),
                          (1, 1, Rotation.from_quat([0.26427679, 0.66615146, 0.68880127, -0.10932899]))])
def test_compute_reactant_rotation(set_up_separate_molecules, index, which, expected):
    coordinates = (set_up_separate_molecules[0] if which == 0 else set_up_separate_molecules[1]).get_positions()
    molecules = set_up_separate_molecules[4] if which == 0 else set_up_separate_molecules[5]
    reactivity_matrix = set_up_separate_molecules[6]

    result = compute_reactant_rotation(coordinates, index, molecules, reactivity_matrix)

    assert_allclose(result.as_quat(False), expected.as_quat(False))


def test_reorient_reactants_one_reactant(one_molecule_breakdown):
    reactant, product, _, _, reactant_molecules, _, reactivity_matrix = one_molecule_breakdown

    expected_products = product.get_positions()
    expected_reactants = np.array([[-2.22215572, 2.73581206, 0.17084271],
                                   [-3.01974296, 3.30025832, 2.3332329],
                                   [-4.12267474, 0.40342998, 0.3488867],
                                   [-2.84624631, 0.46266939, 0.77932164],
                                   [-2.63067505, 1.98617703, 1.00812104],
                                   [-3.02368986, 2.33458638, 2.22383774],
                                   [-1.78286277, -0.09213116, -0.20209948],
                                   [-0.39187891, -0.09500888, 0.41832601],
                                   [0.64486586, -0.63650998, -0.56349045],
                                   [1.96284391, -0.55953385, 0.02559034],
                                   [3.09856259, -1.04190083, -0.5693595],
                                   [3.06605419, -1.19078225, -1.94223376],
                                   [-2.72976208, -0.0175948, 1.7704567],
                                   [-1.80356852, 0.53465223, -1.09431681],
                                   [-2.08428876, -1.10327327, -0.47895457],
                                   [-0.38884258, -0.71225165, 1.32064594],
                                   [-0.10882769, 0.92176753, 0.6980897],
                                   [0.61944443, -0.02380815, -1.47192018],
                                   [0.3883378, -1.66928073, -0.84412309],
                                   [2.00091679, -0.67069165, 1.02839666],
                                   [2.40376897, -0.63769271, -2.46077966],
                                   [3.96474362, -1.28475137, -2.38860744],
                                   [4.08406601, -1.321543, 0.18694564],
                                   [4.92161175, -1.62259863, -0.29680879]])

    reorient_reactants(reactant, reactant_molecules, reactivity_matrix)

    assert_allclose(reactant.get_positions(), expected_reactants, rtol=0, atol=10e-9)
    assert np.all(product.get_positions() == expected_products)


def test_reorient_reactants_two_reactants(set_up_separate_molecules):
    reactant, product, _, _, reactant_molecules, product_molecules, reactivity_matrix = set_up_separate_molecules

    expected_products = product.get_positions()
    expected_reactants = np.array([[-3.30003681, 3.10930287, -0.45428187],
                                   [-2.67913795, 2.89153779, 0.41701476],
                                   [-2.45335044, 3.18508882, -1.71242491],
                                   [-4.04369652, 2.31455617, -0.54844895],
                                   [-3.82714266, 4.05113739, -0.28900852],
                                   [-2.65953898, 2.45471615, -2.65909457],
                                   [-1.46682932, 4.1171208, -1.74358349],
                                   [-1.18076628, 5.00611404, -0.63485551],
                                   [-0.35095946, 5.65847626, -0.91921311],
                                   [-2.05878949, 5.61890354, -0.41384912],
                                   [-0.89253704, 4.42161186, 0.24290315],
                                   [0.47877851, -0.74381285, 0.72615268],
                                   [0.618894, -1.49608387, 0.16316544]])

    reorient_reactants(reactant, reactant_molecules, reactivity_matrix)

    assert_allclose(reactant.get_positions(), expected_reactants)
    assert np.all(product.get_positions() == expected_products)


def test_reorient_products_two_products_two_reactants(set_up_separate_molecules):
    reactant, product, _, _, reactant_molecules, product_molecules, reactivity_matrix = set_up_separate_molecules
    original_reactant = reactant.get_positions()

    reorient_products(product, [m.get_tags() for m in product_molecules],
                      reactant, [m.get_tags() for m in reactant_molecules])

    expected_product = np.array([[-1.31705836, 0.21902746, -3.79581672],
                                 [-0.73550967, -0.68955376, -3.62482322],
                                 [-0.72062076, 1.37541232, -3.0260643],
                                 [-1.32000976, 0.43551057, -4.86692912],
                                 [-2.34577451, 0.03887142, -3.47533814],
                                 [-0.2432662, 2.35747025, -3.55466106],
                                 [1.85962746, 1.61766193, 2.66772757],
                                 [1.96212142, 2.53067816, 1.61364266],
                                 [0.95062793, 2.84110576, 1.2727724],
                                 [2.50015183, 3.43287812, 1.97057088],
                                 [2.54301433, 2.08822631, 0.77549653],
                                 [-0.7618547, 1.21865909, -1.72941558],
                                 [-1.18455746, 0.36275032, -1.54466332]])

    assert_allclose(product.get_positions(), expected_product, atol=10e-9, rtol=0)
    assert np.all(reactant.get_positions() == original_reactant)


def test_reorient_products_two_products_one_reactant(one_molecule_breakdown):
    reactant, product, reactant_indices, product_indices, _, _, reactivity_matrix = one_molecule_breakdown
    original_reactant = reactant.get_positions()

    reorient_products(product, reactant_indices, reactant, product_indices)

    expected_product = np.array([[-1.59114546, 1.21435282, -10.80428531],
                                 [-3.73786772, 0.53899427, -9.51337948],
                                 [0.54619849, -4.60171966, 3.48046071],
                                 [0.3129154, -3.42513661, 3.70068777],
                                 [-2.03224912, 1.44064148, -9.72458685],
                                 [-3.08047219, 1.122953, -9.06394124],
                                 [1.04911907, -2.2269713, 3.12417574],
                                 [0.07943949, -1.27699306, 2.42728969],
                                 [0.80599323, -0.05630464, 1.86830427],
                                 [-0.12577303, 0.78661546, 1.14979204],
                                 [0.17434194, 2.06545854, 0.75099834],
                                 [1.51935687, 2.34899353, 0.57403526],
                                 [-0.50044832, -3.11592813, 4.37690733],
                                 [1.80653306, -2.58627136, 2.42653664],
                                 [1.54706009, -1.71549805, 3.95221146],
                                 [-0.68596, -0.94616658, 3.13444506],
                                 [-0.418996, -1.79916935, 1.60788912],
                                 [1.58942029, -0.39568423, 1.17861648],
                                 [1.29347153, 0.49329666, 2.6879537],
                                 [-1.08915209, 0.71824929, 1.44428899],
                                 [2.13198493, 1.57615455, 0.37058772],
                                 [1.71338427, 3.17655937, 0.03182656],
                                 [-0.7893601, 2.87818684, 0.59091379],
                                 [-0.51779463, 3.78538717, 0.22827222]])

    assert_allclose(product.get_positions(), expected_product, atol=10e-9, rtol=0)
    assert np.all(reactant.get_positions() == original_reactant)
