import numpy as np
from numpy.testing import assert_allclose
import pytest

from Src.common_functions import separate_molecules, get_reactivity_matrix
from Src.stage1 import *

from Tests.common_fixtures import *


def test_reposition_reactants_one_molecule(one_molecule_breakdown):
    reactant, product, reactant_indices, _, _, _, reactivity_matrix = one_molecule_breakdown

    product_expected = product.get_positions()
    reactant_expected = reactant.get_positions()

    reposition_reactants(reactant, reactant_indices, reactivity_matrix)

    assert_allclose(reactant.get_positions(), reactant_expected, rtol=0, atol=10e-9)
    assert_allclose(product.get_positions(), product_expected)


def test_reposition_reactants_two_molecules(set_up_separate_molecules):
    reactant, product, reactant_indices, _, _, _, reactivity_matrix = set_up_separate_molecules

    product_expected = product.get_positions()
    reactant_expected = np.array([[-1.55600045, -0.01439364, -0.40663091],
                                  [-1.23201045, -1.04254364, -0.23333091],
                                  [-0.70949045, 0.64189636, -1.48285091],
                                  [-2.60432045, -0.03587364, -0.71336091],
                                  [-1.48642045, 0.54308636, 0.52962909],
                                  [-1.20709045, 1.08634636, -2.49626091],
                                  [0.62632955, 0.72891636, -1.25726091],
                                  [1.26464955, 0.20698636, -0.06497091],
                                  [2.33653955, 0.41217636, -0.12793091],
                                  [0.85680955, 0.70057636, 0.82120909],
                                  [1.10953955, -0.87355364, -0.00536091],
                                  [-0.091485, -0.0342225, -0.7057725],
                                  [0.030495, 0.0114075, 0.2352575]])

    reposition_reactants(reactant, reactant_indices, reactivity_matrix)
    print(repr(reactant.get_positions()))
    assert_allclose(reactant.get_positions(), reactant_expected, rtol=0, atol=10e-9)
    assert_allclose(product.get_positions(), product_expected)


def test_reposition_reactants_complex(overlapping_system_reactive):
    reactant, reactant_indices, reactivity_matrix = overlapping_system_reactive

    reactant_expected = np.array([[-9.99700357e-01, -3.49945833e-01, 5.34245873e-01],
                                  [-5.14580357e-01, -9.91025833e-01, 1.24034587e+00],
                                  [-3.48470357e-01, -1.77475833e-01, -2.97054127e-01],
                                  [-1.90064036e+00, -8.14215833e-01, 1.91225873e-01],
                                  [-1.23509036e+00, 5.82934167e-01, 1.00246587e+00],
                                  [-1.13070357e-01, -1.11034583e+00, -7.65274127e-01],
                                  [-8.33580357e-01, 4.63614167e-01, -1.00315413e+00],
                                  [5.52469643e-01, 2.86794167e-01, 4.59658730e-02],
                                  [1.03757964e+00, -3.54295833e-01, 7.52065873e-01],
                                  [1.20368964e+00, 4.59264167e-01, -7.85334127e-01],
                                  [3.17069643e-01, 1.21966417e+00, 5.14185873e-01],
                                  [1.43908964e+00, -4.73615833e-01, -1.25355413e+00],
                                  [7.18579643e-01, 1.10034417e+00, -1.49143413e+00],
                                  [2.10462964e+00, 9.23534167e-01, -4.42304127e-01],
                                  [1.56701111e-01, 1.51053889e-01, -2.58518333e-01],
                                  [-5.53798889e-01, -4.47496111e-01, -4.57088333e-01],
                                  [2.78681111e-01, 1.96683889e-01, 6.82511667e-01],
                                  [8.89816667e-02, 2.71795417e-01, -9.19741667e-02],
                                  [-1.98333333e-04, -3.53734583e-01, -9.05394167e-01],
                                  [-8.22148333e-01, 3.50925417e-01, 3.81825833e-01],
                                  [7.84211667e-01, -1.13674583e-01, 5.62985833e-01],
                                  [2.05763696e-01, 7.93627319e-01, -6.41072029e-01],
                                  [6.70103696e-01, 1.45743732e+00, 5.79679710e-02],
                                  [-7.76106304e-01, 1.14994732e+00, -8.73162029e-01],
                                  [7.90823696e-01, 7.51867319e-01, -1.53598203e+00],
                                  [1.38243696e-01, -1.84732681e-01, -2.13112029e-01],
                                  [-4.46816304e-01, -1.42972681e-01, 6.81797971e-01],
                                  [-3.26086304e-01, -8.48532681e-01, -9.12152029e-01],
                                  [1.12011370e+00, -5.41052681e-01, 1.89779710e-02],
                                  [8.58343696e-01, 1.73023732e+00, -1.96394203e+00],
                                  [1.77269370e+00, 3.95557319e-01, -1.30389203e+00],
                                  [3.26483696e-01, 8.80673188e-02, -2.23502203e+00],
                                  [1.32267370e+00, 2.39403732e+00, -1.26490203e+00],
                                  [1.44340370e+00, 1.68847732e+00, -2.85885203e+00],
                                  [-1.23526304e-01, 2.08654732e+00, -2.19603203e+00],
                                  [-1.42868630e+00, 2.13337319e-01, 4.49707971e-01],
                                  [-5.14336304e-01, -1.12133268e+00, 1.10975797e+00],
                                  [1.75236957e-02, 5.20827319e-01, 1.38083797e+00],
                                  [-9.78666304e-01, -1.78514268e+00, 4.10717971e-01],
                                  [4.67533696e-01, -1.47765268e+00, 1.34184797e+00],
                                  [-1.09939630e+00, -1.07958268e+00, 2.00466797e+00],
                                  [1.05259370e+00, -1.51941268e+00, 4.46937971e-01],
                                  [9.31873696e-01, -8.13852681e-01, 2.04088797e+00],
                                  [4.00013696e-01, -2.45601268e+00, 1.76979797e+00],
                                  [2.57510000e-01, 8.26100000e-02, -2.46110000e-01],
                                  [-3.56300000e-02, -7.95460000e-01, -3.26500000e-02],
                                  [-2.21880000e-01, 7.12850000e-01, 2.78760000e-01],
                                  [-2.97466667e-02, -2.97536667e-01, 2.10460000e-01],
                                  [6.63213333e-01, 3.49553333e-01, 2.70270000e-01],
                                  [-6.33466667e-01, -5.20166667e-02, -4.80730000e-01]])

    reposition_reactants(reactant, reactant_indices, reactivity_matrix)

    assert_allclose(reactant.get_positions(), reactant_expected, rtol=0, atol=10e-9)


def test_reposition_products_two_products_from_two_reactants(set_up_separate_molecules):
    reactant, product, reactant_indices, product_indices, _, _, reactivity_matrix = set_up_separate_molecules

    reactant_expected = reactant.get_positions()
    product_expected = np.array([[-1.10267583, 1.38115208, 1.69827458],
                                 [-0.75933583, 0.35489208, 1.84614458],
                                 [-0.21757583, 2.09084208, 0.69906458],
                                 [-2.13544583, 1.35610208, 1.34202458],
                                 [-1.07999583, 1.89746208, 2.66073458],
                                 [-0.60885583, 2.49192208, -0.37692542],
                                 [1.631778, 0.7532195, 0.196242],
                                 [0.401618, 0.8707895, 0.850522],
                                 [0.067398, -0.1213105, 1.224302],
                                 [0.520928, 1.5476095, 1.721452],
                                 [-0.357142, 1.3121595, 0.168492],
                                 [1.01821417, 2.23061208, 1.10013458],
                                 [1.09251417, 1.83391208, 1.98472458]])

    reposition_products(reactant, product, reactant_indices, product_indices, reactivity_matrix)

    assert_allclose(product.get_positions(), product_expected)
    assert np.all(reactant.get_positions() == reactant_expected)


def test_reposition_products_two_products_from_one_reactant(one_molecule_breakdown):
    reactant, product, reactant_indices, product_indices, _, _, reactivity_matrix = one_molecule_breakdown

    reactant_expected = reactant.get_positions()
    product_expected = np.array([[-1.45985413, 4.2320747, 0.63396219],
                                 [-1.57526141, 3.14331857, 2.98603675],
                                 [-4.17810812, -1.40505203, 1.2137193],
                                 [-3.00759165, -1.73289673, 1.11534409],
                                 [-1.20355428, 3.16166789, 1.08125116],
                                 [-1.25091658, 2.59868647, 2.22883946],
                                 [-2.03700376, -1.32729848, 0.01834306],
                                 [-0.78865144, -0.67884057, 0.6095335],
                                 [0.20477299, -0.29243851, -0.48327326],
                                 [1.34182045, 0.38647285, 0.10082001],
                                 [2.50332393, 0.63213407, -0.58845877],
                                 [2.39044241, 0.7014507, -1.9680334],
                                 [-2.52897745, -2.3900391, 1.85934387],
                                 [-2.544989, -0.64047483, -0.65966659],
                                 [-1.76588417, -2.23118727, -0.53362842],
                                 [-0.30619485, -1.37169197, 1.30418725],
                                 [-1.07052668, 0.2176169, 1.16580112],
                                 [-0.29259678, 0.38292921, -1.19116269],
                                 [0.51596706, -1.19137242, -1.03698614],
                                 [1.53700319, 0.15623884, 1.06435392],
                                 [1.4907252, 0.94891502, -2.34652315],
                                 [3.16565649, 1.14256321, -2.43810255],
                                 [3.57483355, 0.76153978, 0.08235958],
                                 [4.38806342, 1.01163305, -0.46934658]])

    reposition_products(reactant, product, reactant_indices, product_indices, reactivity_matrix)
    print(repr(product.positions))
    assert_allclose(product.get_positions(), product_expected, rtol=0, atol=10e-9)
    assert np.all(reactant.get_positions() == reactant_expected)
