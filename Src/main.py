from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import ase
from ase.calculators.calculator import Calculator
from ase.build import separate, connected_indices
from ase.geometry.analysis import Analysis
from ase.optimize import BFGS
import ase.io

import numpy as np

if TYPE_CHECKING:
    from typing import Union
    from numpy.typing import ArrayLike
    from scipy.sparse import dok_matrix


class ConvergenceError(Exception):
    pass


class InputError(Exception):
    pass


def main(start=None, end=None, both=None):
    if both is not None:
        reactant = ase.io.read(both, 0)
        product = ase.io.read(both, 1)
    elif start is not None and end is not None:
        reactant = ase.io.read(start)
        product = ase.io.read(end)
    else:
        raise InputError()

    reactant_molecules = separate_molecules(reactant)
    product_molecules = separate_molecules(product)

    reactivity_matrix = get_reactivity_matrix(reactant, product)

    initial_positioning(reactant, product, reactant_molecules, product_molecules, reactivity_matrix)

    fix_overlaps(reactant, [mol.get_tags() for mol in reactant_molecules])
    fix_overlaps(product, [mol.get_tags() for mol in reactant_molecules])

    ase.io.write('start.xyz', reactant)
    ase.io.write('end.xyz', product)


def initial_positioning(reactant: ase.Atoms,
                        product: ase.Atoms,
                        reactant_molecules: list[ase.Atoms],
                        product_molecules: list[ase.Atoms],
                        reactivity_matrix: dok_matrix):
    coordinates = reactant.get_positions()

    shared_atom_geometric_centres = []
    reactive_atom_geometric_centres = []

    for reactant_mol in reactant_molecules:
        # Translate reactant molecules so that their geometric centres are (approximately) at origin
        reactant_mol.translate(-np.mean(reactant_mol.get_positions(), axis=0))

        # Calculate geometric centres of reactive (B, beta) and shared atoms (C, sigma)
        for product_mol in product_molecules:
            shared = get_shared_atoms(reactant_mol, product_mol)
            reactive = get_reactive_atoms(shared, reactivity_matrix)

            if shared.size > 0:
                shared_atom_geometric_centres.append(np.mean(coordinates[shared, :], axis=0))
            if reactive.size > 0:
                reactive_atom_geometric_centres.append(np.mean(coordinates[reactive, :], axis=0))

    # Calculate the geometric centres of bond-forming atoms (A, alpha)
    bonding_atom_geometric_centres = []
    for rmol1, rmol2 in combinations(reactant_molecules, 2):
        atoms = get_bond_forming_atoms(rmol1, rmol2, True, reactivity_matrix)
        bonding_atom_geometric_centres.append(np.mean(coordinates[atoms, :], axis=0))

    # Calculate the vectors which will be used to reposition the reactants and products
    n_reactants = len(reactant_molecules)
    beta = np.sum(np.array(reactive_atom_geometric_centres), axis=0) / n_reactants
    sigma = np.sum(np.array(shared_atom_geometric_centres), axis=0) / n_reactants

    reactant_repositioning_vector = np.sum(np.array(bonding_atom_geometric_centres), axis=0) / n_reactants
    product_repositioning_vector = (beta + sigma) / 2

    # Translate all reactants and products using the computed vectors
    reactant.translate(reactant_repositioning_vector)
    for reactant_mol in reactant_molecules:
        reactant_mol.translate(reactant_repositioning_vector)

    product.translate(product_repositioning_vector)
    for product_mol in product_molecules:
        product_mol.translate(product_repositioning_vector)


def determine_overlaps(molecules: list[ase.Atoms],
                       geometric_centres: list[np.ndarray],
                       estimated_radii: list[float]) -> np.ndarray:
    size = len(molecules)
    overlaps = np.zeros((size, size), dtype=np.int8)
    indices = list(range(size))

    for mol1, mol2 in combinations(indices, 2):
        centre_distance = np.linalg.norm(geometric_centres[mol2] - geometric_centres[mol1])
        combined_radius = estimated_radii[mol2] + estimated_radii[mol1]

        if centre_distance < combined_radius:
            overlaps[mol1, mol2], overlaps[mol2, mol1] = 1, 1
        # Else keep it 0

    return overlaps


def estimate_molecular_radius(molecule: ase.Atoms, geometric_centre: np.ndarray) -> float:
    distances = np.zeros(len(molecule))
    for i, atom in enumerate(molecule.get_positions()):
        distances[i] = np.linalg.norm(atom - geometric_centre)

    mean = np.mean(distances)
    std = np.std(distances)

    return mean + 2 * std


def fix_overlaps(system: ase.Atoms,
                 molecules: list[list[int]],
                 force_constant: float = 1.0,
                 fmax: float = 1e-5,
                 max_iter: int = 1000,
                 non_convergence_limit: Union[float, None] = 0.001,
                 non_convergence_roof: Union[float, None] = 0.5,
                 trial_constants: Union[None, float, tuple[float], tuple[float, float], tuple[float, float, float],
                                        list[float], np.ndarray] = 10.0):
    system.calc = HardSphereCalculator(molecules, force_constant)
    if non_convergence_roof is None or non_convergence_limit is None:
        dyn = BFGS(system)
    else:
        dyn = ConstrainedBFGS(system, non_convergence_limit, non_convergence_roof)

    # Try using ASE optimiser, but switch to custom optimisation scheme if it does not converge
    try:
        dyn.run(fmax=fmax, steps=max_iter)
    except OptimisationNotConvergingError as e:
        if trial_constants is None:
            raise ConvergenceError(f'Molecule overlaps failed to be fixed: The geometry optimisation using '
                                   f'`ase.optimize.BFGS` was aborted early (iteration={dyn._total_iteration}, latest '
                                   f'fmax={e.fmax}, average fmax on previous iteration={e.previous_average}, average '
                                   f'fmax on current iteration={e.new_average}) because the optimisation was not '
                                   f'converging.\n\n> If you\'d like to disable this behaviour and run BFGS until '
                                   f'completion ({max_iter=}), pass in `None` to the `non_convergence_limit` and/or '
                                   f'`non_convergence_limit` parameters, but beware that this is likely to result in '
                                   f'a VERY long optimisation that is highly unlikely to converge.'
                                   f'\n\n> If you\'d like to reach convergence, you can enable further attempts at '
                                   f'optimisation using increasing values of force constant with a very simple scheme '
                                   f'(`simple_optimise_structure`). See documentation for `fix_overlaps` for more '
                                   f'details.')

        # TODO: Write tests for each of these input methods
        # Create a range of increasing force constants
        if isinstance(trial_constants, float):
            trial_constants = np.arange(force_constant, trial_constants, 1.0)
        elif isinstance(trial_constants, tuple):
            if len(trial_constants) == 1:
                trial_constants = np.arange(force_constant, trial_constants[0], 1.0)
            elif len(trial_constants) == 2:
                trial_constants = np.arange(trial_constants[0], trial_constants[1], 1.0)
            elif len(trial_constants) == 3:
                trial_constants = np.arange(trial_constants[0], trial_constants[1], trial_constants[2])

        # Try optimising structure using a series of increasing force constants
        for trial_constant in trial_constants:
            # TODO: Talk about this function and its results when used after BFGS vs without
            new_positions = simple_optimise_structure(system, molecules, trial_constant, fmax, max_iter)
            if new_positions is not None:
                break
        else:
            raise ConvergenceError('Molecule overlaps failed to converge: molecule overlaps could not be resolved '
                                   f'within the provided iterations ({max_iter=}) and range of force constants ('
                                   f'{trial_constants=}). This is likely due to optimisation failing to converge. '
                                   f'Increasing the upper bound of the `trial_constants` parameter should allow for '
                                   f' convergence to be reached, though possibly at the cost of the molecules ending '
                                   f'further apart.')

        system.set_positions(new_positions)


def get_all_bond_forming_atoms_in_molecule(molecule: ase.Atoms,
                                           reactants: bool,
                                           reactivity_matrix: dok_matrix) -> np.ndarray:
    search = 1 if reactants else -1
    atoms = molecule.get_tags()

    bonding_atoms = []
    for key, val in reactivity_matrix.items():
        if val == search and (key[0] in atoms or key[1] in atoms):
            bonding_atoms.append(key[0])

    return np.array(bonding_atoms)


def get_bond_forming_atoms(molecule1: ase.Atoms,
                           molecule2: ase.Atoms,
                           reactants: bool,
                           reactivity_matrix: dok_matrix) -> np.ndarray:
    search = 1 if reactants else -1

    atoms1, atoms2 = molecule1.get_tags(), molecule2.get_tags()

    bonding_atoms = []
    for key, val in reactivity_matrix.items():
        if val == search:
            if key[0] in atoms1 and key[1] in atoms2:
                bonding_atoms.append(key[0])
            elif key[1] in atoms1 and key[0] in atoms2:
                bonding_atoms.append(key[1])

    return np.array(bonding_atoms)


def get_reactive_atoms(shared_atoms: np.ndarray,
                       reactivity_matrix: dok_matrix) -> np.ndarray:
    reactive_atoms = []
    for atom in shared_atoms:
        row = reactivity_matrix[atom, :]
        if row.count_nonzero() > 0:
            reactive_atoms.append(atom)

    return np.array(reactive_atoms)


def get_reactivity_matrix(reactant: ase.Atoms, product: ase.Atoms) -> dok_matrix:
    reactant_connectivity = Analysis(reactant).adjacency_matrix[0]
    product_connectivity = Analysis(product).adjacency_matrix[0]

    return (product_connectivity - reactant_connectivity).todok()


def get_indices(arr: np.ndarray, values):
    sorter = np.argsort(arr)
    return sorter[np.searchsorted(arr, values, sorter=sorter)]


# noinspection PyTypeChecker
def get_shared_atoms(reactant_molecule: ase.Atoms, product_molecule: ase.Atoms) -> np.ndarray:
    intersection = np.intersect1d(reactant_molecule.get_tags(), product_molecule.get_tags())
    return intersection


def reorient_reactants(reactant: ase.Atoms, molecules: list[ase.Atoms], reactivity_matrix: dok_matrix):
    for molecule in molecules:
        g = np.mean(molecule.get_positions(), axis=0)

        bonding_atoms = get_all_bond_forming_atoms_in_molecule(molecule, True, reactivity_matrix)
        gamma = np.mean(reactant.get_positions()[bonding_atoms], axis=0)


def rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def separate_molecules(system: ase.Atoms, molecules: Union[None, list[list[int]]] = None) -> list[ase.Atoms]:
    if molecules is None:
        return _separate_molecules_using_connectivity(system)
    else:
        return _separate_molecules_using_list(system, molecules)


def _separate_molecules_using_connectivity(system: ase.Atoms) -> list[ase.Atoms]:
    indices = list(range(len(system)))

    separated = []
    while indices:
        my_indices = connected_indices(system, indices[0])
        separated.append(ase.Atoms(cell=system.cell, pbc=system.pbc))

        for i in my_indices:
            separated[-1].append(system[i])
            del indices[indices.index(i)]

        separated[-1].set_tags(my_indices)

    return separated


def _separate_molecules_using_list(system: ase.Atoms,
                                   molecules: list[list[int]]) -> list[ase.Atoms]:
    separated = []
    for molecule in molecules:
        separated.append(ase.Atoms(cell=system.cell, pbc=system.pbc))
        for index in molecule:
            separated[-1].append(system[index])

        separated[-1].set_tags(molecule)

    return separated


def simple_optimise_structure(system: ase.Atoms,
                              molecule_indices: list[list[int]],
                              force_constant: float = 1.0,
                              fmax: float = 1e-5,
                              max_iter: int = 500) -> Union[np.ndarray, None]:
    calc = HardSphereCalculator(molecule_indices, force_constant)
    molecules = separate_molecules(system, molecule_indices)

    forces = calc.compute_hard_sphere_forces(molecules)
    max_force = np.sqrt(np.max(np.sum(forces ** 2, axis=1)))

    for i in range(max_iter):
        if max_force < fmax:
            break

        for force, molecule in zip(forces, molecules):
            molecule.translate(force)

        forces = calc.compute_hard_sphere_forces(molecules)
        max_force = np.sqrt(np.max(np.sum(forces ** 2, axis=1)))
    else:
        return None

    new_positions = np.zeros((len(system), 3))
    for molecule, indices in zip(molecules, molecule_indices):
        new_positions[indices] = molecule.get_positions()

    return new_positions


class OptimisationNotConvergingError(Exception):
    def __init__(self, fmax, previous_average, new_average, *args):
        self.fmax = fmax
        self.previous_average = previous_average
        self.new_average = new_average

        super().__init__(*args)


class ConstrainedBFGS(BFGS):
    def __init__(self,
                 atoms: ase.Atoms,
                 non_convergence_limit: float = 0.001,
                 non_convergence_roof: float = 0.5,
                 logfile: str = '-',
                 maxstep: Union[float, None] = None,
                 master: Union[bool, None] = None,
                 alpha: Union[float, None] = None):
        self._total_fmax: float = 0.0
        self._total_iteration: float = 0.0
        self.non_convergence_limit = non_convergence_limit
        self.non_convergence_roof = non_convergence_roof

        super().__init__(atoms=atoms, restart=None, logfile=logfile, trajectory=None, maxstep=maxstep,
                         master=master, alpha=alpha)

    def converged(self, forces=None) -> bool:
        if forces is None:
            forces = self.atoms.get_forces()

        max_force = (forces ** 2).sum(axis=1).max()

        try:
            average_until_now = self._total_fmax / self._total_iteration
        except ZeroDivisionError:
            average_until_now = 1000

        self._total_fmax += max_force
        self._total_iteration += 1.0

        new_average = self._total_fmax / self._total_iteration

        if max_force > self.non_convergence_roof and abs(average_until_now - new_average) < self.non_convergence_limit:
            raise OptimisationNotConvergingError(max_force, average_until_now, new_average)

        if hasattr(self.atoms, "get_curvature"):
            return max_force < self.fmax ** 2 and self.atoms.get_curvature() < 0.0
        return max_force < self.fmax ** 2


class HardSphereCalculator(Calculator):
    implemented_properties = ['forces', 'energy']

    def __init__(self, molecules: list[list[int]], force_constant: float = 1.0, label=None, atoms=None,
                 directory='.', **kwargs):
        self.force_constant = force_constant
        self.molecules = molecules

        super().__init__(restart=None, label=label, atoms=atoms, directory=directory, **kwargs)

    def calculate(self, atoms=None, properties=None, system_changes=None) -> None:
        self.atoms = atoms.copy()
        molecules = separate_molecules(self.atoms, self.molecules)
        forces = self.compute_hard_sphere_forces(molecules)

        submit_forces = np.zeros((len(self.atoms), 3))

        for mol, force in zip(molecules, forces):
            for index in mol.get_tags():
                submit_forces[index, :] = force

        self.results['forces'] = submit_forces
        self.results['energy'] = 0.0

    def compute_hard_sphere_forces(self, molecules: list[ase.Atoms]) -> np.ndarray:
        geometric_centres = [np.mean(mol.get_positions(), axis=0) for mol in molecules]
        molecular_radii = [estimate_molecular_radius(mol, centre) for mol, centre in zip(molecules, geometric_centres)]

        overlaps = determine_overlaps(molecules, geometric_centres, molecular_radii)

        forces = np.zeros((len(molecules), 3), dtype=np.float64)
        for i, affected_mol in enumerate(molecules):
            n_atoms = len(affected_mol)
            n = 3 * n_atoms * np.sum(overlaps[i, :])

            pairwise_forces = []
            for j, other_mol in enumerate(molecules):
                if overlaps[i, j] == 0 or i == j:
                    continue

                centre_diff = geometric_centres[i] - geometric_centres[j]
                distance = np.linalg.norm(centre_diff)
                phi = self.force_constant * (distance - (molecular_radii[i] + molecular_radii[j])) / n
                pairwise_forces.append(phi * centre_diff / distance)

            forces[i, :] = n_atoms * np.sum(np.array(pairwise_forces), axis=0)

        return forces
