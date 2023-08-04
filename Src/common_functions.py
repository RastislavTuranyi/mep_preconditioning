from __future__ import annotations

import abc
from abc import ABC
from typing import TYPE_CHECKING

import ase
from ase.calculators.calculator import Calculator
from ase.build import connected_indices
from ase.geometry.analysis import Analysis
from ase.optimize import BFGS

import numpy as np

if TYPE_CHECKING:
    from typing import Union
    from numpy.typing import ArrayLike
    from scipy.sparse import dok_matrix


class ConvergenceError(Exception):
    pass


def compute_alpha_vector(coordinates: np.ndarray,
                         target: int,
                         molecules: list[ase.Atoms],
                         reactant: bool,
                         reactivity_matrix: dok_matrix) -> np.ndarray:
    r"""
    .. math::
        \alpha_{Rm} = \frac{1}{N^R} \sum_{n≠m}^{N^R}{\frac{1}{|A^{Rnm}|}\sum_{a∈A^{Rnm}}{\xrightarrow{r_{Rn}^a}}}
    :param coordinates:
    :param target:
    :param molecules:
    :param reactant:
    :param reactivity_matrix:
    :return:
    """
    alpha = np.zeros(3)
    target_mol = molecules[target]
    n_mol = len(molecules)

    for j, mol2 in enumerate(molecules):
        if target == j:
            continue

        atoms = get_bond_forming_atoms(target_mol, mol2, reactant, reactivity_matrix)
        if atoms.size > 0:
            alpha += np.mean(coordinates[atoms, :], axis=0)

    return alpha / n_mol


def get_all_bond_forming_atoms_in_molecule(molecule: ase.Atoms,
                                           reactants: bool,
                                           reactivity_matrix: dok_matrix) -> np.ndarray:
    # TODO: Write test
    search = 1 if reactants else -1
    atoms = molecule.get_tags()

    bonding_atoms = []
    for key, val in reactivity_matrix.items():
        if val == search and (key[0] in atoms or key[1] in atoms):
            bonding_atoms.append(key[0])

    return np.array(bonding_atoms)


def get_bond_forming_atoms(molecule1: Union[ase.Atoms, list[int]],
                           molecule2: Union[ase.Atoms, list[int]],
                           reactants: bool,
                           reactivity_matrix: dok_matrix,
                           return_both: bool = False) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    search = 1 if reactants else -1

    try:
        atoms1, atoms2 = molecule1.get_tags(), molecule2.get_tags()
    except AttributeError:
        atoms1, atoms2 = molecule1, molecule2

    bonding_atoms_molecule1, bonding_atoms_molecule2 = [], []
    for key, val in reactivity_matrix.items():
        if val == search:
            if key[0] in atoms1 and key[1] in atoms2:
                bonding_atoms_molecule1.append(key[0])
                bonding_atoms_molecule2.append(key[1])
            elif key[1] in atoms1 and key[0] in atoms2:
                bonding_atoms_molecule1.append(key[1])
                bonding_atoms_molecule2.append(key[0])

    if return_both:
        return np.array(bonding_atoms_molecule1), np.array(bonding_atoms_molecule2)
    else:
        return np.array(bonding_atoms_molecule1)


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


# noinspection PyTypeChecker
def get_shared_atoms(reactant_molecule: Union[ase.Atoms, list[int]],
                     product_molecule: Union[ase.Atoms, list[int]]) -> np.ndarray:
    try:
        intersection = np.intersect1d(reactant_molecule.get_tags(), product_molecule.get_tags())
    except AttributeError:
        intersection = np.intersect1d(reactant_molecule, product_molecule)
    return intersection


def optimise_system(system: ase.Atoms,
                    calc: _CustomBaseCalculator,
                    molecules: list[list[int]],
                    force_constant: float = 1.0,
                    fmax: float = 1e-5,
                    max_iter: int = 1000,
                    non_convergence_limit: Union[float, None] = 0.001,
                    non_convergence_roof: Union[float, None] = 0.5,
                    trial_constants: Union[None, float, tuple[float], tuple[float, float], tuple[float, float, float],
                                           list[float], np.ndarray] = 10.0) -> Union[np.ndarray, None]:
    system.calc = calc
    if non_convergence_roof is None or non_convergence_limit is None:
        dyn = BFGS(system)
    else:
        dyn = ConstrainedBFGS(system, non_convergence_limit, non_convergence_roof)

    # Try using ASE optimiser, but switch to custom optimisation scheme if it does not converge
    try:
        dyn.run(fmax=fmax, steps=max_iter)
        return None
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

        # Create a range of increasing force constants
        trial_constants = optimise_system_create_trial_constants(force_constant, trial_constants)

        # Try optimising structure using a series of increasing force constants
        for trial_constant in trial_constants:
            # TODO: Talk about this function and its results when used after BFGS vs without
            calc.force_constant = trial_constant
            new_positions = simple_optimise_structure(system, calc, molecules, fmax, max_iter)
            if new_positions is not None:
                break
        else:
            raise ConvergenceError('Molecule overlaps failed to converge: molecule overlaps could not be resolved '
                                   f'within the provided iterations ({max_iter=}) and range of force constants ('
                                   f'{trial_constants=}). This is likely due to optimisation failing to converge. '
                                   f'Increasing the upper bound of the `trial_constants` parameter should allow for '
                                   f' convergence to be reached, though possibly at the cost of the molecules ending '
                                   f'further apart.')

        return new_positions


def optimise_system_create_trial_constants(force_constant: float,
                                           trial_constants: Union[None, float, tuple[float], tuple[float, float],
                                                                  tuple[float, float, float], list[float], np.ndarray]
                                           ) -> np.ndarray:
    if isinstance(trial_constants, float):
        trial_constants = np.arange(force_constant, trial_constants, 1.0)
    elif isinstance(trial_constants, tuple):
        if len(trial_constants) == 1:
            trial_constants = np.arange(force_constant, trial_constants[0], 1.0)
        elif len(trial_constants) == 2:
            trial_constants = np.arange(trial_constants[0], trial_constants[1], 1.0)
        elif len(trial_constants) == 3:
            trial_constants = np.arange(trial_constants[0], trial_constants[1], trial_constants[2])

    return np.array(trial_constants)


def simple_optimise_structure(system: ase.Atoms,
                              calc: _CustomBaseCalculator,
                              molecule_indices: list[list[int]],
                              fmax: float = 1e-5,
                              max_iter: int = 500) -> Union[np.ndarray, None]:
    molecules = separate_molecules(system, molecule_indices)

    forces = calc.compute_forces(molecules)
    max_force = np.sqrt(np.max(np.sum(forces ** 2, axis=1)))

    for i in range(max_iter):
        if max_force < fmax:
            break

        for force, molecule in zip(forces, molecules):
            molecule.translate(force)

        forces = calc.compute_forces(molecules)
        max_force = np.sqrt(np.max(np.sum(forces ** 2, axis=1)))
    else:
        return None

    new_positions = np.zeros((len(system), 3))
    for molecule, indices in zip(molecules, molecule_indices):
        new_positions[indices] = molecule.get_positions()

    return new_positions


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


class _CustomBaseCalculator(Calculator, ABC):
    implemented_properties = ['forces', 'energy']

    def __init__(self, molecules: list[list[int]], force_constant: float = 1.0, label=None, atoms=None,
                 directory='.', **kwargs):
        self.force_constant = force_constant
        self.molecules = molecules

        super().__init__(restart=None, label=label, atoms=atoms, directory=directory, **kwargs)

    def calculate(self, atoms=None, properties=None, system_changes=None) -> None:
        self.atoms = atoms.copy()
        # TODO: Look into removing this separation step
        molecules = separate_molecules(self.atoms, self.molecules)
        forces = self.compute_forces(molecules)

        submit_forces = np.zeros((len(self.atoms), 3))

        for mol, force in zip(molecules, forces):
            for index in mol.get_tags():
                submit_forces[index, :] = force

        self.results['forces'] = submit_forces
        self.results['energy'] = 0.0

    @abc.abstractmethod
    def compute_forces(self, molecules: list[ase.Atoms]) -> np.ndarray:
        pass
