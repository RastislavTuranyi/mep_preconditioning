from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import ase
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS

import numpy as np

from Src.common_functions import separate_molecules, _CustomBaseCalculator

if TYPE_CHECKING:
    from typing import Union
    from numpy.typing import ArrayLike
    from scipy.sparse import dok_matrix


class ConvergenceError(Exception):
    pass


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


def simple_optimise_structure(system: ase.Atoms,
                              molecule_indices: list[list[int]],
                              force_constant: float = 1.0,
                              fmax: float = 1e-5,
                              max_iter: int = 500) -> Union[np.ndarray, None]:
    calc = HardSphereCalculator(molecule_indices, force_constant)
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


class HardSphereCalculator(_CustomBaseCalculator):
    def compute_forces(self, molecules: list[ase.Atoms]) -> np.ndarray:
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
