from __future__ import annotations

import logging

import ase
from ase.optimize import BFGS
import numpy as np


class DualBFGS(BFGS):
    def __init__(self, reactant: ase.Atoms, product: ase.Atoms, logfile='-', maxstep=None, alpha=None):
        """
        BFGS optimizer.

        Parameters:

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.2 Å).

        alpha: float
            Initial guess for the Hessian (curvature of energy surface). A
            conservative value of 70.0 is the default, but number of needed
            steps to converge might be less if a lower value is used. However,
            a lower value also means risk of instability.
        """
        self.reactant = reactant
        self.product = product

        self.hessian_reactant = None
        self.hessian_product = None

        self.pos0_reactant = None
        self.pos0_product = None

        self.forces0_reactant = None
        self.forces0_product = None

        self.max_fr = 0
        self.max_fp = 0

        self.H0 = None

        self.calc_reactant: bool = True

        super().__init__(reactant, restart=None, logfile=logfile, trajectory=None, maxstep=maxstep, master=None,
                         alpha=alpha)

    def initialize(self):
        # initial hessian
        self.H0 = np.eye(3 * len(self.atoms)) * self.alpha

    def step(self, forces=None):
        self.calc_reactant = True
        forces_reactant = self.reactant.get_forces()
        self.max_fr = np.sqrt((forces_reactant ** 2).sum(axis=1).max())
        pos_reactant = self.reactant.get_positions()
        dpos_reactant, steplengths_reactant = self.prepare_step(pos_reactant, forces_reactant)
        dpos_reactant = self.determine_step(dpos_reactant, steplengths_reactant)
        self.reactant.set_positions(pos_reactant + dpos_reactant)

        self.calc_reactant = False
        forces_product = self.product.get_forces()
        self.max_fp = np.sqrt((forces_product ** 2).sum(axis=1).max())
        pos_product = self.product.get_positions()
        dpos_product, steplengths_product = self.prepare_step(pos_product, forces_product)
        dpos_product = self.determine_step(dpos_product, steplengths_product)
        self.product.set_positions(pos_product + dpos_product)

    def prepare_step(self, pos, forces):
        forces = forces.reshape(-1)
        self.update(pos.flat, forces, self.pos0, self.forces0)
        omega, V = np.linalg.eigh(self.H)

        dpos = np.dot(V, np.dot(forces, V) / np.fabs(omega)).reshape((-1, 3))
        # Inexplicably, after several iteration the dpos for the reactant starts being non-uniform, causing part of the
        # molecule to separate from the rest, even though the forces remain uniform (same for every atom)
        steplengths = (dpos ** 2).sum(1) ** 0.5
        self.pos0 = pos.flat.copy()
        self.forces0 = forces.copy()
        return dpos, steplengths

    def converged(self, forces=None):
        """Did the optimization converge?"""
        if forces is None:
            forces = self.reactant.get_forces()
        return (forces ** 2).sum(axis=1).max() < self.fmax ** 2

    def log(self, forces=None):
        try:
            logging.info(f'{self.nsteps}  {self.max_fr}   {self.max_fp}')
        except TypeError:
            pass

    @property
    def pos0(self):
        if self.calc_reactant:
            return self.pos0_reactant
        else:
            return self.pos0_product

    @pos0.setter
    def pos0(self, value):
        if self.calc_reactant:
            self.pos0_reactant = value
        else:
            self.pos0_product = value

    @property
    def forces0(self):
        if self.calc_reactant:
            return self.forces0_reactant
        else:
            return self.forces0_product

    @forces0.setter
    def forces0(self, value):
        if self.calc_reactant:
            self.forces0_reactant = value
        else:
            self.forces0_product = value

    @property
    def H(self):
        if self.calc_reactant:
            return self.hessian_reactant
        else:
            return self.hessian_product

    @H.setter
    def H(self, value):
        if self.calc_reactant:
            self.hessian_reactant = value
        else:
            self.hessian_product = value

    @property
    def atoms(self):
        if self.calc_reactant:
            return self.reactant
        else:
            return self.product

    @atoms.setter
    def atoms(self, value):
        if self.calc_reactant:
            self.reactant = value
        else:
            self.product = value
