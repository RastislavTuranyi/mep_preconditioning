from __future__ import annotations

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

        self.H0 = None

        self.calc_reactant: bool = True

        super().__init__(ase.Atoms(), restart=None, logfile=logfile, trajectory=None, maxstep=maxstep, master=None,
                         alpha=alpha)

    def initialize(self):
        # initial hessian
        self.H0 = np.eye(3 * len(self.atoms)) * self.alpha

    def step(self, forces=None):
        reactant = self.reactant
        product = self.product

        forces_reactant = reactant.get_forces()
        forces_product = product.get_forces()

        pos_reactant = reactant.get_positions()
        pos_product = product.get_positions()

        self.calc_reactant = True
        dpos_reactant, steplengths_reactant = self.prepare_step(pos_reactant, forces_reactant)
        dpos_reactant = self.determine_step(dpos_reactant, steplengths_reactant)
        reactant.set_positions(pos_reactant + dpos_reactant)

        self.calc_reactant = False
        dpos_product, steplengths_product = self.prepare_step(pos_product, forces_product)
        dpos_product = self.determine_step(dpos_product, steplengths_product)
        reactant.set_positions(pos_product + dpos_product)

    def prepare_step(self, pos, forces):
        forces = forces.reshape(-1)
        self.update(pos.flat, forces, self.pos0, self.forces0)
        omega, V = np.eigh(self.H)

        dpos = np.dot(V, np.dot(forces, V) / np.fabs(omega)).reshape((-1, 3))
        steplengths = (dpos ** 2).sum(1) ** 0.5
        self.pos0 = pos.flat.copy()
        self.forces0 = forces.copy()
        return dpos, steplengths

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
