from abc import ABC, abstractmethod

import numpy as np


# Equation {{{1
class Equation(ABC):
    # flux {{{2
    @abstractmethod
    def flux(self, u: float) -> float:
        """Compute the physical flux f(u)."""
        pass

    # dfdu {{{2
    @abstractmethod
    def dfdu(self, u: float) -> float:
        """Compute the physical speed f'(u)."""
        pass

    # speed {{{2
    def speed(self, u1: float, u2: float) -> float:
        """Compute the numerical wave speed."""
        du = u2 - u1
        if abs(du) < 1e-14:
            return self.dfdu(u1)
        return (self.flux(u2) - self.flux(u1)) / du

    # solve_for_u {{{2
    def solve_for_u(self, rhs: float, dt_dx: float) -> float:
        """
        Solves the implicit equation u + dt/dx * f(u) = rhs for u.
        """
        raise NotImplementedError

    def initial_guess(self, u_old_i: float, u_upw: float, dt_dx: float) -> float:
        """
        Provides a first-order implicit upwind guess.
        Solves: u + dt/dx * f(u) = u_old + dt/dx * f(u_upw)
        """
        rhs = u_old_i + dt_dx * self.flux(u_upw)
        return self.solve_for_u(rhs, dt_dx)
# Burgers {{{1
class Burgers(Equation):
    def flux(self, u: float) -> float:
        return 0.5 * u**2

    def dfdu(self, u: float) -> float:
        return u

    def solve_for_u(self, rhs: float, dt_dx: float) -> float:
        # u + dt/dx * (u^2/2) = rhs => 0.5 * dt/dx * u^2 + u - rhs = 0
        if abs(dt_dx) < 1e-14:
            return rhs
        return (-1.0 + np.sqrt(1.0 + 2.0 * dt_dx * rhs)) / dt_dx

# LinearAdvection {{{1
class LinearAdvection(Equation):
    def __init__(self, a: float = 1.0):
        self.a = a

    def flux(self, u: float) -> float:
        return self.a * u

    def dfdu(self, u: float) -> float:
        return self.a

    def solve_for_u(self, rhs: float, dt_dx: float) -> float:
        # u + dt/dx * (a * u) = rhs => u = rhs / (1 + a * dt/dx)
        if abs(dt_dx) < 1e-14:
            return rhs
        return rhs / (1.0 + self.a * dt_dx)
