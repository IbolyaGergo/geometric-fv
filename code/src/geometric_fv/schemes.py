from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from geometric_fv.slope import SlopeType, compute_slope
from geometric_fv.utils import simple_fixed_point


class Scheme(ABC):
    nghost: int

    @abstractmethod
    def sweep(self, u_old: np.ndarray, u_new: np.ndarray, cfl: float):
        pass


@dataclass(frozen=True)
class ImplicitUpwind(Scheme):
    nghost: int = 1

    def sweep(self, u_old: np.ndarray, u_new: np.ndarray, cfl: float):
        nghost = self.nghost
        if cfl > 0:
            for i in range(nghost, len(u_old) - nghost):
                u_new[i] = (u_old[i] + cfl * u_new[i - 1]) / (1.0 + cfl)
        else:
            for i in reversed(range(nghost, len(u_old) - nghost)):
                u_new[i] = (u_old[i] + (-cfl) * u_new[i + 1]) / (1.0 + (-cfl))


@dataclass(frozen=True)
class Box(Scheme):
    nghost: int = 1

    def sweep(self, u_old: np.ndarray, u_new: np.ndarray, cfl: float):
        nghost = self.nghost
        coeff = (1 - cfl) / (1 + cfl)
        for i in range(nghost, len(u_old) - nghost):
            u_new[i] = coeff * u_old[i] + u_old[i - 1] - coeff * u_new[i - 1]


@dataclass(frozen=True)
class SecondOrderImplicit(Scheme):
    nghost: int = 1
    slope_type: SlopeType = SlopeType.TVD_BOX
    tol: float = 1e-6
    maxiter: int = 50

    def _update_cell_guess(
        self, u_old: np.ndarray, u_new: np.ndarray, cfl: float, i: int
    ) -> float:
        coeff = (1 - cfl) / (1 + cfl)
        u_new_i_guess = coeff * u_old[i] + u_old[i - 1] - coeff * u_new[i - 1]
        if self.slope_type is SlopeType.TVD_BOX:
            u_new_i_guess = np.median([u_new_i_guess, u_old[i], u_new[i-1]])

        return u_new_i_guess

    def _update_cell_iter(
        self,
        u_new_i_current: float,
        u_old: np.ndarray,
        u_new: np.ndarray,
        slope: np.ndarray,
        cfl: float,
        i: int,
    ) -> float:
        slope[i] = compute_slope(
                u_old=u_old,
                u_new=u_new,
                slope=slope,
                cfl=cfl,
                i=i,
                u_new_i_current=u_new_i_current,
                slope_type=self.slope_type
        )

        # fmt: off
        u_new_i_next = (u_old[i] + cfl * u_new[i - 1]) / (1.0 + cfl) \
                - 0.5 * cfl * (slope[i] - slope[i - 1])
        # fmt: on
        return u_new_i_next

    def sweep(self, u_old: np.ndarray, u_new: np.ndarray, cfl: float):
        nghost = self.nghost
        slope = np.zeros(len(u_old))
        for i in range(nghost, len(u_old) - nghost):
            u_new_i_guess = self._update_cell_guess(
                    u_old=u_old,
                    u_new=u_new,
                    cfl=cfl,
                    i=i
            )

            result = simple_fixed_point(
                self._update_cell_iter,
                u_new_i_guess,
                args=(u_old, u_new, slope, cfl, i),
                tol=self.tol,
                maxiter=self.maxiter,
            )
            if result.success:
                u_new[i] = result.x

                niters = result.nit
                print(f"Cell {i} converged in {niters} number of iterations.")
            else:
                print(f"Warning: Solver failed to converge at cell {i}.")
                print(f"Message: {result.message}")

                u_new[i] = u_new_i_guess

            # slope[i] = compute_slope(u_old, u_new, cfl, i, slope_type=self.slope_type)
