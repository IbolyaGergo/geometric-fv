from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from geometric_fv.slope import SlopeType, compute_slope
from geometric_fv.solver import SolverState
from geometric_fv.utils import simple_fixed_point


class Scheme(ABC):
    nghost: int

    @abstractmethod
    def sweep(self, state):
        pass


@dataclass(frozen=True)
class ImplicitUpwind(Scheme):
    nghost: int = 1

    def sweep(self, state):
        nghost = self.nghost

        cfl = state.get_cfl()
        if cfl > 0:
            for i in range(nghost, len(state.u_old) - nghost):
                u_old_i = state.get_u_old(i)
                u_old_im1 = state.get_u_old(i - 1)
                u_new_im1 = state.get_u_new(i - 1)
                u_new_ip1 = state.get_u_new(i + 1)

                state.u_new[i] = (u_old_i + cfl * u_new_im1) / (1.0 + cfl)
        else:
            for i in reversed(range(nghost, len(state.u_old) - nghost)):
                u_old_i = state.get_u_old(i)
                u_old_im1 = state.get_u_old(i - 1)
                u_new_im1 = state.get_u_new(i - 1)
                u_new_ip1 = state.get_u_new(i + 1)

                state.u_new[i] = (u_old_i + (-cfl) * u_new_ip1) / (1.0 + (-cfl))

@dataclass(frozen=True)
class SecondOrderImplicit(Scheme):
    nghost: int = 1
    slope_type: SlopeType = SlopeType.TVD_BOX
    tol: float = 1e-6
    maxiter: int = 50

    def _update_cell_guess(
        self, state, i: int
    ) -> float:
        u_old_i = state.get_u_old(i)
        u_old_im1 = state.get_u_old(i - 1)
        u_new_im1 = state.get_u_new(i - 1)
        cfl = state.get_cfl()

        coeff = (1 - cfl) / (1 + cfl)
        u_new_i_guess = coeff * u_old_i + u_old_im1 - coeff * u_new_im1
        if self.slope_type is SlopeType.TVD_BOX:
            u_new_i_guess = np.median([u_new_i_guess, u_old_i, u_new_im1])

        return u_new_i_guess

    def _update_cell_iter(
        self,
        u_new_i_current: float,
        state,
        i: int,
    ) -> float:
        slope_i = compute_slope(
                state,
                i=i,
                u_new_i_current=u_new_i_current,
                slope_type=self.slope_type
        )
        state.set_slope(i, slope_i)

        u_old_i = state.get_u_old(i)
        u_old_im1 = state.get_u_old(i - 1)
        u_new_im1 = state.get_u_new(i - 1)
        slope_im1 = state.get_slope(i - 1)
        cfl = state.get_cfl()

        # fmt: off
        u_new_i_next = (u_old_i + cfl * u_new_im1) / (1.0 + cfl) \
                - 0.5 * cfl * (state.slope[i] - slope_im1)
        # fmt: on
        return u_new_i_next

    def sweep(self, state):
        nghost = self.nghost

        for i in range(nghost, len(state.u_old) - nghost):
            u_new_i_guess = self._update_cell_guess(
                    state,
                    i=i
            )

            result = simple_fixed_point(
                self._update_cell_iter,
                u_new_i_guess,
                args=(state, i),
                tol=self.tol,
                maxiter=self.maxiter,
            )
            if result.success:
                state.u_new[i] = result.x

                niters = result.nit
                print(f"Cell {i} converged in {niters} number of iterations.")
            else:
                print(f"Warning: Solver failed to converge at cell {i}.")
                print(f"Message: {result.message}")

                state.u_new[i] = u_new_i_guess

            # slope[i] = compute_slope(u_old, u_new, cfl, i, slope_type=self.slope_type)
