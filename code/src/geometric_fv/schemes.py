from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from geometric_fv.config import SolverConfig
from geometric_fv.enums import LimiterType
from geometric_fv.slope import compute_slope
from geometric_fv.solver import SolverState
from geometric_fv.utils import simple_fixed_point


class Scheme(ABC):
    nghost: int
    config: SolverConfig

    def apply_bc(self, state: SolverState) -> None:
        # Local import to avoid circular dependency at the top of the file
        from geometric_fv.boundary import apply_bc as _apply_bc_kernel

        _apply_bc_kernel(
            state=state,
            nghost=self.nghost,
            config=self.config.boundary,
            reconst_config=self.config.reconst,
        )

    @abstractmethod
    def sweep(self, state: SolverState):
        pass


@dataclass(frozen=True)
class SecondOrderImplicit(Scheme):
    nghost: int = 2
    config: SolverConfig = SolverConfig()

    def _update_cell_guess(self, state: SolverState, i: int) -> float:
        u_old = state.u_old
        u_new = state.u_new
        cfl = state.cfl

        coeff = (1 - cfl) / (1 + cfl)
        u_new_i_guess = coeff * u_old[i] + u_old[i - 1] - coeff * u_new[i - 1]

        limiter_type = self.config.reconst.limiter_type
        if limiter_type is not LimiterType.NONE:
            u_new_i_guess = np.median([u_new_i_guess, u_old[i], u_new[i - 1]])

        return u_new_i_guess

    def _update_cell_iter(
        self,
        u_new_i_current: float,
        state: SolverState,
        i: int,
    ) -> float:
        u_old = state.u_old
        u_new = state.u_new
        slope = state.slope

        cfl = state.cfl

        slope[i] = compute_slope(
            state, i=i, u_new_i=u_new_i_current, reconst_config=self.config.reconst
        )

        # fmt: off
        u_new_i_next = (u_old[i] + cfl * u_new[i - 1]) / (1.0 + cfl) \
                - 0.5 * cfl * (slope[i] - slope[i - 1])
        # fmt: on
        return u_new_i_next

    def sweep(self, state: SolverState):
        nghost = self.nghost

        for i in range(nghost, len(state.u_old) - nghost):
            u_new_i_guess = self._update_cell_guess(state, i=i)

            result = simple_fixed_point(
                self._update_cell_iter,
                u_new_i_guess,
                args=(state, i),
                tol=self.config.iteration.tol,
                maxiter=self.config.iteration.maxiter,
            )
            if result.success:
                state.u_new[i] = result.x

                # niters = result.nit
                # print(f"Cell {i} converged in {niters} number of iterations.")
            else:
                print(f"Warning: Solver failed to converge at cell {i}.")
                print(f"Message: {result.message}")

                state.u_new[i] = u_new_i_guess
