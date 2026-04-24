from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from geometric_fv.config import SolverConfig
from geometric_fv.equations import LinearAdvection
from geometric_fv.reconst import compute_flux_corr, compute_guess, compute_slope
from geometric_fv.solver import SolverState
from geometric_fv.utils import simple_fixed_point


# Scheme() {{{1
class Scheme(ABC):
    nghost: int
    config: SolverConfig

    def __init__(self, config: SolverConfig):
        self.config = config

    # allocate_state() {{{2
    def allocate_state(self, u0: np.ndarray) -> SolverState:
        """Creates a SolverState from an existing array."""
        u_padded = np.pad(u0, (self.nghost, self.nghost), mode="constant")
        return SolverState(
            u_old=u_padded.copy(),
            u_new=u_padded.copy(),
            slope=np.zeros_like(u_padded),
            speed=np.zeros_like(u_padded),
            flux=np.zeros_like(u_padded),
            niter=np.zeros_like(u_padded, dtype=int),
        )

    # init_state() {{{2
    def init_state(self, func: Callable[[np.ndarray], np.ndarray]) -> SolverState:
        """
        Generates u0 using func and the mesh defined in the scheme's config.
        """
        mesh = self.config.mesh.create_mesh()

        u0 = func(mesh.centers)

        return self.allocate_state(u0)

    # apply_bc() {{{2
    def apply_bc(self, state: SolverState) -> None:
        # Local import to avoid circular dependency at the top of the file
        from geometric_fv.boundary import apply_bc as _apply_bc_kernel

        _apply_bc_kernel(
            state=state,
            nghost=self.nghost,
            config=self.config,
        )

    # cell_indices() {{{2
    def cell_indices(self, state: SolverState, sweep_sign: int) -> range | reversed[int]:
        """
        Returns an iterator over the internal cell indices.

        Args:
            state: The current state to get the total length.
            sweep_sign: 1 for left-to-right (positive), -1 for right-to-left
            (negative).
        """
        ntotal = len(state.u_old)
        nghost = self.nghost

        # Define the range of physical (non-ghost) cells
        idx_range = range(nghost, ntotal - nghost)

        if sweep_sign == -1:
            return reversed(idx_range)
        return idx_range

    # sweep() {{{2
    @abstractmethod
    def sweep(self, state: SolverState):
        pass


# SecondOrderImplicit() {{{1
@dataclass(frozen=True)
class SecondOrderImplicit(Scheme):
    nghost: int = 2
    config: SolverConfig = SolverConfig(equation=LinearAdvection())

    # _update_cell_iter() {{{2
    def _update_cell_iter(
        self,
        u_new_i_current: float,
        state: SolverState,
        i: int,
    ) -> float:
        u_old = state.u_old
        u_new = state.u_new
        slope = state.slope

        dt_dx = self.config.dt_dx
        eq = self.config.equation

        slope_i_current = compute_slope(
            state, i=i, u_new_i=u_new_i_current, config=self.config
        )

        # fmt: off
        cfl = eq.dfdu(u_new_i_current) * dt_dx
        i_upw = i-1 if cfl > 0 else i+1
        u_new_i_next = \
                (u_old[i] + abs(cfl) * u_new[i_upw]) / (1.0 + abs(cfl)) \
                - 0.5 * cfl * (slope_i_current - slope[i_upw])
        # fmt: on
        return u_new_i_next

    # sweep() {{{2
    def sweep(self, state: SolverState):
        if state.niter is None:
            state.niter = np.zeros_like(state.u_old, dtype=int)

        eq = self.config.equation
        cfl = eq.dfdu(state.u_old[0]) * self.config.dt_dx
        sweep_sign = 1 if cfl > 0 else -1
        for i in self.cell_indices(state, sweep_sign):
            u_new_i_guess = compute_guess(state, i=i, config=self.config)

            result = simple_fixed_point(
                self._update_cell_iter,
                u_new_i_guess,
                args=(state, i),
                tol=self.config.iteration.tol,
                maxiter=self.config.iteration.maxiter,
            )
            if result.success:
                state.u_new[i] = result.x
                state.slope[i] = compute_slope(state, i, result.x, self.config)
                state.niter[i] = result.nit

                # print(f"Cell {i} converged in {state.niter[i]} number of iterations.")
            else:
                print(f"Warning: Solver failed to converge at cell {i}.")
                print(f"Message: {result.message}")

                state.u_new[i] = u_new_i_guess


# HighResImplicit() {{{1
@dataclass(frozen=True)
class HighResImplicit(Scheme):
    nghost: int = 2
    config: SolverConfig = SolverConfig()

    # _compute_num_flux() {{{2
    def _compute_num_flux(self, u_curr: float, state: SolverState, i: int) -> float:
        u_old = state.u_old

        eq = self.config.equation
        dt_dx = self.config.dt_dx
        tol = self.config.iteration.tol

        flux_out_corr = compute_flux_corr(state, i, u_curr, self.config)
        flux_in = state.flux[i - 1]

        rhs = u_old[i] + dt_dx * (flux_in - flux_out_corr)
        res = eq.invert_implicit(rhs, dt_dx, tol)

        if res.is_invertible:
            flux_out = eq.flux(res.u) + flux_out_corr
        else:
            flux_out = 0.0

        return flux_out

    # _compute_update() {{{2
    def _compute_update(self, u_curr: float, state: SolverState, i: int) -> float:
        """
        Args:
            u_curr: Current iterate for cell i.
            state: Full solver state containing old values, slopes, fluxes.
            i: Index of the cell being updated.

        Returns:
            u_next.
        """
        u_old = state.u_old

        dt_dx = self.config.dt_dx

        flux_in = state.flux[i - 1]
        flux_out = self._compute_num_flux(u_curr, state, i)

        u_next = u_old[i] - dt_dx * (flux_out - flux_in)

        return u_next

    # sweep() {{{2
    def sweep(self, state: SolverState):
        state.flux[self.nghost - 1] = self.config.equation.flux(
            state.u_new[self.nghost - 1]
        )
        for i in self.cell_indices(state, sweep_sign=1):
            u_new_i_guess = state.u_new[i - 1]

            result = simple_fixed_point(
                self._compute_update,
                u_new_i_guess,
                args=(state, i),
                tol=self.config.iteration.tol,
                maxiter=self.config.iteration.maxiter,
            )

            if result.success:
                state.u_new[i] = result.x
                state.flux[i] = self._compute_num_flux(state.u_new[i], state, i)
                state.slope[i] = compute_slope(state, i, result.x, self.config)
                state.niter[i] = result.nit

                # print(f"Cell {i} converged in {state.niter[i]} number of iterations.")
            else:
                print(f"Warning: Solver failed to converge at cell {i}.")
                print(f"Message: {result.message}")

                state.u_new[i] = u_new_i_guess
                state.flux[i] = self._compute_num_flux(u_new_i_guess, state, i)


# Lozano() {{{1
@dataclass(frozen=True)
class Lozano(Scheme):
    nghost: int = 2
    config: SolverConfig = SolverConfig()

    # _compute_num_flux() {{{2
    def _compute_num_flux(self, u: float, sweep_sign: int) -> float:
        eq = self.config.equation
        if (sweep_sign == 1 and u > 0.0) or (sweep_sign == -1 and u < 0.0):
            return eq.flux(u)
        return 0.0

    # _update_cell() {{{2
    def _update_cell(self, state: SolverState, i: int, sweep_sign: int) -> float:
        u_old = state.u_old
        u_new = state.u_new
        dt_dx = self.config.dt_dx
        eq = self.config.equation
        tol = self.config.iteration.tol

        u_base = u_old[i] if sweep_sign == 1 else u_new[i]
        rhs = u_base + (sweep_sign) * dt_dx *\
                self._compute_num_flux(u_new[i - sweep_sign], sweep_sign)

        if sweep_sign * rhs > 0.0:
            res = eq.invert_implicit(rhs, dt_dx, tol, sweep_sign)
            return res.u
        return rhs

    # sweep() {{{2
    def sweep(self, state: SolverState):
        for sign in [1, -1]:
            for i in self.cell_indices(state, sweep_sign=sign):
                state.u_new[i] = self._update_cell(state, i, sweep_sign=sign)


# BoxBurgers() {{{1
@dataclass(frozen=True)
class BoxBurgers(Scheme):
    nghost: int = 2
    config: SolverConfig = SolverConfig()

    # _flux_pos() {{{2
    def _flux_pos(self, state: SolverState, i: int) -> float:
        u_old = state.u_old
        u_new = state.u_new
        dt_dx = self.config.dt_dx

        if state.u_new[i] > 0.0:
            return 0.5 * (u_new[i] * (u_old[i] - 1 / dt_dx) + u_old[i] / dt_dx)
        else:
            return 0.0

    # _update_cell() {{{2
    def _update_cell(self, state: SolverState, i: int, sweep_sign: str) -> float:
        u_old = state.u_old
        dt_dx = self.config.dt_dx

        u_pos = (u_old[i] + 2 * dt_dx * self._flux_pos(state, i - 1)) / (
            1 + dt_dx * u_old[i]
        )
        if u_pos > 0.0:
            return u_pos
        else:
            return u_old[i] + dt_dx * self._flux_pos(state, i - 1)

    # sweep() {{{2
    def sweep(self, state: SolverState):
        for i in self.cell_indices(state, sweep_sign=1):
            state.u_new[i] = self._update_cell(state, i, sweep_sign="pos")
