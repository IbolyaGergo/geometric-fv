from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from geometric_fv.enums import BCType, GuessType, LimiterType, SlopeType
from geometric_fv.equations import Equation, LinearAdvection

if TYPE_CHECKING:
    from geometric_fv.mesh import Mesh1D


# MeshConfig {{{1
@dataclass(frozen=True)
class MeshConfig:
    """
    x_min : float
        The minimum coordinate of the mesh.
    x_max : float
        The maximum coordinate of the mesh.
    ncells : int
        The number of cells in the mesh.

    Raises
    ------
    ValueError
        If `x_max` is not greater than `x_min` or if `ncells` is less than 1.
    """

    x_min: float = 0.0
    x_max: float = 1.0
    ncells: int = 100

    def __post_init__(self):
        if self.x_min > self.x_max:
            raise ValueError("x_max must be greater than x_min")
        if self.ncells <= 0:
            raise ValueError("ncells must be greater than 0")

    def create_mesh(self) -> Mesh1D:
        from geometric_fv.mesh import Mesh1D

        return Mesh1D.uniform(self.x_min, self.x_max, self.ncells)


# ReconstConfig {{{1
@dataclass(frozen=True)
class ReconstConfig:
    slope_type: SlopeType = SlopeType.BOX
    limiter_type: LimiterType = LimiterType.NONE
    guess_type: GuessType = GuessType.BOX


# BoundaryConfig {{{1
@dataclass(frozen=True)
class BoundaryConfig:
    bc_type: BCType = BCType.QUASI_PERIODIC


# IterationConfig {{{1
@dataclass(frozen=True)
class IterationConfig:
    tol: float = 1e-9
    maxiter: int = 50


# SolverConfig {{{1
@dataclass(frozen=True)
class SolverConfig:
    mesh: MeshConfig = MeshConfig()
    boundary: BoundaryConfig = BoundaryConfig()
    reconst: ReconstConfig = ReconstConfig()
    iteration: IterationConfig = IterationConfig()
    equation: Equation = LinearAdvection()
