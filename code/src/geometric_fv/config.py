from dataclasses import dataclass

from geometric_fv.enums import BCType, LimiterType, SlopeType


@dataclass(frozen=True)
class MeshConfig:
    x_min: float = 0.0
    x_max: float = 1.0
    ncells: int = 100


@dataclass(frozen=True)
class ReconstConfig:
    slope_type: SlopeType = SlopeType.BOX
    limiter_type: LimiterType = LimiterType.TVD


@dataclass(frozen=True)
class BoundaryConfig:
    bc_type: BCType = BCType.QUASI_PERIODIC


@dataclass(frozen=True)
class IterationConfig:
    tol: float = 1e-6
    maxiter: int = 50


@dataclass(frozen=True)
class SolverConfig:
    mesh: MeshConfig = MeshConfig()
    boundary: BoundaryConfig = BoundaryConfig()
    reconst: ReconstConfig = ReconstConfig()
    iteration: IterationConfig = IterationConfig()
