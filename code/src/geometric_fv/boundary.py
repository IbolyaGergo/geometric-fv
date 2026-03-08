import numpy as np

from geometric_fv.config import BoundaryConfig, ReconstConfig
from geometric_fv.enums import BCType
from geometric_fv.slope import compute_slope
from geometric_fv.solver import SolverState


def _apply_bc_constant_extend(state: SolverState, nghost: int) -> None:
    u_old = state.u_old
    u_new = state.u_new

    # first/last idx of the physical domain
    first = nghost
    last = -nghost - 1

    for i in range(nghost):
        u_old[i] = u_old[first]
        u_new[i] = u_old[first]
        u_old[-1 - i] = u_old[last]
        u_new[-1 - i] = u_old[last]


# nghost = 1
# 0 \\ 1 \ 2 \ ... \ -2 \\ -1
# for i in [0]:
#     u_old[0] = u_old[-2]
#     u_old[-1] = u_old[1]
# nghost = 2
# 0 \ 1 \\ 2 \ 3 \ ... \ -4 \ -3 \\ -2 \ -1
# for i in [0, 1]:
#     u_old[0] = u_old[-4]
#     u_old[1] = u_old[-3]
#     u_old[-1] = u_old[3]
#     u_old[-2] = u_old[2]
def _apply_bc_quasi_periodic(state: SolverState, nghost: int) -> None:
    u_old = state.u_old
    u_new = state.u_new
    cfl = state.cfl

    # first/last idx of the physical domain
    first = nghost
    last = -nghost - 1
    for i in range(nghost):
        # Old values
        u_old[first - 1 - i] = u_old[last - i]
        u_old[last + 1 + i] = u_old[first + i]

        # New values
        # fmt: off
        if cfl > 0.0 or np.isclose(cfl, 0.0):
            cfl_frac = np.mod(cfl, 1)
            u_new[first - 1 - i] = (1 - cfl_frac) * u_old[last - i - int(cfl)] \
                           + cfl_frac * u_old[last - i - 1 - int(cfl)]
        else:
            cfl_frac = np.mod(-cfl, 1)
            u_new[last + 1 + i] = (1 - cfl_frac) * u_old[first + i + int(-cfl)] \
                                      + cfl_frac * u_old[first + i + 1 + int(-cfl)]
        # fmt: on


_apply_bc_types = {
    BCType.CONSTANT_EXTEND: _apply_bc_constant_extend,
    BCType.QUASI_PERIODIC: _apply_bc_quasi_periodic,
}


def apply_bc(
    state: SolverState,
    nghost: int,
    config: BoundaryConfig,
    reconst_config: ReconstConfig,
) -> None:
    bc_type = config.bc_type
    apply_bc_func = _apply_bc_types.get(bc_type)
    if apply_bc_func is None:
        raise ValueError(f"Unsupported BC type: {bc_type}")
    apply_bc_func(state, nghost)

    u_new = state.u_new
    slope = state.slope
    cfl = state.cfl

    # first/last idx of the physical domain
    first = nghost
    last = -nghost - 1
    if cfl > 0.0:
        i = first - 1
    else:
        i = last + 1
    slope[i] = compute_slope(
        state=state,
        i=i,
        u_new_i=u_new[i],
        reconst_config=reconst_config,
    )
