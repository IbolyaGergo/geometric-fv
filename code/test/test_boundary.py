import numpy as np
import pytest

from geometric_fv.boundary import apply_bc
from geometric_fv.config import BoundaryConfig, ReconstConfig, SolverConfig
from geometric_fv.enums import BCType
from geometric_fv.mesh import Mesh1D
from geometric_fv.solver import SolverState


# FIXTURE {{{1
@pytest.fixture
def mesh():
    return Mesh1D.uniform(x_min=0.0, x_max=1.0, ncells=50)


@pytest.fixture
def u0(mesh):
    return np.sin(2 * np.pi * mesh.centers)


def create_solver_state(u0, nghost, dt_dx=0.0):
    u_old = np.pad(u0, (nghost, nghost), "constant", constant_values=0.0)
    u_new = np.copy(u_old)
    slope = np.zeros_like(u_old)
    speed=np.zeros_like(u_old),
    flux=np.zeros_like(u_old),
    niter=np.zeros_like(u_old, dtype=int),

    return SolverState(u_old=u_old, u_new=u_new, slope=slope, speed=speed,
                       flux=flux, niter=niter)


# TESTs {{{1
# test_constant_extend_bc() {{{2
@pytest.mark.parametrize("nghost", np.arange(1, 5))
def test_constant_extend_bc(mesh, u0, nghost):
    bc_type = BCType.CONSTANT_EXTEND
    dt_dx = 0.0

    bc_config = BoundaryConfig(bc_type=bc_type)
    reconst_config = ReconstConfig()
    config = SolverConfig(
        boundary=bc_config,
        reconst=reconst_config,
        dt_dx=dt_dx,
    )
    state = create_solver_state(u0, nghost)

    apply_bc(state=state, nghost=nghost, config=config)

    # Left boundary: all ghost cells should match the first physical cell
    assert np.all(state.u_old[:nghost] == u0[0])
    assert np.all(state.u_new[:nghost] == u0[0])

    # Right boundary: all ghost cells should match the last physical cell
    assert np.all(state.u_old[-nghost:] == u0[-1])
    assert np.all(state.u_new[-nghost:] == u0[-1])


# test_apply_bc_quasi_periodic_u_old() {{{2
@pytest.mark.parametrize("nghost", [1, 2])
def test_apply_bc_quasi_periodic_u_old(mesh, u0, nghost):
    bc_type = BCType.QUASI_PERIODIC

    bc_config = BoundaryConfig(bc_type=bc_type)
    reconst_config = ReconstConfig()
    config = SolverConfig(
        boundary=bc_config,
        reconst=reconst_config,
        dt_dx=0.0,
    )
    state = create_solver_state(u0, nghost, dt_dx=0.0)

    apply_bc(state=state, nghost=nghost, config=config)

    if nghost == 1:
        # 0 \\ 1 \ 2 \ ... \ -2 \\ -1
        assert state.u_old[0] == pytest.approx(state.u_old[-2])
        assert state.u_old[-1] == pytest.approx(state.u_old[1])
    elif nghost == 2:
        # 0 \ 1 \\ 2 \ 3 \ ... \ -4 \ -3 \\ -2 \ -1
        assert state.u_old[0] == pytest.approx(state.u_old[-4])
        assert state.u_old[1] == pytest.approx(state.u_old[-3])
        assert state.u_old[-2] == pytest.approx(state.u_old[2])
        assert state.u_old[-1] == pytest.approx(state.u_old[3])


# test_apply_bc_quasi_periodic_u_new_dt_dx_is_whole_positive() {{{2
@pytest.mark.parametrize("nghost", [1, 2])
def test_apply_bc_quasi_periodic_u_new_dt_dx_is_whole_positive(mesh, u0, nghost):
    bc_type = BCType.QUASI_PERIODIC
    bc_config = BoundaryConfig(bc_type=bc_type)
    reconst_config = ReconstConfig()

    for dt_dx in [0.0, 1.0, 2.0]:
        config = SolverConfig(
            boundary=bc_config,
            reconst=reconst_config,
            dt_dx=dt_dx,
        )
        state = create_solver_state(u0, nghost, dt_dx=dt_dx)

        apply_bc(
            state=state,
            nghost=nghost,
            config=config,
        )
        if nghost == 1:
            # 0 \\ 1 \ 2 \ ... \ -2 \\ -1
            assert state.u_new[0] == pytest.approx(state.u_old[-2 - int(dt_dx)])
        elif nghost == 2:
            # 0 \ 1 \\ 2 \ 3 \ ... \ -4 \ -3 \\ -2 \ -1
            assert state.u_new[0] == pytest.approx(state.u_old[-4 - int(dt_dx)])
            assert state.u_new[1] == pytest.approx(state.u_old[-3 - int(dt_dx)])


# test_apply_bc_quasi_periodic_u_new_general_dt_dx_positive() {{{2
@pytest.mark.parametrize("nghost", [1, 2])
def test_apply_bc_quasi_periodic_u_new_general_dt_dx_positive(mesh, u0, nghost):
    bc_type = BCType.QUASI_PERIODIC
    bc_config = BoundaryConfig(bc_type=bc_type)
    reconst_config = ReconstConfig()

    # fmt: off
    for dt_dx in [0.6, 1.5, 2.7]:
        config = SolverConfig(
            boundary=bc_config,
            reconst=reconst_config,
            dt_dx=dt_dx,
        )
        state = create_solver_state(u0, nghost, dt_dx=dt_dx)
        apply_bc(state=state, nghost=nghost, config=config)

        dt_dx_frac = np.mod(dt_dx, 1)
        if nghost == 1:
            # 0 \\ 1 \ 2 \ ... \ -2 \\ -1
            assert state.u_new[0] == pytest.approx(
                (1 - dt_dx_frac) * state.u_old[-2 - int(dt_dx)] +\
                      dt_dx_frac * state.u_old[-3 - int(dt_dx)]
            )
        elif nghost == 2:
            # 0 \ 1 \\ 2 \ 3 \ ... \ -4 \ -3 \\ -2 \ -1
            assert state.u_new[0] == pytest.approx(
                (1 - dt_dx_frac) * state.u_old[-4 - int(dt_dx)] +\
                      dt_dx_frac * state.u_old[-5 - int(dt_dx)]
            )
            assert state.u_new[1] == pytest.approx(
                (1 - dt_dx_frac) * state.u_old[-3 - int(dt_dx)] +\
                      dt_dx_frac * state.u_old[-4 - int(dt_dx)]
            )
    # fmt: on


# test_apply_bc_quasi_periodic_u_new_dt_dx_is_whole_negative() {{{2
@pytest.mark.parametrize("nghost", [1, 2])
def test_apply_bc_quasi_periodic_u_new_dt_dx_is_whole_negative(mesh, u0, nghost):
    bc_type = BCType.QUASI_PERIODIC
    bc_config = BoundaryConfig(bc_type=bc_type)
    reconst_config = ReconstConfig()

    for dt_dx in [-1.0, -2.0, -3.0]:
        config = SolverConfig(
            boundary=bc_config,
            reconst=reconst_config,
            dt_dx=dt_dx,
        )
        state = create_solver_state(u0, nghost, dt_dx=dt_dx)
        apply_bc(state=state, nghost=nghost, config=config)
        if nghost == 1:
            # 0 \\ 1 \ 2 \ ... \ -2 \\ -1
            assert state.u_new[-1] == pytest.approx(state.u_old[1 + int(-dt_dx)])
        elif nghost == 2:
            # 0 \ 1 \\ 2 \ 3 \ ... \ -4 \ -3 \\ -2 \ -1
            assert state.u_new[-1] == pytest.approx(state.u_old[3 + int(-dt_dx)])
            assert state.u_new[-2] == pytest.approx(state.u_old[2 + int(-dt_dx)])


# # params {{{3
# @pytest.mark.parametrize(
#     "dt_dx, target_idx, expected_indices, weights",
#     [
#         # Positive dt_dx (updates left boundary u_new[0])
#         (0.6, 0, (-2, -3), (0.4, 0.6)),
#         (1.5, 0, (-3, -4), (0.5, 0.5)),
#         (2.7, 0, (-4, -5), (0.3, 0.7)),
#         # Negative dt_dx (updates right boundary u_new[-1])
#         (-0.6, -1, (1, 2), (0.4, 0.6)),
#         (-1.5, -1, (2, 3), (0.5, 0.5)),
#         (-2.7, -1, (3, 4), (0.3, 0.7)),
#     ],
# )
# test_apply_bc_quasi_periodic_u_new_dt_dx_is_general_negative() {{{2
@pytest.mark.parametrize("nghost", [1, 2])
def test_apply_bc_quasi_periodic_u_new_dt_dx_is_general_negative(mesh, u0, nghost):
    bc_type = BCType.QUASI_PERIODIC
    bc_config = BoundaryConfig(bc_type=bc_type)
    reconst_config = ReconstConfig()

    for dt_dx in [-0.6, -1.6, -2.7]:
        config = SolverConfig(
            boundary=bc_config,
            reconst=reconst_config,
            dt_dx=dt_dx,
        )
        state = create_solver_state(u0, nghost, dt_dx=dt_dx)
        apply_bc(state=state, nghost=nghost, config=config)

        dt_dx_frac = np.mod(-dt_dx, 1)
        if nghost == 1:
            # 0 \\ 1 \ 2 \ ... \ -2 \\ -1
            assert state.u_new[-1] == pytest.approx(
                (1 - dt_dx_frac) * state.u_old[1 + int(-dt_dx)]
                + dt_dx_frac * state.u_old[2 + int(-dt_dx)]
            )
        elif nghost == 2:
            # 0 \ 1 \\ 2 \ 3 \ ... \ -4 \ -3 \\ -2 \ -1
            assert state.u_new[-1] == pytest.approx(
                (1 - dt_dx_frac) * state.u_old[3 + int(-dt_dx)]
                + dt_dx_frac * state.u_old[4 + int(-dt_dx)]
            )
            assert state.u_new[-2] == pytest.approx(
                (1 - dt_dx_frac) * state.u_old[2 + int(-dt_dx)]
                + dt_dx_frac * state.u_old[3 + int(-dt_dx)]
            )
