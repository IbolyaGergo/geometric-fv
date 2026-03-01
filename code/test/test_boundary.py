import numpy as np
import pytest

from geometric_fv.boundary import apply_bc
from geometric_fv.config import BoundaryConfig, MeshConfig
from geometric_fv.enums import BCType
from geometric_fv.mesh import Mesh1D
from geometric_fv.solver import SolverState


@pytest.fixture
def mesh():
    config = MeshConfig(x_min=0.0, x_max=1.0, ncells=50)
    return Mesh1D.uniform(config)

@pytest.fixture
def u0(mesh):
    return np.sin(2 * np.pi * mesh.centers)

def create_solver_state(u0, nghost, cfl=0.0):
    u_old = np.pad(u0, (nghost, nghost), "constant", constant_values=0.0)
    u_new = np.copy(u_old)
    slope = np.zeros_like(u_old)

    return SolverState(u_old=u_old, u_new=u_new, slope=slope, cfl=cfl)

@pytest.mark.parametrize("nghost", np.arange(1, 5))
def test_constant_extend_bc(mesh, u0, nghost):
    config = BoundaryConfig(bc_type=BCType.CONSTANT_EXTEND)
    state = create_solver_state(u0, nghost)

    apply_bc(state, config, nghost)

    # Left boundary: all ghost cells should match the first physical cell
    assert np.all(state.u_old[:nghost] == u0[0])
    assert np.all(state.u_new[:nghost] == u0[0])

    # Right boundary: all ghost cells should match the last physical cell
    assert np.all(state.u_old[-nghost:] == u0[-1])
    assert np.all(state.u_new[-nghost:] == u0[-1])

@pytest.mark.parametrize("nghost", [1, 2])
def test_apply_bc_quasi_periodic_ghost_cells(mesh, u0, nghost):
    config = BoundaryConfig(bc_type=BCType.QUASI_PERIODIC)
    state = create_solver_state(u0, nghost)

    apply_bc(state, config, nghost)

    if nghost == 1:
        # 0 \\ 1 \ 2 \ ... \ -2 \\ -1
        assert state.u_old[0] == state.u_old[-1 - nghost]
        assert state.u_old[-1] == state.u_old[1]
    elif nghost == 2:
        # 0 \ 1 \\ 2 \ 3 \ ... \ -4 \ -3 \\ -2 \ -1
        assert state.u_old[0] == state.u_old[-4]
        assert state.u_old[1] == state.u_old[-3]
        assert state.u_old[-2] == state.u_old[2]
        assert state.u_old[-1] == state.u_old[3]

@pytest.mark.parametrize(
    "cfl, target_idx, expected_indices, weights",
    [
        # Positive CFL (updates left boundary u_new[0])
        (0.6, 0, (-2, -3), (0.4, 0.6)),
        (1.5, 0, (-3, -4), (0.5, 0.5)),
        (2.7, 0, (-4, -5), (0.3, 0.7)),
        # Negative CFL (updates right boundary u_new[-1])
        (-0.6, -1, (1, 2), (0.4, 0.6)),
        (-1.5, -1, (2, 3), (0.5, 0.5)),
        (-2.7, -1, (3, 4), (0.3, 0.7)),
    ],
)
def test_quasi_periodic_bc_cfl(mesh, u0, cfl, target_idx, expected_indices, weights):
    """Verifies u_new and slope calculations for various CFL values."""
    nghost = 1
    config = BoundaryConfig(bc_type=BCType.QUASI_PERIODIC)
    state = create_solver_state(u0, nghost=nghost, cfl=cfl)

    apply_bc(state, config, nghost)

    i1, i2 = expected_indices
    w1, w2 = weights
    expected_u_new = w1 * state.u_old[i1] + w2 * state.u_old[i2]
    assert state.u_new[target_idx] == pytest.approx(expected_u_new)

    expected_slope = (state.u_old[0] - state.u_new[0]) / cfl
    assert state.slope[0] == pytest.approx(expected_slope)
