import numpy as np
import pytest

from geometric_fv.config import MeshConfig
from geometric_fv.mesh import Mesh1D


@pytest.mark.parametrize(
    ("x_min", "x_max", "ncells"),
    [
        (1.0, 0.0, 10),  # x_max < x_min
        (0.0, 1.0, 0),  # ncells = 0
        (0.0, 1.0, -1),  # ncells < 0
    ],
)
def test_mesh_uniform_raises_error_for_invalid_inputs(x_min, x_max, ncells):
    with pytest.raises(ValueError):
        Mesh1D.uniform(MeshConfig(x_min=x_min, x_max=x_max, ncells=ncells))


@pytest.mark.parametrize(
    ("x_min", "x_max", "ncells"),
    [
        (0.0, 1.0, 1),
        (0.0, 1.0, 10),
        (-1.0, 1.0, 20),
        (10.0, 20.0, 5),
    ],
)
def test_mesh_uniform_valid_inputs(x_min, x_max, ncells):
    mesh = Mesh1D.uniform(MeshConfig(x_min=x_min, x_max=x_max, ncells=ncells))

    # Check number of cells
    assert mesh.ncells == ncells

    # Check faces
    expected_faces = np.linspace(x_min, x_max, ncells + 1)
    np.testing.assert_allclose(mesh.faces, expected_faces)

    # Check dx
    expected_dx = (x_max - x_min) / ncells
    assert mesh.dx[0] == pytest.approx(expected_dx)

    # Check centers
    expected_centers = x_min + (np.arange(ncells) + 0.5) * expected_dx
    np.testing.assert_allclose(mesh.centers, expected_centers)
