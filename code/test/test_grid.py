import numpy as np
import pytest

from geometric_fv.grid import Grid1D


@pytest.mark.parametrize(
    ("x_min", "x_max", "ncells"),
    [
        (1.0, 0.0, 10),  # x_max < x_min
        (0.0, 1.0, 0),  # ncells = 0
        (0.0, 1.0, -1),  # ncells < 0
    ],
)
def test_grid_uniform_raises_error_for_invalid_inputs(x_min, x_max, ncells):
    with pytest.raises(ValueError):
        Grid1D.uniform(x_min=x_min, x_max=x_max, ncells=ncells)


@pytest.mark.parametrize(
    ("x_min", "x_max", "ncells"),
    [
        (0.0, 1.0, 1),
        (0.0, 1.0, 10),
        (-1.0, 1.0, 20),
        (10.0, 20.0, 5),
    ],
)
def test_grid_uniform_valid_inputs(x_min, x_max, ncells):
    grid = Grid1D.uniform(x_min=x_min, x_max=x_max, ncells=ncells)

    # Check number of cells
    assert grid.ncells == ncells

    # Check faces
    expected_faces = np.linspace(x_min, x_max, ncells + 1)
    np.testing.assert_allclose(grid.faces, expected_faces)

    # Check dx
    expected_dx = (x_max - x_min) / ncells
    assert grid.dx[0] == pytest.approx(expected_dx)

    # Check centers
    expected_centers = x_min + (np.arange(ncells) + 0.5) * expected_dx
    np.testing.assert_allclose(grid.centers, expected_centers)
