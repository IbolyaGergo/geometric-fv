"""A module for creating and managing 1D computational meshes."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mesh1D:
    """A 1D mesh with faces and cell centers.

    This class represents a 1D mesh and provides properties to access cell
    centers, cell sizes, and the number of cells. It is immutable to prevent
    accidental modification.

    Parameters
    ----------
    faces : np.ndarray
        An array of face locations.
    """

    faces: np.ndarray

    @property
    def centers(self) -> np.ndarray:
        """The locations of the cell centers."""
        return 0.5 * (self.faces[:-1] + self.faces[1:])

    @property
    def dx(self) -> np.ndarray:
        """The widths of the cells."""
        return np.diff(self.faces)

    @property
    def ncells(self) -> int:
        """The number of cells in the mesh."""
        return len(self.faces) - 1

    @classmethod
    def uniform(cls, x_min: float, x_max: float, ncells: int) -> "Mesh1D":
        """Create a uniform 1D mesh.
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
        if x_min > x_max:
            raise ValueError("x_max must be greater than x_min")
        if ncells <= 0:
            raise ValueError("ncells must be greater than 0")
        return cls(np.linspace(x_min, x_max, ncells + 1))
