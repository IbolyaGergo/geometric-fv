"""A module for creating and managing 1D computational meshes."""

from dataclasses import dataclass

import numpy as np

from geometric_fv.config import MeshConfig


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
    def uniform(cls, mesh_config: MeshConfig) -> "Mesh1D":
        """Create a uniform 1D mesh.

        Parameters
        ----------
        mesh_config
            For setting boundary coordinates and number of cells.

        Returns
        -------
        Mesh1D
            A new Mesh1D object with uniform cell spacing.
        """
        x_min = mesh_config.x_min
        x_max = mesh_config.x_max
        ncells = mesh_config.ncells
        return cls(np.linspace(x_min, x_max, ncells + 1))
