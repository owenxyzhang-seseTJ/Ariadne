"""Independent, auditable small-molecule crystallographic algorithms.

Native numerical capabilities are described in capabilities.py and the
technical report. SHELXT/SHELXL are separately installed optional backends.
"""

__version__ = "0.1.0"

from .model import Atom, Cell, Reflections, Structure
from .scattering import calculated_intensities, structure_factors

__all__ = ["Atom", "Cell", "Reflections", "Structure", "calculated_intensities", "structure_factors"]

