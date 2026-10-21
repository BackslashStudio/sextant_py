"""matplotlib.patches over sextant: ``from sextant.patches import Circle`` and ``ax.add_patch(...)``
on a ``sextant.pyplot`` axes. The shapes sextant draws: a circle, a wedge (ring segment) and a polygon."""

from ._mpl import Circle, Polygon, Wedge

__all__ = ["Circle", "Polygon", "Wedge"]
