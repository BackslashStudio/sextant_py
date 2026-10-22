"""matplotlib.colors' normalisations, as sextant draws them: ``Normalize``, ``LogNorm``,
``TwoSlopeNorm`` and ``BoundaryNorm``, accepted by ``norm=`` of ``imshow``, ``pcolormesh``,
``scatter``, ``quiver``, ``contour`` and the rest. They only describe the mapping (vmin and
vmax, and a ``sextant.ColorNorm``); the colouring happens in the renderer, so they are not
callable on arrays. A matplotlib norm of those four classes is accepted as well, as are the
strings ``"linear"`` and ``"log"``."""

import numpy as np

from . import _sextant

__all__ = ["Normalize", "LogNorm", "TwoSlopeNorm", "BoundaryNorm", "resolve_norm"]


class Normalize:
    """Linear between vmin and vmax (unset: the data's range)."""

    def __init__(self, vmin=None, vmax=None, clip=False):
        self.vmin, self.vmax, self.clip = vmin, vmax, clip

    def _core(self):
        return _sextant.ColorNorm.linear()

    def __repr__(self):
        return f"{type(self).__name__}(vmin={self.vmin!r}, vmax={self.vmax!r})"


class LogNorm(Normalize):
    """Logarithmic; vmin and vmax must be positive (the data's range when unset)."""

    def _core(self):
        return _sextant.ColorNorm.log()


class TwoSlopeNorm(Normalize):
    """vcenter is the colormap's middle; each side is spread over its own half."""

    def __init__(self, vcenter, vmin=None, vmax=None):
        super().__init__(vmin, vmax)
        self.vcenter = vcenter

    def _core(self):
        return _sextant.ColorNorm.two_slope(float(self.vcenter))


class BoundaryNorm(Normalize):
    """Steps at the sorted boundaries. ncolors and extend are accepted for matplotlib's
    signature; sextant spreads the steps over the colormap itself."""

    def __init__(self, boundaries, ncolors=None, clip=False, *, extend="neither"):
        super().__init__(None, None, clip)
        self.boundaries = np.asarray(boundaries, dtype=float)
        if self.boundaries.ndim != 1 or self.boundaries.size < 2:
            raise ValueError("BoundaryNorm: boundaries must be at least two increasing values")
        self.ncolors, self.extend = ncolors, extend

    def _core(self):
        return _sextant.ColorNorm.boundaries(self.boundaries)


def resolve_norm(norm):
    """norm= as (sextant.ColorNorm or None, vmin, vmax); vmin and vmax None where the norm
    sets none. Takes these classes, matplotlib's four, a sextant.ColorNorm and the strings
    "linear" and "log"."""
    if norm is None:
        return None, None, None
    if isinstance(norm, (_sextant.ColorNorm, str)):
        return norm, None, None
    if isinstance(norm, Normalize):
        return norm._core(), norm.vmin, norm.vmax
    name = type(norm).__name__
    if name == "BoundaryNorm":
        return _sextant.ColorNorm.boundaries(np.asarray(norm.boundaries, dtype=float)), None, None
    if name == "TwoSlopeNorm":
        return _sextant.ColorNorm.two_slope(float(norm.vcenter)), norm.vmin, norm.vmax
    if name == "LogNorm":
        return _sextant.ColorNorm.log(), norm.vmin, norm.vmax
    if name == "Normalize":
        return _sextant.ColorNorm.linear(), norm.vmin, norm.vmax
    raise TypeError(f"norm= takes Normalize, LogNorm, TwoSlopeNorm, BoundaryNorm, 'linear' or 'log', "
                    f"not {type(norm).__name__}")
