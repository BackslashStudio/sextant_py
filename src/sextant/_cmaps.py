"""matplotlib's colormap registry over sextant's: ``plt.colormaps``, ``plt.get_cmap``, ``plt.cm``.

A colormap here is its name (a ``str``, so it goes wherever ``cmap=`` takes one); sextant
samples colours itself, so a handle cannot be called on values."""

from . import _sextant


class Colormap(str):
    """A sextant colormap name: ``.name``, ``.reversed()``, and the str it is."""

    @property
    def name(self):
        return str(self)

    def reversed(self, name=None):
        base = str(self)[:-2] if str(self).endswith("_r") else str(self) + "_r"
        return Colormap(name if name is not None else base)

    def __call__(self, *args, **kw):
        raise NotImplementedError("a sextant colormap is a name; it cannot map values to colours from Python")

    def __repr__(self):
        return f"Colormap({str(self)!r})"


def _names():
    return list(_sextant.colormaps())


def resolve(name):
    """The core's own spelling of a colormap name (matching ignores case; '_r' reverses)."""
    if isinstance(name, Colormap):
        return name
    if not isinstance(name, str):
        raise TypeError(f"a colormap name is a str, not {type(name).__name__}")
    base, rev = (name[:-2], True) if name.lower().endswith("_r") else (name, False)
    for known in _names():
        if known.lower() == base.lower():
            return Colormap(known + "_r" if rev else known)
    raise ValueError(f"{name!r} is not a colormap sextant has; sextant.colormaps() lists them: {_names()}")


class ColormapRegistry:
    """plt.colormaps: name -> Colormap, over sextant.colormaps() (and what register_colormap added);
    ``plt.cm`` is the same object, with the names as attributes."""

    def __getitem__(self, name):
        try:
            return resolve(name)
        except TypeError as e:
            raise KeyError(name) from e
        except ValueError as e:
            raise KeyError(str(e)) from None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return resolve(name)
        except ValueError:
            raise AttributeError(f"no colormap {name!r} (sextant.colormaps() lists them)") from None

    def __iter__(self):
        return iter(_names())

    def __len__(self):
        return len(_names())

    def __contains__(self, name):
        try:
            resolve(name)
            return True
        except (TypeError, ValueError):
            return False

    def keys(self):
        return list(self)

    def get_cmap(self, name=None, lut=None):
        return get_cmap(name)

    def register(self, cmap=None, *, name=None, force=False):
        raise NotImplementedError("register: use sextant.register_colormap(name, colors)")


def get_cmap(name=None, lut=None):
    """The colormap `name`; None is the rcParams['image.cmap'] one."""
    if name is None:
        from ._rc import rcParams
        name = rcParams["image.cmap"]
    return resolve(name)


colormaps = ColormapRegistry()
