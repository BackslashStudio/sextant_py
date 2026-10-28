"""matplotlib's rcParams, for the keys sextant honours; the others are kept and warned about once.

Honoured: figure.figsize, figure.dpi, figure.facecolor, axes.facecolor, axes.grid, axes.prop_cycle
(colours only), image.cmap, mathtext.fontset. A key at its default value leaves sextant's own
default alone (sextant's figure background is grey, not matplotlib's white)."""

import contextlib
import warnings

_TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


class PropCycle:
    """A colour cycle standing in for a cycler (the package is not a dependency)."""

    def __init__(self, colors):
        self.colors = list(colors)

    def by_key(self):
        return {"color": list(self.colors)}

    def __iter__(self):
        return iter({"color": c} for c in self.colors)

    def __len__(self):
        return len(self.colors)

    def __eq__(self, other):
        return isinstance(other, PropCycle) and self.colors == other.colors

    __hash__ = None


def _as_cycle(value):
    if isinstance(value, PropCycle):
        return value
    if hasattr(value, "by_key"):
        value = value.by_key()
    if isinstance(value, dict):
        value = value.get("color", value.get("c"))
    if value is None or not len(value):
        raise ValueError("axes.prop_cycle: sextant cycles colours only, give at least one")
    return PropCycle(value)


_DEFAULTS = {
    # honoured
    "figure.figsize": [6.4, 4.8],
    "figure.dpi": 100.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
    "axes.prop_cycle": PropCycle(_TAB10),
    "image.cmap": "viridis",
    "mathtext.fontset": "dejavusans",
    # known to matplotlib, not used here
    "font.size": 10.0, "font.family": ["sans-serif"], "font.sans-serif": ["DejaVu Sans"],
    "lines.linewidth": 1.5, "lines.markersize": 6.0, "lines.linestyle": "-",
    "axes.titlesize": "large", "axes.labelsize": "medium", "axes.linewidth": 0.8,
    "axes.edgecolor": "black", "axes.labelcolor": "black", "axes.axisbelow": "line",
    "axes.spines.top": True, "axes.spines.right": True, "axes.unicode_minus": True,
    "grid.color": "#b0b0b0", "grid.linewidth": 0.8, "grid.linestyle": "-", "grid.alpha": 1.0,
    "legend.fontsize": "medium", "legend.frameon": True, "legend.loc": "best",
    "xtick.labelsize": "medium", "ytick.labelsize": "medium", "xtick.direction": "out", "ytick.direction": "out",
    "savefig.dpi": "figure", "savefig.format": "png", "savefig.bbox": None, "savefig.transparent": False,
    "figure.autolayout": False, "figure.titlesize": "large", "figure.edgecolor": "white",
    "text.color": "black", "text.usetex": False, "image.interpolation": "antialiased",
    "scatter.marker": "o", "patch.linewidth": 1.0, "hist.bins": 10,
}
_HONOURED = {"figure.figsize", "figure.dpi", "figure.facecolor", "axes.facecolor", "axes.grid",
             "axes.prop_cycle", "image.cmap", "mathtext.fontset"}
_ALIASES = {"lines": {"lw": "linewidth", "ls": "linestyle", "ms": "markersize", "c": "color", "mew": "markeredgewidth",
                      "mec": "markeredgecolor", "mfc": "markerfacecolor"},
            "patch": {"lw": "linewidth", "ec": "edgecolor", "fc": "facecolor"}}


def _copy(v):
    return list(v) if isinstance(v, list) else v


class RcParams(dict):
    """plt.rcParams. Setting a key sextant does not use stores it and warns once."""

    def __init__(self):
        super().__init__({k: _copy(v) for k, v in _DEFAULTS.items()})
        self._warned = set()

    def __setitem__(self, key, value):
        if key == "axes.prop_cycle":
            value = _as_cycle(value)
        elif key == "figure.figsize":
            value = [float(v) for v in value]
            if len(value) != 2:
                raise ValueError("figure.figsize: (width, height) in inches")
        elif key not in _HONOURED and key not in self._warned:
            self._warned.add(key)
            why = "is not used by sextant" if key in _DEFAULTS else "is not a matplotlib rc parameter sextant knows"
            warnings.warn(f"rcParams[{key!r}] {why}; the value is kept", UserWarning, stacklevel=3)
        super().__setitem__(key, value)

    def update(self, *args, **kw):
        for k, v in dict(*args, **kw).items():
            self[k] = v

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def changed(self, key):
        """Whether `key` is not at its default (so a script asked for something)."""
        return self[key] != _DEFAULTS[key]

    def reset(self):
        dict.clear(self)
        dict.update(self, {k: _copy(v) for k, v in _DEFAULTS.items()})

    def copy(self):
        out = RcParams()
        dict.update(out, {k: _copy(v) for k, v in self.items()})
        out._warned = self._warned
        return out


rcParams = RcParams()
rcParamsDefault = RcParams()


def rc(group, **kw):
    """rc('lines', linewidth=2) sets 'lines.linewidth'; `group` may be a tuple of groups."""
    groups = (group,) if isinstance(group, str) else tuple(group)
    for g in groups:
        for key, value in kw.items():
            key = _ALIASES.get(g, {}).get(key, key)
            rcParams[f"{g}.{key}"] = value


def rcdefaults():
    rcParams.reset()


class rc_context(contextlib.ContextDecorator):
    """with rc_context({'axes.grid': True}): ... (also a decorator); the rcParams come back on exit,
    also after an exception. `fname` (an rc file) is not read."""

    def __init__(self, rc=None, fname=None):
        if fname is not None:
            raise NotImplementedError("rc_context(fname=): sextant does not read matplotlibrc files")
        self._rc = rc
        self._saved = None

    def __enter__(self):
        self._saved = rcParams.copy()
        if self._rc:
            rcParams.update(self._rc)
        return self

    def __exit__(self, *exc):
        dict.clear(rcParams)
        dict.update(rcParams, self._saved)
        return False
