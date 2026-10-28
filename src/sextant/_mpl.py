"""matplotlib-shaped adapters over sextant's core objects, for sextant.pyplot.

``Figure``, ``Axes`` and ``Axes3D`` here wrap a core ``sextant.Figure`` /
``sextant.Axes`` / ``sextant.Axes3D`` (``.core``) and speak matplotlib: format
strings, ``c=``/``s=``/``label=``, the tab10 colour cycle, sizes in points,
``set_xlabel``/``get_xlim``, ``plot_surface``, ``fig.canvas.mpl_connect``.
What sextant has no equivalent for is not offered (an AttributeError), rather
than accepted and ignored -- except a few calls that only tune a layout
sextant does by itself (``tight_layout``, ``subplots_adjust``).

sextant has no per-object setters, so every adapter records the core calls it
made (``_log``); changing an option of an existing object -- ``set_color``,
``colorbar()``, ``legend(labels)`` -- clears the core axes and replays the
record. Data updates (``set_data``, ``set_ydata``, ...) go straight to the core
``set_*_data`` calls and replay nothing.
"""

import math
import re
import string
import sys
import functools
import time
import warnings
import weakref

import numpy as np

from . import _sextant
from . import _rc
from .colors import resolve_norm

# Sizes in points become pixels at matplotlib's default 100 dpi: a figsize in
# inches times 100 is the figure's size in sextant's logical pixels.
PT = 100.0 / 72.0

TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
         "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
_TAB = dict(zip(["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                 "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"], TAB10))
_TAB["tab:grey"] = _TAB["tab:gray"]
_BASE = {"b": (0.0, 0.0, 1.0), "g": (0.0, 0.5, 0.0), "r": (1.0, 0.0, 0.0), "c": (0.0, 0.75, 0.75),
         "m": (0.75, 0.0, 0.75), "y": (0.75, 0.75, 0.0), "k": (0.0, 0.0, 0.0), "w": (1.0, 1.0, 1.0)}
# Common CSS names, with the values matplotlib gives them ("green" is #008000,
# not sextant's). Names beyond these go to matplotlib when it is installed.
_CSS = {"red": "#ff0000", "green": "#008000", "blue": "#0000ff", "orange": "#ffa500", "purple": "#800080",
        "cyan": "#00ffff", "aqua": "#00ffff", "magenta": "#ff00ff", "fuchsia": "#ff00ff",
        "yellow": "#ffff00", "black": "#000000", "white": "#ffffff", "gray": "#808080", "grey": "#808080",
        "lightgray": "#d3d3d3", "lightgrey": "#d3d3d3", "darkgray": "#a9a9a9", "darkgrey": "#a9a9a9",
        "silver": "#c0c0c0", "brown": "#a52a2a", "pink": "#ffc0cb", "navy": "#000080", "teal": "#008080",
        "olive": "#808000", "lime": "#00ff00", "maroon": "#800000", "gold": "#ffd700",
        "darkblue": "#00008b", "darkred": "#8b0000", "darkgreen": "#006400", "skyblue": "#87ceeb",
        "salmon": "#fa8072", "violet": "#ee82ee", "indigo": "#4b0082", "crimson": "#dc143c",
        "coral": "#ff7f50", "tomato": "#ff6347", "turquoise": "#40e0d0"}

# matplotlib marker -> sextant marker. Stand-ins: "." "," "8" draw circles, the tri
# markers "1"-"4" filled triangles pointing the same way, "H" a hexagon, "d" a diamond.
_MARKERS = {".": "circle", ",": "circle", "o": "circle", "8": "circle",
            "v": "triangle_down", "^": "triangle", "<": "triangle_left", ">": "triangle_right",
            "1": "triangle_down", "2": "triangle", "3": "triangle_left", "4": "triangle_right",
            "s": "square", "p": "pentagon", "h": "hexagon", "H": "hexagon", "*": "star",
            "+": "thin_plus", "x": "thin_cross", "P": "plus", "X": "cross", "|": "vline", "_": "hline",
            "D": "diamond", "d": "diamond"}
_LINESTYLES = {"-": "solid", "--": "dashed", ":": "dotted", "-.": "dashdot", "": "none", " ": "none",
               "None": "none", "none": "none", "solid": "solid", "dashed": "dashed",
               "dotted": "dotted", "dashdot": "dashdot"}
_LOC = {"best": "inside_tr", "upper right": "inside_tr", "upper left": "inside_tl",
        "lower left": "inside_bl", "lower right": "inside_br", "right": "inside_tr",
        "center left": "inside_tl", "center right": "inside_tr", "lower center": "inside_bl",
        "upper center": "inside_tl", "center": "inside_tr"}
_LOC_CODES = ["best", "upper right", "upper left", "lower left", "lower right", "right",
              "center left", "center right", "lower center", "upper center", "center"]

# matplotlib keywords that only tune rendering details sextant has no knob for.
_IGNORED = {"zorder", "antialiased", "aa", "rasterized", "clip_on", "picker", "pickradius", "gid",
            "url", "snap", "animated", "solid_capstyle", "solid_joinstyle", "dash_capstyle",
            "dash_joinstyle", "markeredgecolor", "mec", "markeredgewidth", "mew", "interpolation",
            "filternorm", "resample", "rstride", "cstride", "ccount", "rcount",
            "antialiaseds", "shade_surface"}

MOUSE_BUTTONS = {0: 1, 1: 3, 2: 2}  # sextant left/right/middle -> matplotlib 1/3/2

_adapters = weakref.WeakValueDictionary()  # id(core object) -> adapter


def axes_for(core):
    """The adapter wrapping a core Axes/Axes3D (e.g. an Event's inaxes), or None."""
    return _adapters.get(id(core)) if core is not None else None


# --- colours and formats ----------------------------------------------------------

def to_rgba(c, alpha=None):
    """A matplotlib colour spec as an (r, g, b, a) tuple sextant takes. Unknown
    names go to matplotlib's colour table when matplotlib is installed, else
    through unchanged, for sextant to accept or refuse."""
    if c is None:
        return None
    rgba = None
    if isinstance(c, str):
        s = c.strip()
        low = s.lower()
        if low == "none":
            rgba = (0.0, 0.0, 0.0, 0.0)
        elif re.fullmatch(r"c\d+", low):
            rgba = _hex(TAB10[int(low[1:]) % 10])
        elif low in _TAB:
            rgba = _hex(_TAB[low])
        elif s in _BASE:
            rgba = _BASE[s] + (1.0,)
        elif low in _CSS:
            rgba = _hex(_CSS[low])
        elif re.fullmatch(r"#[0-9a-fA-F]{3,4}", s):
            rgba = _hex("#" + "".join(ch * 2 for ch in s[1:]))
        elif re.fullmatch(r"#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?", s):
            rgba = _hex(s)
        elif re.fullmatch(r"\d*\.?\d+", s) and 0.0 <= float(s) <= 1.0:
            rgba = (float(s),) * 3 + (1.0,)  # grayscale, as matplotlib reads "0.5"
        else:
            try:
                from matplotlib.colors import to_rgba as mpl_to_rgba
                rgba = tuple(float(v) for v in mpl_to_rgba(s))
            except (ImportError, ValueError):
                if alpha is not None:
                    raise ValueError(f"unknown colour {c!r}") from None
                return s
    else:
        v = [float(x) for x in c]
        if len(v) not in (3, 4):
            raise ValueError(f"a colour tuple has 3 or 4 values, got {c!r}")
        rgba = tuple(v) + ((1.0,) if len(v) == 3 else ())
    if alpha is not None:
        rgba = rgba[:3] + (float(alpha),)
    return rgba


def _hex(h):
    h = h.lstrip("#")
    vals = [int(h[i:i + 2], 16) / 255.0 for i in range(0, len(h), 2)]
    return tuple(vals) + ((1.0,) if len(vals) == 3 else ())


def parse_fmt(fmt):
    """matplotlib's '[marker][line][color]' format string: (linestyle, marker, colour)."""
    ls = marker = color = None
    i = 0
    while i < len(fmt):
        two = fmt[i:i + 2]
        if two in ("--", "-."):
            ls, i = two, i + 2
            continue
        ch = fmt[i]
        if ch in "-:":
            ls = ch
        elif ch == "C" and i + 1 < len(fmt) and fmt[i + 1].isdigit():
            j = i + 1
            while j < len(fmt) and fmt[j].isdigit():
                j += 1
            color, i = fmt[i:j], j
            continue
        elif ch in _BASE:
            color = ch
        elif ch in _MARKERS:
            marker = ch
        else:
            raise ValueError(f"unrecognized character {ch!r} in format string {fmt!r}")
        i += 1
    return ls, marker, color


def _pop(kw, *names, default=None):
    """The first of `names` present in kw (all removed), else default."""
    value, found = default, False
    for n in names:
        if n in kw:
            v = kw.pop(n)
            if not found:
                value, found = v, True
    return value


def _drop_ignored(kw):
    for k in list(kw):
        if k in _IGNORED:
            del kw[k]


def _takes_visible(method):
    """matplotlib's visible= on a plotting call: the call, then set_visible(False) on
    what it returns (an artist, or a list or tuple holding them)."""
    @functools.wraps(method)
    def call(self, *args, **kw):
        visible = kw.pop("visible", True)
        out = method(self, *args, **kw)
        if not visible:
            _hide(out)
        return out
    return call


# The 2D core kinds that take a zorder.
_ZORDERED = frozenset({"line", "scatter", "scatter_z", "bar", "heatmap", "heatmap_tri", "contour", "fill", "quiver",
                       "axline", "axspan", "shape_poly", "shape_circle", "shape_wedge", "pie"})
# A pick event's PickKind -> the adapter's kind (the core method's name) where they differ.
_PICK_KINDS = {"poly": "shape_poly", "circle": "shape_circle", "wedge": "shape_wedge", "heatmap_tri": "heatmap_tri",
               "contour": "contour", "quiver3d": "quiver3d"}


def _zorder(default):
    """matplotlib's zorder= on a 2D plotting call, with that artist's default (images 0,
    patches and collections 1, lines 2): popped from the call's keywords and given to
    every core object the call makes (a plot() line and its markers), through
    _AxesBase._add. None means the default."""
    def decorate(method):
        @functools.wraps(method)
        def call(self, *args, **kw):
            z = kw.pop("zorder", None)
            outer = self._z
            # A decorated call made by another one (pcolormesh -> imshow) keeps the outer's.
            self._z = outer if z is None and outer is not None else _finite_zorder(
                default if z is None else z)
            try:
                return method(self, *args, **kw)
            finally:
                self._z = outer
        return call
    return decorate


def _finite_zorder(z):
    z = float(z)
    if not math.isfinite(z):
        raise ValueError(f"zorder must be finite, not {z}")
    return z


def _hide(out):
    if isinstance(out, _Artist):
        out.set_visible(False)
    elif isinstance(out, (list, tuple)):
        for a in out:
            _hide(a)


def _marker(m):
    if m is None:
        return None
    if m in ("", " ", "None", "none"):
        return "none"
    return _MARKERS.get(m, m)  # sextant names ("circle", ...) pass through


def _linestyle(ls):
    return None if ls is None else _LINESTYLES.get(ls, ls)


def _drawstyle(ds):
    # sextant takes matplotlib's names ("steps", "steps-mid", ...) as they are.
    return None if ds is None else str(ds)


def _vec(a):
    return np.asarray(a, dtype=float).ravel()


def _auto_limits(data, vmin, vmax):
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite)]
    lo = float(finite.min()) if vmin is None and finite.size else vmin
    hi = float(finite.max()) if vmax is None and finite.size else vmax
    lo = 0.0 if lo is None else lo
    hi = 1.0 if hi is None else hi
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return lo, hi


def _is_log(core):
    if isinstance(core, str):
        core = core.lower()
    return core is not None and (core == "log" or getattr(core, "kind", None) == "log")


def _rc_cmap(cmap):
    """The colormap a colormapped call uses: its own, else rcParams['image.cmap'] when a script changed it."""
    if cmap is None and _rc.rcParams.changed("image.cmap"):
        from ._cmaps import resolve
        return resolve(_rc.rcParams["image.cmap"])
    return cmap


def _norm_opts(values, vmin, vmax, norm):
    """vmin, vmax and norm options for a colormapped call: matplotlib's norm= (a norm
    object or "log") with its own vmin/vmax unless the call gives them; a log norm reads
    its range from the positive values."""
    core, nvmin, nvmax = resolve_norm(norm)
    vmin = nvmin if vmin is None else vmin
    vmax = nvmax if vmax is None else vmax
    if _is_log(core):
        a = np.asarray(values, dtype=float)
        a = a[np.isfinite(a) & (a > 0)]
        if a.size:
            vmin = float(a.min()) if vmin is None else vmin
            vmax = float(a.max()) if vmax is None else vmax
    lo, hi = _auto_limits(values, vmin, vmax)
    out = {"vmin": lo, "vmax": hi}
    if core is not None:
        out["norm"] = core
    return out


def _errorbar(err, n, name):
    """matplotlib's xerr/yerr: scalar, (N,) symmetric or (2, N) lower/upper -> (lo, hi)."""
    if err is None:
        return None, None
    e = np.asarray(err, dtype=float)
    if e.ndim == 0:
        return np.full(n, float(e)), None
    if e.ndim == 1:
        return e, None
    if e.shape[0] == 2:
        return e[0], e[1]
    raise ValueError(f"{name} must be a scalar, shape (N,) or shape (2, N)")


# --- artists ------------------------------------------------------------------------

class _Artist:
    """A handle on one core plot object: its kind, index and log entry."""

    def __init__(self, axes, kind, index, entry, label=None):
        self.axes = axes
        self._kind = kind
        self._index = index
        self._entry = entry
        self._label = label
        axes._artists[(kind, index)] = self

    @property
    def figure(self):
        return self.axes.figure

    def get_label(self):
        return self._label or ""

    def set_label(self, label):
        self._label = label
        self._set_options(name=label or "")

    def remove(self):
        """Take the artist out of its axes, as matplotlib's Artist.remove(): it is no longer
        drawn, saved or in the legend, and the axes' other artists keep working. A second
        call raises ValueError. The axes replays its recorded calls without it, so every
        other artist's sextant index (not its id) moves; a removed artist is not to be set
        or read again."""
        self.axes._remove_artist(self)

    def _check(self):
        if self._index is None:
            raise ValueError("this artist was removed from its axes")

    def _relocate(self, where):
        """After the axes replayed its log: this artist's objects' new (kind, index)."""
        self._kind, self._index = where[id(self._entry)]

    def _set_options(self, **kw):
        self._check()
        self._entry[2].update(kw)
        self.axes._rebuild()

    def _set_args(self, method, *args):
        self._check()
        getattr(self.axes.core, method)(self._index, *args)
        self._entry[1][:len(args)] = args
        self.axes.figure._changed()

    def _parts(self):
        """(kind, index, entry) of each core object this artist draws."""
        return [(self._kind, self._index, self._entry)]

    def get_zorder(self):
        return float(self._entry[2].get("zorder", 0.0))

    def set_zorder(self, z):
        """Paint order: objects draw in ascending zorder, equal ones in the order they were
        made. Sets every core object the artist draws (a plot() line and its markers) and
        replays the axes' calls."""
        z = _finite_zorder(z)
        self._check()
        for _, _, entry in self._parts():
            entry[2]["zorder"] = z
        self.axes._rebuild()

    def get_visible(self):
        return self._entry[2].get("visible", True)

    def set_visible(self, b):
        """Hide or show it: a hidden artist is not drawn or saved, has no legend entry
        or colorbar, is not picked, and leaves the automatic limits (matplotlib's still
        count)."""
        b = bool(b)
        self._check()
        for kind, index, entry in self._parts():
            entry[2]["visible"] = b
            getattr(self.axes.core, f"set_{kind}_visible")(index, b)
        self.axes.figure._changed()


class Line2D(_Artist):
    """ax.plot()'s return: a line, markers (a scatter over it), or both."""

    def __init__(self, axes, kind, index, entry, label, marker=None):
        super().__init__(axes, kind, index, entry, label)
        self._markers = marker  # (index, entry) of the scatter drawing the markers

    def _parts(self):
        parts = super()._parts()
        if self._markers is not None:
            parts.append(("scatter", *self._markers))
        return parts

    def _relocate(self, where):
        super()._relocate(where)
        if self._markers is not None:
            self._markers = (where[id(self._markers[1])][1], self._markers[1])

    def get_data(self):
        core = self.axes.core
        d = core.line_data(self._index) if self._kind == "line" else core.scatter_data(self._index)
        return d.x, d.y

    def get_xdata(self):
        return self.get_data()[0]

    def get_ydata(self):
        return self.get_data()[1]

    def set_data(self, *args):
        x, y = args if len(args) == 2 else args[0]
        x, y = _vec(x), _vec(y)
        if self._kind == "line":
            self._set_args("set_line_data", x, y)
        else:
            self._set_args("set_scatter_data", x, y)
        if self._markers is not None:
            i, entry = self._markers
            self.axes.core.set_scatter_data(i, x, y)
            entry[1][:2] = [x, y]

    def set_xdata(self, x):
        self.set_data(x, self.get_ydata())

    def set_ydata(self, y):
        self.set_data(self.get_xdata(), y)

    def get_color(self):
        return self._entry[2].get("color")

    def set_color(self, c):
        rgba = to_rgba(c)
        self._entry[2]["color"] = rgba
        if self._markers is not None:
            self._markers[1][2]["color"] = rgba
        self.axes._rebuild()

    def set_linewidth(self, lw):
        self._set_options(linewidth=lw * PT)

    def set_linestyle(self, ls):
        self._set_options(linestyle=_linestyle(ls))

    def set_alpha(self, a):
        self._entry[2]["alpha"] = a
        if self._markers is not None:
            self._markers[1][2]["alpha"] = a
        self.axes._rebuild()

    def get_drawstyle(self):
        return str(self._entry[2].get("drawstyle", "default")) if self._kind == "line" else "default"

    def set_drawstyle(self, ds):
        """'default', 'steps' / 'steps-pre', 'steps-mid' or 'steps-post'."""
        if self._kind == "line":
            self._set_options(drawstyle=_drawstyle(ds))

    set_ds = set_drawstyle


def _keep_point_sizes(entry, n):
    """A new point count drops per-point sizes, as sextant's (x, y) setters do."""
    sizes = entry[2].get("sizes")
    if sizes is not None and np.size(sizes) != n:
        del entry[2]["sizes"]


class PathCollection(_Artist):
    """ax.scatter()'s return."""

    def get_offsets(self):
        core = self.axes.core
        d = core.scatter_data(self._index) if self._kind == "scatter" else core.scatter_z_data(self._index)
        return np.column_stack([d.x, d.y])

    def set_offsets(self, xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        _keep_point_sizes(self._entry, len(xy))
        if self._kind == "scatter":
            self._set_args("set_scatter_data", xy[:, 0], xy[:, 1])
        else:
            z = self.axes.core.scatter_z_data(self._index).z
            self._set_args("set_scatter_z_data", xy[:, 0], xy[:, 1], z)

    def get_array(self):
        return self.axes.core.scatter_z_data(self._index).z if self._kind == "scatter_z" else None

    def get_sizes(self):
        """Marker areas in points^2, matplotlib's unit: one per point, or one value."""
        sizes = self._entry[2].get("sizes")
        if sizes is None:
            return np.array([(self._entry[2]["size"] / PT) ** 2])
        return (np.asarray(sizes, dtype=float) / PT) ** 2

    def set_sizes(self, sizes):
        n = len(self.get_offsets())
        self._entry[2].pop("sizes", None)
        self._set_options(**Axes._marker_size(sizes, n))

    def set_array(self, values):
        d = self.axes.core.scatter_z_data(self._index)
        self._set_args("set_scatter_z_data", d.x, d.y, _vec(values))

    def set_clim(self, vmin=None, vmax=None):
        if isinstance(vmin, (tuple, list)):
            vmin, vmax = vmin
        self._set_options(vmin=vmin, vmax=vmax)

    def set_cmap(self, cmap):
        self._set_options(cmap=cmap)


class AxesImage(_Artist):
    """ax.imshow()/pcolormesh()'s return."""

    def get_array(self):
        return self.axes.core.heatmap_data(self._index).data

    def set_data(self, a):
        self._check()
        if self._entry[0] == "hist2d":
            raise NotImplementedError("set_data(): a hist2d image is counted from its points; call hist2d again")
        d = self.axes.core.heatmap_data(self._index)
        arr = np.asarray(a)
        # imshow's conversion: 0..255 bytes are colours only in an RGB(A) image
        d.data = arr / 255.0 if arr.dtype == np.uint8 and arr.ndim == 3 else np.asarray(arr, dtype=float)
        self.axes.core.set_heatmap_data(self._index, d)
        self._entry[1][0] = arr
        self.axes.figure._changed()

    def set_clim(self, vmin=None, vmax=None):
        if isinstance(vmin, (tuple, list)):
            vmin, vmax = vmin
        self._set_options(vmin=vmin, vmax=vmax)

    def set_cmap(self, cmap):
        self._set_options(cmap=cmap)

    def get_extent(self):
        d = self.axes.core.heatmap_data(self._index)
        return (*d.xrange, *d.yrange)


class ContourSet(_Artist):
    """ax.contour()/contourf()'s return."""

    @property
    def levels(self):
        return self.axes.core.contour_data(self._index).levels

    @property
    def filled(self):
        return bool(self.axes.core.contour_data(self._index).filled)

    def get_array(self):
        return self.levels

    def set_clim(self, vmin=None, vmax=None):
        if isinstance(vmin, (tuple, list)):
            vmin, vmax = vmin
        self._set_options(vmin=vmin, vmax=vmax)

    def set_cmap(self, cmap):
        self._set_options(cmap=cmap)


class TriMesh(AxesImage):
    """ax.tripcolor()'s return."""

    def get_array(self):
        return self.axes.core.heatmap_tri_data(self._index).values

    def set_array(self, a):
        d = self.axes.core.heatmap_tri_data(self._index)
        self._set_args("set_heatmap_tri_data", d.x, d.y, np.asarray(a, dtype=float))

    set_data = set_array

    def get_extent(self):
        raise AttributeError("a tripcolor has no extent")


class Quiver3D(_Artist):
    """Axes3D.quiver()'s return: the arrows' anchors and vectors."""

    def _data(self):
        return self.axes.core.quiver3d_data(self._index)

    def get_segments(self):
        d = self._data()
        amp = self._entry[2].get("amp", 1.0)
        a = np.column_stack([d.x, d.y, d.z])
        b = a + amp * np.column_stack([d.u, d.v, d.w])
        return np.stack([a, b], axis=1)

    def set_UVC(self, U, V, W):
        d = self._data()
        self._set_args("set_quiver3d_data", d.x, d.y, d.z, _vec(U), _vec(V), _vec(W))


class BarContainer(_Artist):
    """ax.bar()/barh()/hist()'s bars."""

    @property
    def datavalues(self):
        return self.axes.core.bar_data(self._index).height

    def set_heights(self, heights):
        """Not matplotlib (there, each bar is a Rectangle): new heights, same x."""
        x = self.axes.core.bar_data(self._index).x
        self._set_args("set_bar_data", x, _vec(heights))


class RefLine(_Artist):
    """axhline()/axvline()'s return (a Line2D in matplotlib). Along the line its data
    are the ends as fractions of the frame (xmin/xmax or ymin/ymax); across it, the
    value twice."""

    def _horizontal(self):
        return self._entry[2]["orientation"] == "horizontal"

    def _ends(self):
        return [self._entry[2]["from_"], self._entry[2]["to"]]

    def _value(self):
        return self.axes.core.axline_data(self._index).value

    def get_xdata(self):
        return self._ends() if self._horizontal() else [self._value()] * 2

    def get_ydata(self):
        return [self._value()] * 2 if self._horizontal() else self._ends()

    def _set(self, across, data):
        d = np.ravel(np.asarray(data, dtype=float))
        if across:
            self._set_args("set_axline_data", float(d[0]))
        else:
            self._set_options(from_=float(d[0]), to=float(d[-1]))

    def set_xdata(self, x):
        self._set(not self._horizontal(), x)

    def set_ydata(self, y):
        self._set(self._horizontal(), y)

    def get_color(self):
        return self._entry[2].get("color")

    def set_color(self, c):
        self._set_options(color=to_rgba(c))

    def set_linewidth(self, lw):
        self._set_options(linewidth=lw * PT)

    def set_linestyle(self, ls):
        self._set_options(linestyle=_linestyle(ls))

    def set_alpha(self, a):
        self._set_options(alpha=1.0 if a is None else float(a))


class AxLine(RefLine):
    """axline()'s return for a point and a slope or two points (matplotlib's AxLine): get/set_xy1,
    get/set_xy2 (two points) and get/set_slope (a slope), in data coordinates, over RefLine's style
    setters. It has no finite data."""

    def _no_data(self, *args):
        raise NotImplementedError("an axline has no finite data: use get_xy1(), get_xy2() or get_slope()")

    get_xdata = get_ydata = set_xdata = set_ydata = _no_data

    def _data(self):
        return self.axes.core.axline_data(self._index)

    def get_xy1(self):
        d = self._data()
        return d.x, d.y

    def get_xy2(self):
        d = self._data()
        return (d.x2, d.y2) if d.mode == "two_points" else None

    def get_slope(self):
        d = self._data()
        if d.mode == "point_slope":
            return d.slope
        return (d.y2 - d.y) / (d.x2 - d.x) if d.x2 != d.x else math.inf

    def _edit(self, mode, put):
        """Change the recorded call's numbers and replay; `mode` is the only form `put` fits."""
        self._check()
        if self._data().mode != mode:
            other = "slope" if mode == "two_points" else "xy2"
            raise ValueError(f"Cannot set this value while '{other}' is set; they differ but their "
                             "functionalities overlap")
        put(self._entry[1], self._entry[2])
        self.axes._rebuild()

    def set_xy1(self, x, y):
        self._check()
        self._entry[1][:2] = [float(x), float(y)]
        self.axes._rebuild()

    def set_xy2(self, x, y):
        self._edit("two_points", lambda args, kw: args.__setitem__(slice(2, 4), [float(x), float(y)]))

    def set_slope(self, slope):
        self._edit("point_slope", lambda args, kw: kw.update(slope=float(slope)))


class _Painted:
    """The colour setters of the filled artists whose options are a face, an outline and an alpha
    (Span, the patches)."""

    def get_facecolor(self):
        return self._entry[2].get("color")

    def set_facecolor(self, c):
        self._set_options(color=to_rgba(c))

    set_color = set_facecolor

    def set_edgecolor(self, c):
        opts = {"edgecolor": to_rgba(c)}
        if not self._entry[2].get("linewidth"):
            opts["linewidth"] = PT
        self._set_options(**opts)

    def set_linewidth(self, lw):
        self._set_options(linewidth=float(lw) * PT)

    def set_linestyle(self, ls):
        self._set_options(linestyle=_linestyle(ls))

    def set_alpha(self, a):
        self._set_options(alpha=1.0 if a is None else float(a))


class Span(_Painted, _Artist):
    """axhspan()/axvspan()'s return (a Rectangle in matplotlib): its colors, alpha and
    outline; get_span()/set_span() (not matplotlib) read and move its two ends."""

    def get_span(self):
        d = self.axes.core.axspan_data(self._index)
        return d.lo, d.hi

    def set_span(self, lo, hi):
        self._set_args("set_axspan_data", float(lo), float(hi))


class FillBetweenPolyCollection(_Artist):
    """fill_between()/fill_betweenx()'s return, and each stackplot() layer's: its
    curves (set_data, as matplotlib 3.10's), colours, outline and alpha."""

    def __init__(self, axes, kind, index, entry, label, linewidth=PT):
        super().__init__(axes, kind, index, entry, label)
        self._linewidth = linewidth  # the outline's, in px, whenever there is one

    def get_data(self):
        """Not matplotlib: (t, f1, f2), the run coordinate and the two curves."""
        d = self.axes.core.fill_data(self._index)
        return d.x, d.y1, d.y2

    def set_data(self, t, f1, f2, *, where=None):
        t = _vec(t)
        f1 = np.broadcast_to(np.asarray(f1, dtype=float), t.shape).astype(float)
        f2 = np.broadcast_to(np.asarray(f2, dtype=float), t.shape).astype(float)
        exclude = _exclude(where, len(t), "set_data")
        self.axes.core.set_fill_data(self._index, _sextant.FillData(t, f1, f2, exclude))
        self._entry[1][:3] = [t, f1, f2]
        self._entry[2].pop("exclude", None)
        if exclude is not None:
            self._entry[2]["exclude"] = exclude
        self.axes.figure._changed()

    def get_facecolor(self):
        return self._entry[2].get("color")

    def set_facecolor(self, c):
        self._set_options(color=to_rgba(c))

    def set_color(self, c):
        """Face and outline, as matplotlib's."""
        rgba = to_rgba(c)
        self._entry[2]["color"] = rgba
        self.set_edgecolor(c)

    def get_edgecolor(self):
        return self._entry[2].get("edgecolor") if self._entry[2].get("linewidth") else None

    def set_edgecolor(self, c):
        opts = self._entry[2]
        if c is None or (isinstance(c, str) and c.lower() == "none"):
            opts.pop("edgecolor", None)
            opts.pop("linewidth", None)
        else:
            opts["edgecolor"] = to_rgba(c)
            opts["linewidth"] = self._linewidth
        self.axes._rebuild()

    def set_linewidth(self, lw):
        self._linewidth = float(lw) * PT
        if "edgecolor" in self._entry[2]:
            self._set_options(linewidth=self._linewidth)

    def set_linestyle(self, ls):
        self._set_options(linestyle=_linestyle(ls))

    def set_alpha(self, a):
        self._set_options(alpha=1.0 if a is None else float(a))  # the outline fades too


def _patch_opts(kw):
    """A patch's keywords as the core shape call's options (the colour unset when none is given:
    add_patch() takes the axes' C0, as matplotlib's patch.facecolor)."""
    transform = kw.pop("transform", None)
    if transform is not None and getattr(transform, "coords", None) != "data":
        raise NotImplementedError("patches are in data coordinates (transform= is not mapped)")
    if kw.pop("fill", True) is False:
        raise NotImplementedError("fill=False: sextant shapes always fill")
    if kw.pop("hatch", None) is not None:
        raise ValueError("sextant draws no hatching")
    zorder = kw.pop("zorder", None)
    _drop_ignored(kw)
    color = _pop(kw, "color", "c")
    face = _pop(kw, "facecolor", "fc", default=color)
    edge = _pop(kw, "edgecolor", "ec")
    lw = _pop(kw, "linewidth", "lw")
    ls = _linestyle(_pop(kw, "linestyle", "ls"))
    alpha = kw.pop("alpha", None)
    label = kw.pop("label", None)
    opts = dict(kw)
    if face is not None:
        opts["color"] = to_rgba(face)
    if alpha is not None:
        opts["alpha"] = float(alpha)
    # An outline only when one is asked for: matplotlib's default edge is invisible.
    if edge is not None and not (isinstance(edge, str) and edge.lower() in ("none", "face")):
        opts.update(edgecolor=to_rgba(edge), linewidth=(1.0 if lw is None else float(lw)) * PT)
    elif color is not None and lw is not None:  # color= is the edge too
        opts["linewidth"] = float(lw) * PT
    if ls is not None:
        opts["linestyle"] = ls
    if zorder is not None:
        opts["zorder"] = _finite_zorder(zorder)
    if label is not None:
        opts["name"] = str(label)
    return opts, label


class _Patch(_Painted, _Artist):
    """Circle, Wedge and Polygon: built with their numbers and style, drawn once ax.add_patch()
    has them. Before that the setters just edit what will be drawn; after, they replay the
    axes' calls like any artist's."""

    def __init__(self, kind, args, kw):
        opts, label = _patch_opts(kw)
        self.axes = self._index = None
        self._kind, self._label = kind, label
        self._entry = [kind, args, opts]

    def _add_to(self, axes):
        method, args, opts = self._entry
        opts.setdefault("color", axes._c0())
        index, self._entry = axes._add(method, method, *args, **opts)
        self.axes, self._index = axes, index
        axes._artists[(method, index)] = self
        return self

    def _set_options(self, **kw):
        if self.axes is None:
            self._entry[2].update(kw)
        else:
            super()._set_options(**kw)

    def get_zorder(self):
        return float(self._entry[2].get("zorder", 1.0))

    def set_zorder(self, z):
        if self.axes is None:
            self._entry[2]["zorder"] = _finite_zorder(z)
        else:
            super().set_zorder(z)

    def set_visible(self, b):
        if self.axes is None:
            self._entry[2]["visible"] = bool(b)
        else:
            super().set_visible(b)


class Circle(_Patch):
    """matplotlib.patches.Circle((x, y), radius): the radius in data units, so an ellipse on
    unequal axes, as in matplotlib. Drawn by ax.add_patch()."""

    def __init__(self, xy, radius, **kw):
        x, y = (float(v) for v in xy)
        super().__init__("shape_circle", [x, y, float(radius)], kw)


class Wedge(_Patch):
    """matplotlib.patches.Wedge(center, r, theta1, theta2, width=None): a sector from theta1 to
    theta2 degrees counter-clockwise from +x; `width` makes it a ring segment (hole radius r - width)."""

    def __init__(self, center, r, theta1, theta2, width=None, **kw):
        x, y = (float(v) for v in center)
        super().__init__("shape_wedge", [x, y, float(r), float(theta1), float(theta2)], kw)
        if width is not None:
            self._entry[2]["inner"] = float(r) - float(width)


class Polygon(_Patch):
    """matplotlib.patches.Polygon(xy, closed=True): xy is (N, 2). Always closed and filled."""

    def __init__(self, xy, *, closed=True, **kw):
        if not closed:
            raise NotImplementedError("Polygon(closed=False): sextant polygons are always closed")
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"Polygon: xy must have shape (N, 2), not {xy.shape}")
        super().__init__("shape_poly", [xy[:, 0].copy(), xy[:, 1].copy()], kw)


class Quiver(_Artist):
    """quiver()'s return: the arrows' anchors (X, Y, set_offsets) and vectors (U, V,
    set_UVC), and for a colormapped one its values (get_array, set_clim, set_cmap)."""

    def _data(self):
        return self.axes.core.quiver_data(self._index)

    @property
    def X(self):
        return self._data().x

    @property
    def Y(self):
        return self._data().y

    @property
    def U(self):
        return self._data().u

    @property
    def V(self):
        return self._data().v

    @property
    def N(self):
        return len(self._data().x)

    def get_offsets(self):
        d = self._data()
        return np.column_stack([d.x, d.y])

    def set_offsets(self, xy):
        """New anchors: the same number keeps the vectors and values; another drops the
        values, as the core's setter does."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        d = self._data()
        if len(xy) != len(d.x):
            raise ValueError(f"set_offsets(): {len(d.x)} arrows, got {len(xy)} offsets")
        self._set_args("set_quiver_data", xy[:, 0].copy(), xy[:, 1].copy(), d.u, d.v)

    def set_UVC(self, U, V, C=None):
        """New vectors (and values), the anchors and the amp kept."""
        d = self._data()
        u = np.broadcast_to(np.asarray(U, dtype=float).ravel(), d.x.shape).astype(float)
        v = np.broadcast_to(np.asarray(V, dtype=float).ravel(), d.x.shape).astype(float)
        c = d.c if C is None else _vec(C)
        self.axes.core.set_quiver_data(self._index, _sextant.QuiverData(d.x, d.y, u, v, c))
        args = self._entry[1]
        args[2:4] = [u, v]
        if C is not None:
            args[4:] = [c]
        self.axes.figure._changed()

    def get_array(self):
        return self._data().c

    def set_array(self, values):
        d = self._data()
        self.set_UVC(d.u, d.v, values)

    def set_clim(self, vmin=None, vmax=None):
        if isinstance(vmin, (tuple, list)):
            vmin, vmax = vmin
        self._set_options(vmin=vmin, vmax=vmax)

    def set_cmap(self, cmap):
        self._set_options(cmap=cmap)

    def get_facecolor(self):
        return self._entry[2].get("color")

    def set_color(self, c):
        self._set_options(color=to_rgba(c))

    set_facecolor = set_color

    def set_alpha(self, a):
        self._set_options(alpha=1.0 if a is None else float(a))


class Surface3D(_Artist):
    """plot_surface()/plot_trisurf()/plot_wireframe()'s return."""

    def set_cmap(self, cmap):
        self._set_options(cmap=cmap)


class Path3D(_Artist):
    """Axes3D.plot()/scatter()'s return."""

    def get_data_3d(self):
        core = self.axes.core
        d = core.line3d_data(self._index) if self._kind == "line3d" else core.scatter3d_data(self._index)
        return d.x, d.y, d.z

    def set_data_3d(self, x, y, z):
        _keep_point_sizes(self._entry, len(_vec(x)))
        method = "set_line3d_data" if self._kind == "line3d" else "set_scatter3d_data"
        self._set_args(method, _vec(x), _vec(y), _vec(z))


class Text(_Artist):
    """ax.text()/annotate()'s return (Axes3D.text()/text2D() too). The position is
    the anchor in the coordinates it was given in; an annotation's arrow and point
    stay where they are."""

    def __init__(self, axes, index, entry, slots):
        super().__init__(axes, "text", index, entry)
        self._slots = slots  # where the string and the position sit in the call's args

    def _args(self):
        return self._entry[1]

    # matplotlib's text zorder; sextant draws texts above all data, so it only reads back.
    _zorder_value = 3.0

    def get_zorder(self):
        return self._zorder_value

    def set_zorder(self, z):
        self._zorder_value = _finite_zorder(z)

    def get_text(self):
        return self._args()[self._slots[0]]

    def get_position(self):
        return tuple(self._args()[i] for i in self._slots[1:])

    def _set(self, text, pos):
        args = self._args()
        args[self._slots[0]] = text
        for i, v in zip(self._slots[1:], pos):
            args[i] = v
        self.axes.core.set_text_data(self._index, text, *pos)
        self.axes.figure._changed()

    def set_text(self, s):
        self._set(str(s), self.get_position())

    def set_position(self, xy):
        self._set(self.get_text(), tuple(float(v) for v in xy))

    def set_color(self, c):
        self._set_options(color=to_rgba(c))

    def set_fontsize(self, size):
        self._set_options(fontsize=_fontsize(size))

    def set_rotation(self, r):
        self._set_options(rotation=_rotation(r))

    def set_alpha(self, a):
        self._set_options(alpha=1.0 if a is None else float(a))


class _Pie(_Artist):
    """One core pie: the artist that log replays and picks track. pie() hands out views
    on it, a PieWedge per slice and a PieText per label or percent."""

    def __init__(self, axes, index, entry):
        super().__init__(axes, "pie", index, entry)
        self.wedges = []

    def _fractions(self):
        v = np.asarray(self._entry[1][0], dtype=float)
        return v / v.sum() if self._entry[2].get("normalize", True) else v


class _PiePart:
    """What a PieWedge and a PieText share: they belong to one sextant pie, so zorder,
    visibility and remove() act on the whole pie."""

    def __init__(self, pie, k):
        self._pie, self._k = pie, k

    @property
    def axes(self):
        return self._pie.axes

    @property
    def figure(self):
        return self._pie.figure

    def _opts(self):
        return self._pie._entry[2]

    def get_zorder(self):
        return self._pie.get_zorder()

    def set_zorder(self, z):
        self._pie.set_zorder(z)

    def get_visible(self):
        return self._pie.get_visible()

    def set_visible(self, b):
        self._pie.set_visible(b)

    def remove(self):
        self._pie.remove()


class PieWedge(_PiePart):
    """A slice of a pie() (matplotlib's Wedge): its face colour and label are its own;
    the outline and alpha are the whole pie's."""

    def _angles(self):
        o = self._opts()
        sign = 1.0 if o.get("counterclock", True) else -1.0
        edges = o.get("startangle", 0.0) + sign * 360.0 * np.concatenate([[0.0], np.cumsum(self._pie._fractions())])
        a, b = float(edges[self._k]), float(edges[self._k + 1])
        return min(a, b), max(a, b)

    @property
    def theta1(self):
        return self._angles()[0]

    @property
    def theta2(self):
        return self._angles()[1]

    @property
    def r(self):
        return self._opts().get("radius", 1.0)

    @property
    def center(self):
        """The slice's centre: the pie's, moved out along its middle by explode."""
        o = self._opts()
        explode = o.get("explode")
        out = float(explode[self._k]) * self.r if explode is not None and len(explode) else 0.0
        mid = math.radians(sum(self._angles()) / 2)
        return o.get("cx", 0.0) + out * math.cos(mid), o.get("cy", 0.0) + out * math.sin(mid)

    def get_label(self):
        labels = self._opts().get("labels")
        return labels[self._k] if labels else ""

    def set_label(self, label):
        n = len(self._pie.wedges)
        labels = list(self._opts().get("labels") or [""] * n)
        labels[self._k] = "" if label is None else str(label)
        self._pie._set_options(labels=labels)

    def get_facecolor(self):
        return self._opts()["colors"][self._k]

    def set_facecolor(self, c):
        colors = list(self._opts()["colors"])
        colors[self._k] = to_rgba(c)
        self._pie._set_options(colors=colors)

    set_color = set_facecolor

    def set_edgecolor(self, c):
        opts = {"edgecolor": to_rgba(c)}
        if not self._opts().get("linewidth"):
            opts["linewidth"] = PT
        self._pie._set_options(**opts)

    def set_linewidth(self, lw):
        self._pie._set_options(linewidth=float(lw) * PT)

    def set_alpha(self, a):
        self._pie._set_options(alpha=1.0 if a is None else float(a))


class PieText(_PiePart):
    """A pie()'s label or percent (matplotlib's Text): the colour and size are every
    text's of the pie."""

    def __init__(self, pie, k, percent):
        super().__init__(pie, k)
        self._percent = percent

    def get_text(self):
        if self._percent:
            return self._opts()["autopct"] % (100.0 * self._pie._fractions()[self._k])
        labels = self._opts().get("labels")
        return labels[self._k] if labels else ""

    def set_color(self, c):
        self._pie._set_options(textcolor=to_rgba(c))

    def set_fontsize(self, size):
        self._pie._set_options(fontsize=_fontsize(size))


class _Transform:
    """ax.transData / ax.transAxes: only which coordinates a text is in."""

    def __init__(self, coords):
        self.coords = coords


# matplotlib's named font sizes, relative to its default 10 pt.
_FONT_SCALE = {"xx-small": 0.579, "x-small": 0.694, "small": 0.833, "medium": 1.0, "large": 1.2,
               "x-large": 1.44, "xx-large": 1.728, "larger": 1.2, "smaller": 0.833}
_TEXT_VA = {"top": "top", "center": "center", "center_baseline": "center", "baseline": "baseline",
            "bottom": "bottom"}
_TEXT_COORDS = {"data": "data", "axes fraction": "fraction"}
# arrowstyle -> (head, tail): the head is at the point, the tail at the text.
_ARROWSTYLES = {"-": ("none", "none"), "->": ("open", "none"), "-|>": ("filled", "none"),
                "<-": ("none", "open"), "<|-": ("none", "filled"), "<->": ("open", "open"),
                "<|-|>": ("filled", "filled"), "|-|": ("bar", "bar"), "-[": ("bar", "none"),
                "]-": ("none", "bar"), "]-[": ("bar", "bar"), "fancy": ("filled", "none"),
                "simple": ("filled", "none"), "wedge": ("filled", "none")}
_ARROW_IGNORED = {"mutation_scale", "mutation_aspect", "patchA", "patchB", "relpos", "shrink",
                  "joinstyle", "capstyle", "antialiased", "zorder"}


def _fontsize(size):
    if isinstance(size, str):
        if size not in _FONT_SCALE:
            raise ValueError(f"unknown font size {size!r}; use a number of points or one of {sorted(_FONT_SCALE)}")
        return 10.0 * _FONT_SCALE[size] * PT
    return float(size) * PT


def _rotation(r):
    return float({"horizontal": 0.0, "vertical": 90.0}.get(r, r))


def _style_params(style):
    """'round,pad=0.5' -> ('round', {'pad': 0.5})."""
    name, *params = [part.strip() for part in style.split(",")]
    return name, {k.strip(): float(v) for k, v in (p.split("=") for p in params)}


def _math_kw(kw, where):
    """matplotlib's usetex and parse_math: usetex=True raises (sextant draws its own mathtext
    subset); returns parse_math, True unless False was given."""
    if kw.pop("usetex", None):
        raise ValueError(f"{where}: usetex=True is not supported; sextant draws $...$ math itself")
    parse_math = kw.pop("parse_math", None)
    return True if parse_math is None else bool(parse_math)


def _text_opts(kw, fontdict=None):
    """matplotlib Text keywords -> TextOptions fields. sextant's own names pass through."""
    if fontdict:
        kw = {**fontdict, **kw}
    parse_math = _math_kw(kw, "text")
    clip = kw.pop("clip_on", None)
    _drop_ignored(kw)
    o = {}
    size = _pop(kw, "fontsize", "size")
    if size is not None:
        o["fontsize"] = _fontsize(size)
    font_px = o.get("fontsize", 10.0 * PT)
    color = _pop(kw, "color", "c")
    if color is not None:
        o["color"] = to_rgba(color)
    alpha = _pop(kw, "alpha")
    if alpha is not None:
        o["alpha"] = float(alpha)
    ha = _pop(kw, "ha", "horizontalalignment")
    if ha is not None:
        o["ha"] = ha
    va = _pop(kw, "va", "verticalalignment")
    if va is not None:
        o["va"] = _TEXT_VA.get(va, va)
    rotation = _pop(kw, "rotation")
    if rotation is not None:
        o["rotation"] = _rotation(rotation)
    linespacing = _pop(kw, "linespacing")
    if linespacing is not None:
        o["linespacing"] = float(linespacing)
    if clip:
        o["clip_to_frame"] = True
    if not parse_math:
        o["parse_math"] = False
    bbox = _pop(kw, "bbox")
    if bbox is not None:
        # matplotlib's FancyBboxPatch defaults: face C0, black edge 1 pt, pad 0.3 x the font size.
        b = dict(bbox)
        _, params = _style_params(b.pop("boxstyle", "square"))
        pad = b.pop("pad", params.get("pad", 0.3))
        balpha = b.pop("alpha", None)
        face = to_rgba(_pop(b, "facecolor", "fc", default=TAB10[0]), balpha)
        edge = to_rgba(_pop(b, "edgecolor", "ec", default="black"), balpha)
        lw = _pop(b, "linewidth", "lw", default=1.0)
        b.pop("fill", None)
        if b:
            raise TypeError(f"bbox: unsupported key(s) {sorted(b)}")
        o.update(background=face, edgecolor=edge, edge_linewidth=float(lw) * PT, pad=float(pad) * font_px)
    o.update(kw)
    return o


def _no_parse_math(kw, where):
    """Titles and labels follow the figure's switch only: parse_math=False is per text."""
    if not _math_kw(kw, where):
        raise ValueError(f"{where}: parse_math=False works on text() and annotate() only; "
                         "turn math off for a whole figure with figure(mathtext=False)")


def _arrow_opts(arrowprops):
    """matplotlib's arrowprops -> ArrowOptions fields."""
    ap = dict(arrowprops)
    a = {}
    style = ap.pop("arrowstyle", None)
    if style is None:
        # No arrowstyle: matplotlib's YAArrow, a filled arrow drawn in points.
        a["head"] = "filled"
        a["linewidth"] = float(ap.pop("width", 4.0)) * PT
        a["head_width"] = float(ap.pop("headwidth", 12.0)) * PT
        a["head_length"] = float(ap.pop("headlength", 12.0)) * PT
    else:
        name, params = _style_params(style.replace(" ", ""))
        if name not in _ARROWSTYLES:
            raise ValueError(f"arrowstyle {name!r} is not one sextant draws; use one of {sorted(_ARROWSTYLES)}")
        a["head"], a["tail"] = _ARROWSTYLES[name]
        # Head sizes are in units of the mutation scale, the text's size (10 pt by default).
        if "head_length" in params:
            a["head_length"] = params["head_length"] * 10.0 * PT
        if "head_width" in params:
            a["head_width"] = 2.0 * params["head_width"] * 10.0 * PT
    color = _pop(ap, "color", "facecolor", "fc", "edgecolor", "ec")
    if color is not None:
        a["color"] = to_rgba(color)
    lw = _pop(ap, "linewidth", "lw")
    if lw is not None:
        a["linewidth"] = float(lw) * PT
    ls = _pop(ap, "linestyle", "ls")
    if ls is not None:
        a["linestyle"] = _linestyle(ls)
    a["gap_text"] = float(ap.pop("shrinkA", 2.0)) * PT
    a["gap_point"] = float(ap.pop("shrinkB", 2.0)) * PT
    conn = ap.pop("connectionstyle", None)
    if conn is not None:
        name, params = _style_params(conn.replace(" ", ""))
        if name != "arc3":
            raise ValueError(f"connectionstyle {name!r}: sextant draws arc3 only (straight, or arc3,rad=...)")
        a["arc"] = params.get("rad", 0.0)
    for k in _ARROW_IGNORED:
        ap.pop(k, None)
    a.update(ap)
    return a


class Colorbar:
    """fig.colorbar()'s return. sextant draws a colorbar as part of the object that
    asked for it; this only renames or moves it."""

    def __init__(self, mappable, label, orientation):
        self.mappable = mappable
        self.ax = mappable.axes
        self._orientation = orientation

    def set_label(self, label):
        self.mappable._set_options(name=label or "")


# --- axes -----------------------------------------------------------------------------

# set_axes_style() on a sextant twin keeps its own axis' fields and gives the shared
# axis' and the cell's to the host; these are the host's (`{a}` = the shared axis).
_CELL_STYLE = frozenset({"background", "spine_color", "spine_linewidth", "spine_bottom", "spine_left",
                         "spine_top", "spine_right", "frame_margin", "title_color", "title_fontsize"})
_AXIS_STYLE = ("show_{a}ticks", "show_{a}tick_labels", "minor_{a}ticks", "{a}tick_color", "{a}tick_length",
               "{a}tick_linewidth", "minor_{a}tick_length", "{a}label_color", "{a}label_fontsize",
               "{a}title_color", "{a}title_fontsize", "{a}axis_color")
# Each title call's size field (the call sets it, its default size included).
_TITLE_SIZE = {"set_title": "title_fontsize", "set_xtitle": "xtitle_fontsize",
               "set_ytitle": "ytitle_fontsize", "set_ztitle": "ztitle_fontsize"}


def _host_style(field, shared):
    """Whether a twin sharing `shared` hands `field` to its host."""
    return field in _CELL_STYLE or field in {f.format(a=shared) for f in _AXIS_STYLE}


class _StaticText:
    """A string standing in for matplotlib's Text where sextant keeps only the string
    (tick labels, legend entries)."""

    def __init__(self, text):
        self._text = str(text)

    def get_text(self):
        return self._text

    def get_visible(self):
        return True

    def __str__(self):
        return self._text

    def __repr__(self):
        return f"Text({self._text!r})"


class _Legend:
    """ax.get_legend()'s return once legend() has been called: its entries and frame."""

    def __init__(self, axes):
        self.axes = axes
        self._frame = True
        for e in axes._log:
            if e[0] == "legend":
                self._frame = e[2].get("frameon", True)

    def get_texts(self):
        return [_StaticText(s) for s in self.axes.get_legend_handles_labels()[1]]

    def get_frame_on(self):
        return bool(self._frame)

    def get_visible(self):
        return True


class _AxesBase:
    _GRID_ON = False  # a new axes' grid
    _LINE = "line"  # the core's line call

    def __init__(self, figure, core):
        self.figure = figure
        self.core = core
        self._log = []  # [core method, args, kwargs]: everything this axes was told
        self._artists = {}  # (kind, index) -> artist, for pick events
        self._line_cycle = 0
        self._patch_cycle = 0
        self._cycle = None  # set_prop_cycle()'s colours; None is tab10
        self._grid_on = self._GRID_ON
        self._grid_minor = {}  # grid(which='minor'): minor / minor_color, kept across grid() calls
        self._xticks = self._yticks = self._zticks = None
        self._xticklabels = self._yticklabels = None
        self._style = {}  # AxesStyle fields set through this adapter (set_axes_style takes the whole struct)
        self.transData = _Transform("data")
        self.transAxes = _Transform("fraction")
        _adapters[id(core)] = self

    def _coords(self, transform, method):
        if transform is None:
            return None
        if isinstance(transform, _Transform):
            return transform.coords
        raise ValueError(f"{method}(): sextant places text in ax.transData or ax.transAxes only")

    # --- the record ---

    def _call(self, method, *args, **kw):
        getattr(self.core, method)(*args, **kw)
        entry = [method, list(args), kw]
        self._log.append(entry)
        self.figure._changed()
        return entry

    def _owner(self, axis=None):
        """The adapter that holds `axis`'s calls ("x"/"y"; None = the cell's title,
        frame and background): a twin's host for its shared axis and the cell, as
        sextant forwards them, else this one."""
        host = getattr(self, "_host", None)
        if host is None or (axis is not None and axis != self._shared):
            return self
        return host

    def _set_style(self, **fields):
        """Fields of the axes style, merged into what this adapter already set (sextant's
        set_axes_style() replaces the whole struct). On a twin, the host's fields go
        through the host's record; the twin's call then rewrites the host's fields, so
        the host's are sent again after it."""
        host = getattr(self, "_host", None)
        if host is not None:
            to_host = {k: v for k, v in fields.items() if _host_style(k, self._shared)}
            fields = {k: v for k, v in fields.items() if k not in to_host}
            if to_host:
                host._set_style(**to_host)
            if not fields:
                return
        self._style.update(fields)
        self._call("set_axes_style", **self._style)
        if host is not None:
            host._restyle()

    def _restyle(self):
        """Send this adapter's style again, unrecorded (a twin's style call reset it)."""
        self.core.set_axes_style(**self._style)
        self.figure._changed()

    _z = None  # the running 2D plotting call's zorder (_zorder)

    def _add(self, kind, method, *args, **kw):
        """A plotting call; returns (index, entry)."""
        if kind in _ZORDERED and self._z is not None:
            kw.setdefault("zorder", self._z)
        index = getattr(self.core, f"{kind}_count")()
        return index, self._call(method, *args, **kw)

    def _rebuild(self, locate=False):
        """Clear the core and replay the log. With `locate`, also return {id(entry): (kind,
        index)} for each entry that made an object (the replay's indices)."""
        self.core.cla()
        where = {}
        if locate:
            kinds = [n[:-6] for n in dir(self.core) if n.endswith("_count") and n != "plane_count"]
            seen = dict.fromkeys(kinds, 0)
        for entry in self._log:
            method, args, kw = entry
            getattr(self.core, method)(*args, **kw)
            if locate:
                for k in kinds:
                    n = getattr(self.core, f"{k}_count")()
                    if n != seen[k]:
                        where[id(entry)] = (k, seen[k])
                        seen[k] = n
        host = getattr(self, "_host", None)
        if host is not None:  # a twin's replayed style calls reset the host's fields
            host._restyle()
        self.figure._changed()
        return where

    def _remove_artist(self, artist):
        parts = artist._parts()
        gone = {id(e) for _, _, e in parts}
        if artist._index is None or not all(any(e is x for x in self._log) for _, _, e in parts):
            raise ValueError("remove(): the artist is not in its axes (removed already, or cleared)")
        live = {id(a): a for a in self._artists.values() if a is not artist}
        self._log[:] = [e for e in self._log if id(e) not in gone]
        where = self._rebuild(locate=True)
        artist._index = None
        self._artists.clear()
        for a in live.values():
            a._relocate(where)
            for kind, index, _ in a._parts():
                self._artists[(kind, index)] = a

    def _next_color(self, patches=False):
        # matplotlib's rule, not sextant's single per-axes cycle: lines and
        # patches count separately, so every call passes its colour explicitly.
        if patches:
            i, self._patch_cycle = self._patch_cycle, self._patch_cycle + 1
        else:
            i, self._line_cycle = self._line_cycle, self._line_cycle + 1
        if self._cycle is not None:
            return self._cycle[i % len(self._cycle)]
        return _hex(TAB10[i % 10])

    def set_prop_cycle(self, *args, **kw):
        """set_prop_cycle(color=[...]), set_prop_cycle(cycler(color=[...])) or
        set_prop_cycle(None) (back to tab10). Colours only: sextant cycles nothing else.
        Restarts the cycle, as matplotlib does."""
        if len(args) == 1 and args[0] is None and not kw:
            colors = None
        else:
            if len(args) == 1 and hasattr(args[0], "by_key"):  # a cycler
                kw = {**args[0].by_key(), **kw}
            elif len(args) == 2:
                kw = {args[0]: args[1], **kw}
            elif args:
                raise TypeError("set_prop_cycle(): a cycler, None, or color=[...]")
            colors = _pop(kw, "color", "c")
            if kw:
                raise TypeError(f"set_prop_cycle(): sextant cycles colors only, not {sorted(kw)}")
            if colors is None or not len(colors):
                raise ValueError("set_prop_cycle(): color= needs at least one colour")
            colors = [to_rgba(c) for c in colors]
        self._cycle = colors
        self._line_cycle = self._patch_cycle = 0

    def _apply_rc(self):
        """What a new (or cleared) axes takes from rcParams, for the keys a script changed."""
        rc = _rc.rcParams
        if rc.changed("axes.prop_cycle"):
            self._cycle = [to_rgba(c) for c in rc["axes.prop_cycle"].colors]
        if rc.changed("axes.facecolor"):
            self._set_style(background=to_rgba(rc["axes.facecolor"]))
        if rc.changed("axes.grid") and rc["axes.grid"] and type(self) is Axes:
            self.grid(True)

    def cla(self):
        self.core.cla()
        self._log.clear()
        self._artists.clear()
        self._line_cycle = self._patch_cycle = 0
        self._cycle = None
        self._grid_on = self._GRID_ON
        self._xticks = self._yticks = self._zticks = None
        self._xticklabels = self._yticklabels = None
        self._grid_minor = {}
        self._style.clear()
        self._apply_rc()
        self.figure._changed()

    clear = cla

    # --- decoration ---

    def _title(self, method, label, fontsize, kw):
        if isinstance(fontsize, dict):  # matplotlib's positional fontdict
            kw = {**fontsize, **kw}
            fontsize = None
        _drop_ignored(kw)
        _no_parse_math(kw, method)
        for k in ("fontdict", "loc", "pad", "labelpad", "fontweight", "weight", "y"):
            kw.pop(k, None)
        size = _pop(kw, "fontsize", "size", default=fontsize)
        color = _pop(kw, "color", "c")
        if kw:
            raise TypeError(f"{method}(): unsupported keyword(s) {sorted(kw)}")
        args = (label,) if size is None else (label, float(size) * PT)
        # A twin's title and its shared axis' title are the host's (sextant forwards them).
        owner = self._owner({"set_title": None, "set_xtitle": "x", "set_ytitle": "y"}.get(method, "z"))
        # The size lives in the axes style, so keep it in the record that set_axes_style()
        # resends whole (else a later tick_params() reset it); the call sets it either way.
        if size is None:
            owner._style.pop(_TITLE_SIZE[method], None)
        else:
            owner._style[_TITLE_SIZE[method]] = float(size) * PT
        owner._call(method, *args)
        if color is not None:  # the colour is a style field, which the title call leaves alone
            self._set_style(**{_TITLE_SIZE[method].replace("fontsize", "color"): to_rgba(color)})

    def set_title(self, label, fontsize=None, **kw):
        self._title("set_title", label, fontsize, kw)

    def get_title(self):
        return self.core.title()

    def set_xlabel(self, xlabel, fontsize=None, **kw):
        self._title("set_xtitle", xlabel, fontsize, kw)

    def get_xlabel(self):
        return self.core.xtitle()

    def set_ylabel(self, ylabel, fontsize=None, **kw):
        self._title("set_ytitle", ylabel, fontsize, kw)

    def get_ylabel(self):
        return self.core.ytitle()

    def _lim(self, axis, left, right, kw):
        left = _pop(kw, "left", "bottom", "zmin", "xmin", "ymin", default=left)
        right = _pop(kw, "right", "top", "zmax", "xmax", "ymax", default=right)
        kw.pop("emit", None), kw.pop("auto", None)
        if kw:
            raise TypeError(f"set_{axis}lim(): unsupported keyword(s) {sorted(kw)}")
        if left is not None and right is None and np.ndim(left) == 1:
            left, right = left
        lo, hi = getattr(self.core, f"{axis}lim")()
        lo = lo if left is None else float(left)
        hi = hi if right is None else float(right)
        self._call(f"set_{axis}lim", lo, hi)
        return lo, hi

    def set_xlim(self, left=None, right=None, **kw):
        return self._lim("x", left, right, kw)

    def get_xlim(self):
        return self.core.xlim()

    def set_ylim(self, bottom=None, top=None, **kw):
        return self._lim("y", bottom, top, kw)

    def get_ylim(self):
        return self.core.ylim()

    def invert_yaxis(self):
        lo, hi = self.core.ylim()
        self._call("set_ylim", hi, lo)

    def invert_xaxis(self):
        lo, hi = self.core.xlim()
        self._call("set_xlim", hi, lo)

    def get_xbound(self):
        lo, hi = self.core.xlim()
        return (lo, hi) if lo <= hi else (hi, lo)

    def get_ybound(self):
        lo, hi = self.core.ylim()
        return (lo, hi) if lo <= hi else (hi, lo)

    def set_xbound(self, lower=None, upper=None):
        """Like set_xlim(), but the axis keeps its direction (an inverted axis stays inverted)."""
        self._bound("x", lower, upper)

    def set_ybound(self, lower=None, upper=None):
        self._bound("y", lower, upper)

    def _bound(self, axis, lower, upper):
        if upper is None and lower is not None and np.ndim(lower) == 1:
            lower, upper = lower
        cur = self.get_xbound() if axis == "x" else self.get_ybound()
        inverted = self.xaxis_inverted() if axis == "x" else self.yaxis_inverted()
        lo = cur[0] if lower is None else float(lower)
        hi = cur[1] if upper is None else float(upper)
        lo, hi = min(lo, hi), max(lo, hi)
        self._lim(axis, *((hi, lo) if inverted else (lo, hi)), {})

    def xaxis_inverted(self):
        lo, hi = self.core.xlim()
        return lo > hi

    def yaxis_inverted(self):
        lo, hi = self.core.ylim()
        return lo > hi

    def _tick_values(self, axis):
        owner = self._owner(axis)
        ticks = getattr(owner, f"_{axis}ticks")
        if ticks is None:
            raise RuntimeError(f"get_{axis}ticks(): the ticks are automatic -- sextant chooses them when "
                               f"it draws, so there are none to read; set_{axis}ticks() first")
        return owner, ticks

    def get_xticks(self, minor=False):
        return self._tick_values("x")[1].copy() if not minor else np.empty(0)

    def get_yticks(self, minor=False):
        return self._tick_values("y")[1].copy() if not minor else np.empty(0)

    def _tick_labels(self, axis, minor):
        owner, ticks = self._tick_values(axis)
        if minor:
            return []
        given = getattr(owner, f"_{axis}ticklabels", None)
        texts = given if given else [f"{t:g}" for t in ticks]
        return [_StaticText(s) for s in texts]

    def get_xticklabels(self, minor=False, which=None):
        return self._tick_labels("x", minor)

    def get_yticklabels(self, minor=False, which=None):
        return self._tick_labels("y", minor)

    # --- what is on the axes ---

    # Artist kind -> the list matplotlib files it under. A bar's container stands for its rectangles.
    _FILES = {
        "line": "lines", "axline": "lines", "path3d": "lines",
        "text": "texts",
        "bar": "patches", "axspan": "patches", "pie": "patches", "shape_circle": "patches",
        "shape_wedge": "patches", "shape_poly": "patches",
        "heatmap": "images", "heatmap_tri": "images",
    }

    def _filed(self, name):
        return tuple(a for (kind, _), a in self._artists.items() if self._FILES.get(kind, "collections") == name)

    lines = property(lambda self: self._filed("lines"))
    texts = property(lambda self: self._filed("texts"))
    patches = property(lambda self: self._filed("patches"))
    collections = property(lambda self: self._filed("collections"))
    images = property(lambda self: self._filed("images"))

    def get_lines(self):
        return list(self.lines)

    def get_children(self):
        out = list(self._artists.values())
        out += [a for a in (getattr(self, "xaxis", None), getattr(self, "yaxis", None)) if a is not None]
        return out

    def get_legend_handles_labels(self):
        handles = [a for a in self._artists.values() if a.get_label() and not a.get_label().startswith("_")]
        return handles, [a.get_label() for a in handles]

    def get_legend(self):
        return _Legend(self) if any(e[0] == "legend" for e in self._log) else None

    def get_facecolor(self):
        return self._style.get("background", (1.0, 1.0, 1.0, 1.0))

    def get_visible(self):
        return True

    def _ticks(self, axis, ticks, labels, kw):
        owner = self._owner(axis)
        if owner is not self:  # a twin's shared axis is its host's
            return owner._ticks(axis, ticks, labels, kw)
        kw.pop("minor", None)
        if kw:
            raise TypeError(f"set_{axis}ticks(): unsupported keyword(s) {sorted(kw)}")
        ticks = _vec(ticks)
        setattr(self, f"_{axis}ticks", ticks)
        setattr(self, f"_{axis}ticklabels", [str(s) for s in labels] if labels is not None else None)
        self._call(f"set_{axis}ticks", ticks, [str(s) for s in labels] if labels is not None else [])
        # An empty list means "no ticks" here as in matplotlib (sextant's own set_xticks([]) means
        # "automatic", which is why this goes through the style's show_{axis}ticks).
        show = bool(len(ticks))
        if self._style.get(f"show_{axis}ticks", True) != show:
            self._set_style(**{f"show_{axis}ticks": show})

    def set_xticks(self, ticks, labels=None, **kw):
        self._ticks("x", ticks, labels, kw)

    def set_yticks(self, ticks, labels=None, **kw):
        self._ticks("y", ticks, labels, kw)

    def set_facecolor(self, color):
        self._set_style(background=to_rgba(color))

    set_axis_bgcolor = set_facecolor

    def set_xticklabels(self, labels, **kw):
        if self._xticks is None:
            raise ValueError("set_xticklabels(): set_xticks() first -- sextant labels explicit ticks")
        self._ticks("x", self._xticks, labels, {})

    def set_yticklabels(self, labels, **kw):
        if self._yticks is None:
            raise ValueError("set_yticklabels(): set_yticks() first -- sextant labels explicit ticks")
        self._ticks("y", self._yticks, labels, {})

    def grid(self, visible=None, which="major", axis="both", **kw):
        _drop_ignored(kw)
        color = _pop(kw, "color", "c")
        alpha = kw.pop("alpha", None)
        opts = {}
        if color is not None or alpha is not None:
            opts["color"] = to_rgba(color if color is not None else (0.8, 0.8, 0.8), alpha)
        ls = _linestyle(_pop(kw, "linestyle", "ls"))
        if ls is not None:
            opts["linestyle"] = ls
        lw = _pop(kw, "linewidth", "lw")
        if lw is not None:
            opts["linewidth"] = float(lw) * PT
        if kw:
            raise TypeError(f"grid(): unsupported keyword(s) {sorted(kw)}")
        if which not in ("major", "minor", "both"):
            raise ValueError(f"grid(): which must be 'major', 'minor' or 'both', not {which!r}")
        if which == "minor":
            # sextant's minor lines take the major lines' style and width, and come
            # with the major grid: which='minor' sets their visibility and colour.
            if visible is None:
                visible = True if opts else not self._grid_minor.get("minor", False)
            self._grid_minor["minor"] = bool(visible)
            if "color" in opts:
                self._grid_minor["minor_color"] = opts.pop("color")
            on = self._grid_on or self._grid_minor["minor"]
        else:
            if visible is None:
                visible = True if opts else not self._grid_on
            self._grid_on = bool(visible)
            if which == "both":
                self._grid_minor["minor"] = self._grid_on
                if "color" in opts:
                    self._grid_minor["minor_color"] = opts["color"]
            on = self._grid_on
        self._call("grid", on, **opts, **self._grid_minor)

    def legend(self, *args, loc=None, fontsize=None, frameon=None, **kw):
        for k in ("bbox_to_anchor", "ncol", "ncols", "title", "shadow", "fancybox", "framealpha",
                  "borderaxespad", "handlelength", "markerscale", "numpoints", "scatterpoints"):
            kw.pop(k, None)
        if kw:
            raise TypeError(f"legend(): unsupported keyword(s) {sorted(kw)}")
        if args:
            labels = args[-1]
            if len(args) == 2:
                raise TypeError("legend(handles, labels): pass label= when plotting instead")
            self._relabel(labels)
        if loc is None:
            loc = "best"
        if isinstance(loc, int):
            loc = _LOC_CODES[loc]
        opts = {"anchor": _LOC.get(loc, loc)}
        if fontsize is not None:
            opts["fontsize"] = float(fontsize) * PT
        if frameon is not None:
            opts["frameon"] = bool(frameon)
        self._call("legend", **opts)

    def _relabel(self, labels):
        """legend(['a', 'b']): name the plotted series in order."""
        series = [e for e in self._log if e[0] in self._SERIES and e[2].get("show_legend", True)]
        for entry, label in zip(series, labels):
            entry[2]["name"] = str(label)
        self._rebuild()

    def set(self, **kw):
        """ax.set(title=..., xlabel=..., xlim=..., ...)."""
        for key, value in kw.items():
            setter = getattr(self, f"set_{key}", None)
            if setter is None:
                raise AttributeError(f"set(): no set_{key}")
            setter(*value) if key.endswith("lim") and np.ndim(value) == 1 else setter(value)


class Axes(_AxesBase):
    """A 2D axes, matplotlib style, over a core sextant.Axes (``.core``)."""

    _SERIES = {"line", "scatter", "scatter_z", "bar", "fill", "quiver", "heatmap_tri", "contour"}

    def __init__(self, figure, core):
        super().__init__(figure, core)
        self._span = None  # ((rows, cols), first cell, last cell) on the figure's grid
        self._twins = {}  # "x"/"y" -> the twinx()/twiny() adapter, in the order made
        self._host = None  # on a twin: the axes it twins
        self._shared = None  # on a twin: the axis it shares with its host, "x" or "y"
        self._apply_rc()

    # --- plot --------------------------------------------------------------------

    @_zorder(2)
    @_takes_visible
    def line(self, *args, **kw):
        """line([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs): matplotlib's plot(),
        named after sextant's line(); plot is an alias. Returns a list of Line2D."""
        _drop_ignored(kw)
        scalex = kw.pop("scalex", True)
        scaley = kw.pop("scaley", True)
        del scalex, scaley
        lines = []
        for x, y, fmt in _plot_groups(args, kw.pop("data", None)):
            y = np.asarray(y, dtype=float)
            cols = y.reshape(len(y), -1) if y.ndim > 1 else y[:, None]
            xs = None if x is None else np.asarray(x, dtype=float)
            for j in range(cols.shape[1]):
                xj = None if xs is None else (xs[:, j] if xs.ndim > 1 else xs)
                lines.append(self._line(xj, cols[:, j], fmt, dict(kw)))
        return lines

    plot = line

    def step(self, x, y, *args, where="pre", data=None, **kw):
        """matplotlib's step(): plot() with drawstyle 'steps-<where>' ('pre', 'mid' or
        'post'). Returns a list of Line2D."""
        if where not in ("pre", "mid", "post"):
            raise ValueError(f"step(): where must be 'pre', 'mid' or 'post', not {where!r}")
        kw["drawstyle"] = "steps-" + where
        return self.line(x, y, *args, data=data, **kw)

    def _line(self, x, y, fmt, kw):
        ls, mk, fc = parse_fmt(fmt)
        color = to_rgba(_pop(kw, "color", "c", default=fc))
        alpha = kw.pop("alpha", None)
        label = kw.pop("label", None)
        lw = _pop(kw, "linewidth", "lw")
        ls = _pop(kw, "linestyle", "ls", default=ls)
        mk = kw.pop("marker", mk)
        ms = _pop(kw, "markersize", "ms")
        mfc = to_rgba(_pop(kw, "markerfacecolor", "mfc"))
        ds = _drawstyle(_pop(kw, "drawstyle", "ds"))
        if ds is not None:
            kw["drawstyle"] = ds
        if ls is None:
            ls = "-" if mk is None else "none"
        ls = _linestyle(ls)
        if color is None:
            color = self._next_color()
        if x is None:
            x = np.arange(len(y), dtype=float)
        x, y = _vec(x), _vec(y)
        common = {"alpha": alpha} if alpha is not None else {}
        handle = None
        if ls != "none":
            opts = dict(kw, color=color, linestyle=ls, linewidth=(lw if lw is not None else 1.5) * PT, **common)
            if label is not None:
                opts["name"] = str(label)
            i, entry = self._add("line", self._LINE, x, y, **opts)
            handle = Line2D(self, "line", i, entry, label)
        marker = _marker(mk)
        if marker not in (None, "none"):
            size = (ms if ms is not None else (2.0 if mk in ".," else 6.0)) * PT
            mopts = {"color": mfc or color, "marker": marker, "size": size, "alpha": 1.0 if alpha is None else alpha}
            if handle is not None:
                mopts["show_legend"] = False
            elif label is not None:
                mopts["name"] = str(label)
            j, mentry = self._add("scatter", "scatter", x, y, **mopts)
            if handle is None:
                handle = Line2D(self, "scatter", j, mentry, label)
            else:
                handle._markers = (j, mentry)
                self._artists[("scatter", j)] = handle
        if handle is None:
            raise ValueError("plot(): linestyle and marker are both 'none': nothing to draw")
        return handle

    @_zorder(1)
    @_takes_visible
    def scatter(self, x, y, s=None, c=None, marker=None, cmap=None, vmin=None, vmax=None,
                alpha=None, label=None, colorbar=False, **kw):
        _drop_ignored(kw)
        kw.pop("plotnonfinite", None)
        norm = kw.pop("norm", None)
        edgecolors = _pop(kw, "edgecolors", "edgecolor", "ec")
        linewidths = _pop(kw, "linewidths", "linewidth", "lw")
        facecolors = _pop(kw, "facecolors", "facecolor", "fc")
        color = _pop(kw, "color", default=None)
        x, y = _vec(x), _vec(y)
        opts = dict(kw, **self._marker_size(s, len(x)), alpha=1.0 if alpha is None else alpha)
        # An outline only when asked for (matplotlib's default 'face' outline is invisible here).
        if isinstance(edgecolors, str) and edgecolors.lower() == "none":
            edgecolors, linewidths = None, None
        elif edgecolors is not None or linewidths is not None:
            if edgecolors is not None and not (isinstance(edgecolors, str) and edgecolors == "face"):
                opts["edgecolor"] = to_rgba(edgecolors)
            opts["edge_linewidth"] = (1.5 if linewidths is None else float(np.asarray(linewidths).flat[0])) * PT
        if isinstance(facecolors, str) and facecolors.lower() == "none":
            opts["filled"] = False  # hollow: only the outline
            facecolors = None
        if marker is not None:
            opts["marker"] = _marker(marker)
        if label is not None:
            opts["name"] = str(label)
        values = _values(c, len(x))
        if values is not None:
            opts.update(_norm_opts(values, vmin, vmax, norm), colorbar=bool(colorbar))
            cmap = _rc_cmap(cmap)
            if cmap is not None:
                opts["cmap"] = cmap
            i, entry = self._add("scatter_z", "scatter_z", x, y, values, **opts)
            handle = PathCollection(self, "scatter_z", i, entry, label)
            self.figure._mappable = handle
            return handle
        c = c if c is not None else (facecolors if facecolors is not None else color)
        opts["color"] = to_rgba(c) if c is not None else self._next_color(patches=True)
        i, entry = self._add("scatter", "scatter", x, y, **opts)
        return PathCollection(self, "scatter", i, entry, label)

    @staticmethod
    def _marker_size(s, n):
        """scatter()'s s, an area in points^2 (one, or one per point), as sextant's
        diameters in px: {"size": d}, or {"sizes": [d, ...]} per point (an area of 0
        draws nothing, as in matplotlib)."""
        a = np.ravel(np.asarray(36.0 if s is None else s, dtype=float))  # 36: lines.markersize ** 2
        if a.size == 1:
            return {"size": math.sqrt(max(float(a[0]), 0.0)) * PT}
        if a.size != n:
            raise ValueError(f"scatter(): s must be a scalar or one value per point ({n}), not {a.size}")
        return {"sizes": np.sqrt(np.clip(a, 0.0, None)) * PT}

    @_zorder(1)
    @_takes_visible
    def bar(self, x, height, width=0.8, bottom=None, *, align="center", **kw):
        return self._bar_series("bar", x, height, width, bottom, align, "vertical", **kw)

    @_zorder(1)
    @_takes_visible
    def barh(self, y, width, height=0.8, left=None, *, align="center", **kw):
        """matplotlib's barh(): bars along x from `left`, `height` thick along y; xerr is
        the error on the lengths."""
        return self._bar_series("barh", y, width, height, left, align, "horizontal", **kw)

    def _bar_series(self, method, x, height, width, bottom, align, orientation, color=None, edgecolor=None,
                    linewidth=None, label=None, alpha=None, yerr=None, xerr=None, ecolor=None, capsize=None,
                    error_kw=None, tick_label=None, data=None, **kw):
        """bar(), barh() and hist(): `x` the positions, `height` the lengths, `width` the
        thickness and `bottom` the bases, whichever way the bars lie."""
        _drop_ignored(kw)
        del data
        x, h = _vec(x), _vec(height)
        if np.ndim(width) > 0:
            w = np.asarray(width, dtype=float)
            if not np.all(w == w.flat[0]):
                raise ValueError(f"{method}(): sextant draws one {'width' if method == 'bar' else 'height'} "
                                 "per series")
            width = float(w.flat[0])
        spacing = float(np.min(np.diff(np.unique(x)))) if len(np.unique(x)) > 1 else 1.0
        if align == "edge":
            x = x + width / 2
        opts = dict(kw, width=float(width) / spacing,
                    color=to_rgba(color) if color is not None else self._next_color(patches=True))
        if orientation != "vertical":
            opts["orientation"] = orientation
        if bottom is not None:
            opts["bottom"] = np.ravel(np.asarray(bottom, dtype=float))
        opts["edgecolor"] = to_rgba(edgecolor) if edgecolor is not None else opts["color"]
        opts["linewidth"] = float(linewidth) * PT if linewidth is not None else (0.0 if edgecolor is None else PT)
        if alpha is not None:
            opts["alpha"] = alpha
        if label is not None:
            opts["name"] = str(label)
        err = self._err(len(x), xerr, yerr, ecolor, capsize, error_kw, opts)
        if err is not None:  # (a polar bar takes none)
            opts["err"] = err
        i, entry = self._add("bar", "bar", x, h, **opts)
        if tick_label is not None:
            labels = [tick_label] * len(x) if isinstance(tick_label, str) else list(tick_label)
            (self.set_xticks if orientation == "vertical" else self.set_yticks)(x, labels)
        return BarContainer(self, "bar", i, entry, label)

    @_zorder(1)
    @_takes_visible
    def hist(self, x, bins=10, range=None, density=False, weights=None, cumulative=False, bottom=None,
             histtype="bar", orientation="vertical", color=None, label=None, alpha=None, rwidth=None,
             edgecolor=None, linewidth=None, **kw):
        _drop_ignored(kw)
        if histtype not in ("bar", "stepfilled"):
            raise ValueError(f"hist(): histtype={histtype!r} is not supported ('bar')")
        data = _vec(x)
        n, edges = np.histogram(data[np.isfinite(data)], bins=bins, range=range, weights=weights,
                                density=density)
        if cumulative:
            n = np.cumsum(n * np.diff(edges)) if density else np.cumsum(n)
        centers = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges)
        if not np.allclose(widths, widths[0]):
            raise ValueError("hist(): sextant draws equal-width bins")
        if orientation not in ("vertical", "horizontal"):
            raise ValueError(f"hist(): orientation must be 'vertical' or 'horizontal', not {orientation!r}")
        bars = self._bar_series("hist", centers, n, widths[0] * (rwidth if rwidth is not None else 1.0), bottom,
                                "center", orientation, color=color, label=label, alpha=alpha,
                                edgecolor=edgecolor, linewidth=linewidth, **kw)
        return n, edges, bars

    def _c0(self):
        """matplotlib's 'C0': the cycle's first colour, which axhline() and axhspan()
        take without moving the cycle on."""
        return self._cycle[0] if self._cycle is not None else _hex(TAB10[0])

    @_zorder(2)
    @_takes_visible
    def axhline(self, y=0, xmin=0, xmax=1, **kw):
        """A line across the axes at y, from xmin to xmax (fractions of the frame)."""
        return self._axline(y, xmin, xmax, "horizontal", kw)

    @_zorder(2)
    @_takes_visible
    def axvline(self, x=0, ymin=0, ymax=1, **kw):
        """A line across the axes at x, from ymin to ymax (fractions of the frame)."""
        return self._axline(x, ymin, ymax, "vertical", kw)

    def _axline(self, value, lo, hi, orientation, kw):
        opts, label = self._axline_style(kw)
        opts.update(orientation=orientation, from_=float(lo), to=float(hi))
        i, entry = self._add("axline", "axline", float(value), **opts)
        return RefLine(self, "axline", i, entry, label)

    def _axline_style(self, kw):
        _drop_ignored(kw)
        color = to_rgba(_pop(kw, "color", "c")) or self._c0()
        lw = _pop(kw, "linewidth", "lw")
        ls = _linestyle(_pop(kw, "linestyle", "ls"))
        alpha = kw.pop("alpha", None)
        label = kw.pop("label", None)
        opts = dict(kw, color=color, linewidth=(1.5 if lw is None else float(lw)) * PT)
        if ls is not None:
            opts["linestyle"] = ls
        if alpha is not None:
            opts["alpha"] = float(alpha)
        if label is not None:
            opts["name"] = str(label)
        return opts, label

    @_zorder(2)
    @_takes_visible
    def axline(self, xy1, xy2=None, *, slope=None, **kw):
        """The infinite line through xy1 and xy2, or through xy1 with `slope`, as matplotlib's;
        points in data coordinates (a log axis draws the curve). transform= is not mapped."""
        if kw.pop("transform", None) not in (None, self.transData):
            raise NotImplementedError("axline(): sextant draws it in data coordinates (transform= is not mapped)")
        if xy2 is not None and slope is not None:
            raise TypeError("Cannot specify both 'xy2' and 'slope'")
        if xy2 is None and slope is None:
            raise TypeError("Must specify either 'xy2' or 'slope'")
        opts, label = self._axline_style(kw)
        x1, y1 = (float(v) for v in xy1)
        if xy2 is None:
            args, opts["slope"] = [x1, y1], float(slope)
        else:
            args = [x1, y1, *(float(v) for v in xy2)]
        i, entry = self._add("axline", "axline", *args, **opts)
        return AxLine(self, "axline", i, entry, label)

    @_zorder(1)
    @_takes_visible
    def axhspan(self, ymin, ymax, xmin=0, xmax=1, **kw):
        """A band over y in [ymin, ymax], from xmin to xmax (fractions of the frame).
        Opaque C0 by default, as in matplotlib; pass alpha= to see through it."""
        return self._axspan(ymin, ymax, xmin, xmax, "horizontal", kw)

    @_zorder(1)
    @_takes_visible
    def axvspan(self, xmin, xmax, ymin=0, ymax=1, **kw):
        """A band over x in [xmin, xmax], from ymin to ymax (fractions of the frame)."""
        return self._axspan(xmin, xmax, ymin, ymax, "vertical", kw)

    def _axspan(self, lo, hi, flo, fhi, orientation, kw):
        _drop_ignored(kw)
        color = _pop(kw, "color", "c")
        face = _pop(kw, "facecolor", "fc", default=color)
        edge = _pop(kw, "edgecolor", "ec")
        lw = _pop(kw, "linewidth", "lw")
        alpha = kw.pop("alpha", None)
        label = kw.pop("label", None)
        opts = dict(kw, orientation=orientation, color=to_rgba(face) if face is not None else self._c0(),
                    alpha=1.0 if alpha is None else float(alpha), from_=float(flo), to=float(fhi))
        # An outline only when one is asked for: matplotlib's default edge is invisible.
        if edge is not None and not (isinstance(edge, str) and edge.lower() == "none"):
            opts.update(edgecolor=to_rgba(edge), linewidth=(1.0 if lw is None else float(lw)) * PT)
        elif color is not None and lw is not None:  # color= is the edge's too
            opts["linewidth"] = float(lw) * PT
        if label is not None:
            opts["name"] = str(label)
        i, entry = self._add("axspan", "axspan", float(lo), float(hi), **opts)
        return Span(self, "axspan", i, entry, label)

    @_zorder(1)
    def add_patch(self, p):
        """Draw a sextant.patches Circle, Wedge or Polygon in this axes (matplotlib's name);
        returns it."""
        if not isinstance(p, _Patch):
            raise TypeError(f"add_patch(): sextant draws Circle, Wedge and Polygon patches, not {type(p).__name__}")
        if p.axes is not None:
            raise ValueError("Can not reset the Axes. You are probably trying to re-use an artist in more "
                             "than one Axes which is not supported")
        return p._add_to(self)

    @_zorder(1)
    def pie(self, x, explode=None, labels=None, colors=None, autopct=None, pctdistance=0.6, shadow=False,
            labeldistance=1.1, startangle=0, radius=1, counterclock=True, wedgeprops=None, textprops=None,
            center=(0, 0), frame=False, rotatelabels=False, *, normalize=True, hatch=None, data=None):
        """matplotlib's pie(): (wedges, texts), and autotexts with autopct -- a PieWedge
        per slice, a PieText per label and per percent. One sextant pie draws them all.
        Turns the axis off unless frame=True; the aspect becomes 1. wedgeprops takes
        width, edgecolor, linewidth and alpha; textprops color and fontsize."""
        if data is not None and isinstance(x, str):
            x = data[x]
        if callable(autopct):
            raise TypeError("pie(): sextant formats the percents itself; give autopct as a format string "
                            "such as '%1.1f%%'")
        if autopct is not None and not isinstance(autopct, str):
            raise TypeError("pie(): autopct must be a format string")
        if shadow:
            raise ValueError("pie(): sextant draws no shadow")
        if hatch is not None:
            raise ValueError("pie(): sextant draws no hatching")
        if rotatelabels:
            raise ValueError("pie(): sextant draws the labels horizontal (rotatelabels=False)")
        values = _vec(x)
        n = len(values)
        radius = float(radius)
        opts = {"radius": radius, "pctdistance": float(pctdistance), "startangle": float(startangle),
                "counterclock": bool(counterclock), "normalize": bool(normalize),
                "labeldistance": None if labeldistance is None else float(labeldistance),
                "cx": float(center[0]), "cy": float(center[1])}
        if labels is not None:
            opts["labels"] = [str(s) for s in labels]
        if explode is not None:
            # matplotlib's explode is in data units, sextant's a fraction of the radius.
            opts["explode"] = _vec(explode) / radius if radius > 0 else _vec(explode)
        if autopct is not None:
            opts["autopct"] = autopct
        # The slices take the patch cycle in turn, as matplotlib's; colors= repeats.
        if colors is None:
            opts["colors"] = [self._next_color(patches=True) for _ in range(n)]
        else:
            colors = list(colors)
            opts["colors"] = [to_rgba(colors[k % len(colors)]) for k in range(n)] if colors else []
        w = dict(wedgeprops or {})
        _drop_ignored(w)
        width = w.pop("width", None)
        if width is not None:
            opts["inner"] = radius - float(width)
        edge = _pop(w, "edgecolor", "ec")
        lw = _pop(w, "linewidth", "lw")
        if edge is not None and not (isinstance(edge, str) and edge.lower() in ("none", "face")):
            opts.update(edgecolor=to_rgba(edge), linewidth=(1.0 if lw is None else float(lw)) * PT)
        alpha = w.pop("alpha", None)
        if alpha is not None:
            opts["alpha"] = float(alpha)
        if w:
            raise TypeError(f"pie(): unsupported wedgeprops key(s) {sorted(w)} (width, edgecolor, linewidth, alpha)")
        t = dict(textprops or {})
        _drop_ignored(t)
        color = _pop(t, "color", "c")
        if color is not None:
            opts["textcolor"] = to_rgba(color)
        size = _pop(t, "fontsize", "size")
        if size is not None:
            opts["fontsize"] = _fontsize(size)
        if t:
            raise TypeError(f"pie(): unsupported textprops key(s) {sorted(t)} (color, fontsize)")
        i, entry = self._add("pie", "pie", values, **opts)
        if frame:
            self._call("set_axis_visible", True)
        pie = _Pie(self, i, entry)
        pie.wedges = [PieWedge(pie, k) for k in range(n)]
        texts = [PieText(pie, k, False) for k in range(n)] if labeldistance is not None else []
        if autopct is None:
            return pie.wedges, texts
        return pie.wedges, texts, [PieText(pie, k, True) for k in range(n)]

    # --- filled areas --------------------------------------------------------------

    @_zorder(1)
    @_takes_visible
    def fill_between(self, x, y1, y2=0, where=None, interpolate=False, step=None, *, data=None, **kw):
        """The area between y1(x) and y2(x), as matplotlib's. Returns a
        FillBetweenPolyCollection."""
        return self._fill_between("fill_between", x, y1, y2, where, interpolate, step, "horizontal", data, kw)

    @_zorder(1)
    @_takes_visible
    def fill_betweenx(self, y, x1, x2=0, where=None, step=None, interpolate=False, *, data=None, **kw):
        """The area between x1(y) and x2(y), as matplotlib's."""
        return self._fill_between("fill_betweenx", y, x1, x2, where, interpolate, step, "vertical", data, kw)

    def _fill_between(self, method, t, f1, f2, where, interpolate, step, orientation, data, kw):
        if data is not None:
            t, f1, f2, where = (data[a] if isinstance(a, str) and a in data else a for a in (t, f1, f2, where))
        t = _vec(t)
        f1 = np.broadcast_to(np.asarray(f1, dtype=float), t.shape).astype(float)
        f2 = float(f2) if np.ndim(f2) == 0 else _vec(f2)
        color = kw.pop("color", None)
        face = _pop(kw, "facecolor", "facecolors", "fc", default=color)
        edge = _pop(kw, "edgecolor", "edgecolors", "ec", default=color)  # color= is the outline's too
        label = kw.pop("label", None)
        face = to_rgba(face) if face is not None else self._next_color(patches=True)
        if isinstance(edge, str) and edge == "face":
            edge = face
        opts = _area_style(method, kw, edge, step)
        opts["color"] = face
        if orientation != "horizontal":
            opts["orientation"] = orientation
        exclude = _exclude(where, len(t), method)
        if exclude is not None:
            opts["exclude"] = exclude
        if interpolate:
            opts["interpolate"] = True
        if label is not None:
            opts["name"] = str(label)
        lw = opts.pop("_linewidth")
        i, entry = self._add("fill", "fill", t, f1, f2, **opts)
        return FillBetweenPolyCollection(self, "fill", i, entry, label, lw)

    @_zorder(1)
    @_takes_visible
    def stackplot(self, x, *args, labels=(), colors=None, hatch=None, baseline="zero", data=None, **kw):
        """Stacked areas, as matplotlib's: one FillBetweenPolyCollection per layer, in a
        list. Layers past `labels` are unnamed; `colors` repeats as needed (unset: the
        line colour cycle, one per layer)."""
        if hatch is not None:
            raise ValueError("stackplot(): sextant draws no hatching")
        if data is not None:
            x = data[x] if isinstance(x, str) and x in data else x
            args = [data[a] if isinstance(a, str) and a in data else a for a in args]
        if not args:
            raise TypeError("stackplot(x, y): no layers given")
        x = _vec(x)
        ys = np.vstack([np.asarray(a, dtype=float) for a in args])
        n = len(ys)
        edge = _pop(kw, "edgecolor", "edgecolors", "ec")
        opts = _area_style("stackplot", kw, edge, kw.pop("step", None))
        lw = opts.pop("_linewidth")
        if colors is not None:
            colors = list(colors)
            fill_colors = [to_rgba(colors[k % len(colors)]) for k in range(n)]
        else:
            fill_colors = [self._next_color() for _ in range(n)]
        labels = [str(s) for s in labels][:n]
        names = labels + [""] * (n - len(labels))
        # The core computes the curves; each layer is then recorded as the plain fill it
        # is, so a layer's own setters and a replay act on it alone.
        i0 = self.core.fill_count()
        self.core.stack(x, ys, baseline=baseline, colors=fill_colors, names=names, **opts)
        handles = []
        for k in range(n):
            d = self.core.fill_data(i0 + k)
            fopts = dict(opts, color=fill_colors[k])
            if names[k]:
                fopts["name"] = names[k]
            if d.exclude is not None:
                fopts["exclude"] = d.exclude
            entry = ["fill", [d.x, d.y1, d.y2], fopts]
            self._log.append(entry)
            handles.append(FillBetweenPolyCollection(self, "fill", i0 + k, entry, names[k] or None, lw))
        self.figure._changed()
        return handles

    # --- arrow fields ---------------------------------------------------------------

    @_zorder(1)
    @_takes_visible
    def quiver(self, *args, scale=None, scale_units=None, units="width", angles="uv", width=None, headwidth=3,
               headlength=5, pivot="tail", color=None, cmap=None, norm=None, clim=None, alpha=None, label=None,
               colorbar=False, data=None, **kw):
        """quiver([X, Y], U, V, [C], **kwargs), as matplotlib's, drawn in data units: an
        arrow runs from (x, y) by (u, v) / scale (angles='uv' draws as 'xy', which is the
        same at equal aspect). Unset scale picks matplotlib's mean arrow length; a scale
        is read in scale_units 'xy', 'x', 'y', 'width' or 'height'. Returns a Quiver."""
        _drop_ignored(kw)
        for k in ("headaxislength", "minshaft", "minlength"):  # shape details sextant has no knob for
            kw.pop(k, None)
        if data is not None:
            args = [data[a] if isinstance(a, str) and a in data else a for a in args]
        x, y, u, v, c = _quiver_args(args)
        if not (isinstance(angles, str) and angles in ("uv", "xy")):
            raise ValueError("quiver(): sextant draws angles='xy' (and 'uv' as 'xy'), not per-arrow angles")
        lw = _pop(kw, "linewidths", "linewidth", "lw")
        edge = _pop(kw, "edgecolors", "edgecolor", "ec")
        if lw is not None and np.any(np.asarray(lw, dtype=float) > 0):
            raise ValueError("quiver(): sextant draws arrows without an outline")
        del edge
        face = _pop(kw, "facecolors", "facecolor", "fc", default=color)
        w_px, h_px = self._frame_px()
        if width is None:
            shaft = 0.06 / min(max(math.sqrt(len(x)), 8.0), 25.0) * w_px
        else:
            per = {"width": w_px, "height": h_px, "dots": 100.0 / self.figure._dpi, "inches": 100.0}.get(units)
            if per is None:
                raise ValueError(f"quiver(): sextant sizes arrows in pixels; width= in units {units!r} needs "
                                 "'width', 'height', 'dots' or 'inches'")
            shaft = float(width) * per
        opts = dict(kw, amp=_quiver_amp(x, y, u, v, scale, scale_units or units), pivot=pivot, width=shaft,
                    head_width=float(headwidth) * shaft, head_length=float(headlength) * shaft)
        if alpha is not None:
            opts["alpha"] = float(alpha)
        if label is not None:
            opts["name"] = str(label)
        if c is not None:
            cv = clim if clim is not None else (None, None)
            opts.update(_norm_opts(c, cv[0], cv[1], norm), colorbar=bool(colorbar))
            cmap = _rc_cmap(cmap)
            if cmap is not None:
                opts["cmap"] = cmap
            i, entry = self._add("quiver", "quiver", x, y, u, v, c, **opts)
        else:
            opts["color"] = to_rgba(face) if face is not None else (0.0, 0.0, 0.0, 1.0)  # matplotlib's black
            i, entry = self._add("quiver", "quiver", x, y, u, v, **opts)
        handle = Quiver(self, "quiver", i, entry, label)
        if c is not None:
            self.figure._mappable = handle
        return handle

    def _frame_px(self):
        """The frame's (width, height) in px at matplotlib's default subplot layout
        (left .125, right .9, bottom .11, top .88, wspace = hspace = .2): what quiver's
        'width'/'height' units are measured against."""
        ax = self._host or self
        (R, C), first, last = ax._span if ax._span is not None else ((1, 1), 1, 1)
        W, H = self.figure._size
        r0, c0 = divmod(first - 1, C)
        r1, c1 = divmod(last - 1, C)
        cw = 0.775 * W / (C + 0.2 * (C - 1))
        ch = 0.77 * H / (R + 0.2 * (R - 1))
        k, m = c1 - c0 + 1, r1 - r0 + 1
        return k * cw + (k - 1) * 0.2 * cw, m * ch + (m - 1) * 0.2 * ch

    @_zorder(2)
    @_takes_visible
    def errorbar(self, x, y, yerr=None, xerr=None, fmt="", ecolor=None, elinewidth=None, capsize=None,
                 label=None, **kw):
        _drop_ignored(kw)
        for k in ("capthick", "barsabove", "errorevery", "lolims", "uplims", "xlolims", "xuplims"):
            kw.pop(k, None)
        x, y = _vec(x), _vec(y)
        if fmt == "none":  # error bars alone
            fmt, kw["linestyle"] = "", "none"
        ls, mk, fc = parse_fmt(fmt)
        mk = kw.pop("marker", mk)
        if mk in ("", " ", "none", "None"):
            mk = None
        color = to_rgba(_pop(kw, "color", "c", default=fc)) or self._next_color()
        ls = _linestyle(_pop(kw, "linestyle", "ls", default=ls if ls is not None else ("-" if mk is None else "none")))
        lw = _pop(kw, "linewidth", "lw")
        alpha = kw.pop("alpha", None)
        kw.pop("markersize", None), kw.pop("ms", None)
        eopts = {}
        err = self._err(len(x), xerr, yerr, ecolor, capsize, {"elinewidth": elinewidth}, eopts)
        common = dict(kw)
        if alpha is not None:
            common["alpha"] = alpha
        if label is not None:
            common["name"] = str(label)
        if mk is None or ls != "none":
            opts = dict(common, color=color, linestyle=ls, linewidth=(lw if lw is not None else 1.5) * PT, **eopts)
            i, entry = self._add("line", "line", x, y, err=err, **opts)
            handle = Line2D(self, "line", i, entry, label)
            if mk is not None:
                j, mentry = self._add("scatter", "scatter", x, y, color=color, marker=_marker(mk),
                                      size=6.0 * PT, alpha=1.0, show_legend=False)
                handle._markers = (j, mentry)
            return handle
        opts = dict(common, color=color, marker=_marker(mk), size=6.0 * PT, **eopts)
        opts.setdefault("alpha", 1.0)
        i, entry = self._add("scatter", "scatter", x, y, err=err, **opts)
        return Line2D(self, "scatter", i, entry, label)

    @staticmethod
    def _err(n, xerr, yerr, ecolor, capsize, error_kw, opts):
        """ErrorBar from matplotlib's xerr/yerr, and its style into opts['errorbar']."""
        xlo, xhi = _errorbar(xerr, n, "xerr")
        ylo, yhi = _errorbar(yerr, n, "yerr")
        if all(v is None for v in (xlo, xhi, ylo, yhi)):
            return None
        style = {"capsize": float(capsize) * PT if capsize is not None else 0.0}
        error_kw = dict(error_kw or {})
        ecolor = error_kw.pop("ecolor", ecolor)
        if ecolor is not None:
            style["color"] = to_rgba(ecolor)
        elw = _pop(error_kw, "elinewidth", "lw", "linewidth")
        if elw is not None:
            style["linewidth"] = float(elw) * PT
        if "capsize" in error_kw:
            style["capsize"] = float(error_kw["capsize"]) * PT
        opts["errorbar"] = style
        return _sextant.ErrorBar(x_cap_lo=xlo, x_cap_hi=xhi, y_cap_lo=ylo, y_cap_hi=yhi)

    @_zorder(0)
    @_takes_visible
    def imshow(self, X, cmap=None, vmin=None, vmax=None, origin=None, extent=None, alpha=None,
               label=None, colorbar=False, aspect=None, **kw):
        """matplotlib's imshow() for 2-D scalar data or (rows, cols, 3 or 4) RGB(A) (alpha= multiplies
        the alpha channel of RGB(A) data only). Unlike matplotlib, the frame keeps
        its shape unless aspect= is given ('equal', 'auto' or a number; see set_aspect())."""
        _drop_ignored(kw)
        norm = kw.pop("norm", None)
        a = np.asarray(X)
        if a.dtype != np.uint8:
            a = a.astype(float)
        if a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] not in (3, 4)):
            raise ValueError("imshow(): data is (rows, cols) or (rows, cols, 3 or 4)")
        rgb = a.ndim == 3
        rows, cols = a.shape[:2]
        origin = origin or "upper"
        if rgb:
            opts = dict(kw)
            if alpha is not None:  # multiplies the alpha channel
                f = a / 255.0 if a.dtype == np.uint8 else a
                al = np.broadcast_to(np.asarray(alpha, dtype=float), (rows, cols))
                f = np.concatenate([f, np.ones((rows, cols, 1))], axis=2) if f.shape[2] == 3 else f.copy()
                f[:, :, 3] = f[:, :, 3] * al
                a = f
        else:
            opts = dict(kw, colorbar=bool(colorbar), **_norm_opts(a, vmin, vmax, norm))
            cmap = _rc_cmap(cmap)
            if cmap is not None:
                opts["cmap"] = cmap
        if label is not None:
            opts["name"] = str(label)
        if extent is None:
            # matplotlib's: pixel centres on integers; 'upper' puts row 0 at the
            # top with the y axis pointing down.
            xr, yr = (-0.5, cols - 0.5), (-0.5, rows - 0.5)
            i, entry = self._place_image(rgb, a, xr, yr, "lower", opts)
            if origin == "upper":
                self._call("set_ylim", rows - 0.5, -0.5)
        else:
            left, right, bottom, top = extent
            i, entry = self._place_image(rgb, a, (left, right), (bottom, top), origin, opts)
        if alpha is not None and not rgb:
            warnings.warn("imshow(): alpha is not supported by sextant scalar heatmaps; ignored", stacklevel=2)
        if aspect is not None:
            self.set_aspect(aspect)
        handle = AxesImage(self, "heatmap", i, entry, label)
        self.figure._mappable = handle
        return handle

    @_zorder(1)
    @_takes_visible
    def tripcolor(self, *args, triangles=None, cmap=None, norm=None, vmin=None, vmax=None, shading="gouraud",
                  label=None, colorbar=False, **kw):
        """tripcolor(x, y, C, [triangles]) or tripcolor(triang, C): one value per vertex,
        interpolated across each triangle (shading='gouraud'; without triangles, the Delaunay
        triangulation). Returns a TriMesh."""
        _drop_ignored(kw)
        for k in ("alpha", "edgecolors", "facecolors", "linewidth", "linewidths", "antialiased"):
            kw.pop(k, None)
        if shading != "gouraud":
            raise ValueError("tripcolor(): sextant draws shading='gouraud' only")
        if len(args) == 2 and hasattr(args[0], "triangles"):
            tri, C = args
            x, y, triangles = tri.x, tri.y, tri.triangles
        elif len(args) == 3:
            x, y, C = args
        elif len(args) == 4:
            x, y, triangles, C = args
        else:
            raise TypeError("tripcolor(x, y, C, [triangles]) or tripcolor(triangulation, C)")
        x, y, C = _vec(x), _vec(y), _vec(C)
        if len(C) != len(x):
            raise ValueError("tripcolor(): one value per vertex is drawn; C must be as long as x")
        opts = dict(kw, colorbar=bool(colorbar), **_norm_opts(C, vmin, vmax, norm))
        cmap = _rc_cmap(cmap)
        if cmap is not None:
            opts["cmap"] = cmap
        if label is not None:
            opts["name"] = str(label)
        tri = None if triangles is None else np.asarray(triangles)
        i, entry = self._add("heatmap_tri", "heatmap_tri", x, y, C, tri, **opts)
        handle = TriMesh(self, "heatmap_tri", i, entry, label)
        self.figure._mappable = handle
        return handle

    @_zorder(1)
    @_takes_visible
    def hist2d(self, x, y, bins=10, range=None, density=False, weights=None, cmin=None, cmax=None,
               cmap=None, norm=None, vmin=None, vmax=None, label=None, colorbar=False, **kw):
        """matplotlib's hist2d(): the counts (or density) of each bin as a heatmap. Returns
        (counts, xedges, yedges, image) with the counts shaped (nx, ny) as matplotlib's.
        weights= sums them per bin here and draws the sums."""
        _drop_ignored(kw)
        x, y = _vec(x), _vec(y)
        ok = np.isfinite(x) & np.isfinite(y)
        if weights is not None:  # sextant counts points: bin here and draw the sums
            w = _vec(weights)
            if len(w) != len(x):
                raise ValueError("hist2d(): weights must be as long as x")
            ok &= np.isfinite(w)
            counts, xe, ye = np.histogram2d(x[ok], y[ok], bins=bins, range=range, weights=w[ok],
                                            density=bool(density))
            if cmin is not None:
                counts = np.where(counts < cmin, np.nan, counts)
            if cmax is not None:
                counts = np.where(counts > cmax, np.nan, counts)
            opts = dict(kw, colorbar=bool(colorbar), **_norm_opts(counts, vmin, vmax, norm))
            cmap = _rc_cmap(cmap)
            if cmap is not None:
                opts["cmap"] = cmap
            if label is not None:
                opts["name"] = str(label)
            i, entry = self._add("heatmap", "heatmap_rec", np.ascontiguousarray(counts.T), xe, ye, **opts)
            handle = AxesImage(self, "heatmap", i, entry, label)
            self.figure._mappable = handle
            return counts, xe, ye, handle
        opts = dict(kw, density=bool(density), colorbar=bool(colorbar))
        if range is not None:
            (opts["xrange"], opts["yrange"]) = (tuple(float(v) for v in r) for r in range)
        if np.ndim(bins) == 0:
            opts["xbins"] = opts["ybins"] = int(bins)
        elif len(bins) == 2:
            for ax_, b in zip("xy", bins):
                if np.ndim(b) == 0:
                    opts[ax_ + "bins"] = int(b)
                else:
                    opts[ax_ + "_edges"] = _vec(b)
        else:
            raise ValueError("hist2d(): bins is an int, (nx, ny) or (xedges, yedges)")
        if cmin is not None:
            opts["cmin"] = float(cmin)
        if cmax is not None:
            opts["cmax"] = float(cmax)
        cmap = _rc_cmap(cmap)
        if cmap is not None:
            opts["cmap"] = cmap
        core, nvmin, nvmax = resolve_norm(norm)
        if core is not None:
            opts["norm"] = core
        for key, v in (("vmin", nvmin if vmin is None else vmin), ("vmax", nvmax if vmax is None else vmax)):
            if v is not None:
                opts[key] = float(v)
        if label is not None:
            opts["name"] = str(label)
        i, entry = self._add("heatmap", "hist2d", x, y, **opts)
        handle = AxesImage(self, "heatmap", i, entry, label)
        self.figure._mappable = handle
        hist_bins = [opts.get("x_edges", opts.get("xbins", 10)), opts.get("y_edges", opts.get("ybins", 10))]
        counts, xe, ye = np.histogram2d(x[ok], y[ok], bins=hist_bins, range=range, density=bool(density))
        if cmin is not None:
            counts = np.where(counts < cmin, np.nan, counts)
        if cmax is not None:
            counts = np.where(counts > cmax, np.nan, counts)
        return counts, xe, ye, handle

    @staticmethod
    def _grid_axes(X, Y, name):
        """The 1-D x and y of a meshgrid pair (or of 1-D inputs); raises for a curvilinear grid."""
        if X.ndim == 2 and Y.ndim == 2:
            if not (np.array_equal(X, np.broadcast_to(X[0], X.shape))
                    and np.array_equal(Y, np.broadcast_to(Y[:, :1], Y.shape))):
                raise ValueError(f"{name}(): X and Y must form a rectilinear meshgrid (sextant has no curvilinear grids)")
            return X[0], Y[:, 0]
        if X.ndim == 1 and Y.ndim == 1:
            return X, Y
        raise ValueError(f"{name}(): X and Y are both 1-D or both 2-D")

    @staticmethod
    def _contour_args(args, name):
        """contour([X, Y], Z, [levels]) as 1-D x, 1-D y, Z and the levels (or None)."""
        args = list(args)
        if len(args) in (2, 4):
            levels = args.pop()
        else:
            levels = None
        if len(args) == 1:
            Z = np.asarray(args[0], dtype=float)
            x, y = np.arange(Z.shape[1], dtype=float), np.arange(Z.shape[0], dtype=float)
        elif len(args) == 3:
            X, Y, Z = (np.asarray(a, dtype=float) for a in args)
            x, y = Axes._grid_axes(X, Y, name)
        else:
            raise TypeError(f"{name}(Z, [levels]) or {name}(X, Y, Z, [levels])")
        if Z.ndim != 2:
            raise ValueError(f"{name}(): Z must be 2-D")
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float), Z, levels

    def _contour(self, filled, args, levels, colors, cmap, norm, vmin, vmax, linewidths, alpha, label, colorbar,
                 fontsize, kw):
        _drop_ignored(kw)
        name = "contourf" if filled else "contour"
        origin, extent = kw.pop("origin", None), kw.pop("extent", None)
        for k, ok_ in (("linestyles", ("solid", None)), ("negative_linestyles", ("solid", None)),
                       ("extend", ("neither", None)), ("hatches", (None,))):
            v = kw.pop(k, None)
            if v not in ok_ if isinstance(v, (str, type(None))) else True:
                raise ValueError(f"{name}(): {k}={v!r} is not supported (sextant draws solid lines, no extension or hatching)")
        for k in ("antialiased", "locator", "corner_mask"):
            kw.pop(k, None)
        x, y, Z, plevels = self._contour_args(args, name)
        if origin is not None or extent is not None:
            if len(args) not in (1, 2):
                raise ValueError(f"{name}(): origin/extent apply to {name}(Z, [levels]) only")
            rows, cols = Z.shape
            if origin not in (None, "upper", "lower", "image"):
                raise ValueError(f"{name}(): origin is 'upper' or 'lower'")
            if extent is None:
                x, y = np.arange(cols, dtype=float), np.arange(rows, dtype=float)
            else:
                x0, x1, y0, y1 = (float(v) for v in extent)
                if origin is None:  # corners are the positions of Z[0, 0] and Z[-1, -1]
                    x, y = np.linspace(x0, x1, cols), np.linspace(y0, y1, rows)
                else:  # pixel boundaries, as imshow's
                    x = x0 + (np.arange(cols) + 0.5) * (x1 - x0) / cols
                    y = y0 + (np.arange(rows) + 0.5) * (y1 - y0) / rows
            if origin in ("upper", "image"):  # row 0 at the top
                y = y[::-1]
            if y[0] > y[-1]:  # the core wants ascending axes
                y, Z = y[::-1], Z[::-1]
            if x[0] > x[-1]:
                x, Z = x[::-1], Z[:, ::-1]
        levels = plevels if plevels is not None else levels
        opts = dict(kw, colorbar=bool(colorbar))
        if levels is not None:
            if np.ndim(levels) == 0:
                opts["nlevels"] = int(levels)
            else:
                opts["levels"] = _vec(levels)
        if colors is not None:
            if filled:
                raise ValueError("contourf(): colors= is not supported; use cmap=")
            cs = [colors] if isinstance(colors, str) or np.ndim(colors) == 1 and np.asarray(colors).dtype.kind in "fiu"                 else list(colors)
            if len(cs) != 1:
                raise ValueError(f"{name}(): one line colour is drawn; colors= is a single colour")
            opts["line_color"] = to_rgba(cs[0])
        if linewidths is not None and not filled:
            opts["linewidth"] = float(np.asarray(linewidths).flat[0]) * PT
        if cmap is not None:
            opts["cmap"] = cmap
        core, nvmin, nvmax = resolve_norm(norm)
        if core is not None:
            opts["norm"] = core
        for key, v in (("vmin", nvmin if vmin is None else vmin), ("vmax", nvmax if vmax is None else vmax)):
            if v is not None:
                opts[key] = float(v)
        if fontsize is not None:
            opts["fontsize"] = float(fontsize) * PT
        if alpha is not None:
            warnings.warn("contour(): alpha is not supported by sextant contours; ignored", stacklevel=3)
        if label is not None:
            opts["name"] = str(label)
        i, entry = self._add("contour", "contourf" if filled else "contour", x, y, Z, **opts)
        handle = ContourSet(self, "contour", i, entry, label)
        if filled or colors is None:
            self.figure._mappable = handle
        return handle

    @_zorder(2)
    @_takes_visible
    def contour(self, *args, levels=None, colors=None, cmap=None, norm=None, vmin=None, vmax=None,
                linewidths=None, alpha=None, label=None, colorbar=False, **kw):
        """matplotlib's contour([X, Y], Z, [levels]): lines at the levels over a grid (X and
        Y 1-D, or the meshgrid). Returns a ContourSet; ax.clabel(cs) labels the lines."""
        return self._contour(False, args, levels, colors, cmap, norm, vmin, vmax, linewidths, alpha, label,
                             colorbar, None, kw)

    @_zorder(1)
    @_takes_visible
    def contourf(self, *args, levels=None, colors=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None,
                 label=None, colorbar=False, **kw):
        """matplotlib's contourf([X, Y], Z, [levels]): bands between the levels."""
        return self._contour(True, args, levels, colors, cmap, norm, vmin, vmax, None, alpha, label,
                             colorbar, None, kw)

    def clabel(self, cs, levels=None, *, fontsize=None, inline=True, **kw):
        """Label a contour's lines inline with their levels (the levels argument, inline=False
        and formatting keywords are not offered)."""
        if levels is not None or not inline:
            raise ValueError("clabel(): sextant labels every line inline; levels= and inline=False are not offered")
        _drop_ignored(kw)
        extra = {} if fontsize is None else {"fontsize": float(fontsize) * PT}
        cs._set_options(labels=True, **extra)
        return []

    def _place_image(self, rgb, a, xr, yr, origin, opts):
        if rgb:
            return self._add("heatmap", "imshow_rgb", a, xrange=xr, yrange=yr, origin=origin, **opts)
        return self._add("heatmap", "heatmap", a, xr, yr, origin=origin, **opts)

    @_zorder(1)
    @_takes_visible
    def pcolormesh(self, *args, cmap=None, vmin=None, vmax=None, shading=None, **kw):
        _drop_ignored(kw)
        if len(args) == 1:
            C = np.asarray(args[0], dtype=float)
            x = np.arange(C.shape[1] + 1, dtype=float)
            y = np.arange(C.shape[0] + 1, dtype=float)
        elif len(args) == 3:
            x, y, C = (np.asarray(a, dtype=float) for a in args)
            x, y = Axes._grid_axes(x, y, "pcolormesh")
        else:
            raise TypeError("pcolormesh(C) or pcolormesh(X, Y, C)")
        rows, cols = C.shape
        if len(x) == cols:  # centres, as shading='nearest'/'auto' reads them
            dx = np.diff(x)
            x = np.concatenate([[x[0] - dx[0] / 2], x[:-1] + dx / 2, [x[-1] + dx[-1] / 2]])
        if len(y) == rows:
            dy = np.diff(y)
            y = np.concatenate([[y[0] - dy[0] / 2], y[:-1] + dy / 2, [y[-1] + dy[-1] / 2]])
        if any(not np.allclose(np.diff(e), np.diff(e)[0]) for e in (x, y)):
            norm = kw.pop("norm", None)
            if kw.pop("alpha", None) is not None:
                warnings.warn("pcolormesh(): alpha is not supported by sextant heatmaps; ignored", stacklevel=2)
            for k in ("edgecolors", "linewidth", "linewidths", "antialiased", "snap", "rasterized"):
                kw.pop(k, None)
            colorbar, label = kw.pop("colorbar", False), kw.pop("label", None)
            opts = dict(kw, colorbar=bool(colorbar), **_norm_opts(C, vmin, vmax, norm))
            if label is not None:
                opts["name"] = str(label)
            cmap = _rc_cmap(cmap)
            if cmap is not None:
                opts["cmap"] = cmap
            i, entry = self._add("heatmap", "heatmap_rec", C, x, y, **opts)
            handle = AxesImage(self, "heatmap", i, entry, label)
            self.figure._mappable = handle
            return handle
        return self.imshow(C, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower",
                           extent=(x[0], x[-1], y[0], y[-1]), **kw)

    # --- text ----------------------------------------------------------------------

    @_takes_visible
    def text(self, x, y, s, fontdict=None, *, transform=None, **kw):
        """Text at (x, y) in data coordinates, or in the axes' fraction with
        transform=ax.transAxes. Returns a Text."""
        coords = self._coords(transform, "text") or "data"
        opts = _text_opts(kw, fontdict)
        i, entry = self._add("text", "text", str(s), float(x), float(y), coords=coords, **opts)
        return Text(self, i, entry, (0, 1, 2))

    @_takes_visible
    def annotate(self, text, xy, xytext=None, xycoords="data", textcoords=None, arrowprops=None,
                 annotation_clip=None, **kw):
        """matplotlib's annotate(). xycoords 'data' (or 'axes fraction' without an
        arrow); textcoords 'data', 'axes fraction', 'offset points' or 'offset pixels'.
        The arrow is drawn when arrowprops is given. Returns a Text."""
        del annotation_clip
        if xycoords not in _TEXT_COORDS:
            raise ValueError(f"annotate(): xycoords {xycoords!r} is not one sextant has; use 'data' or 'axes fraction'")
        if arrowprops is not None and xycoords != "data":
            raise ValueError("annotate(): sextant's arrow points at a data point; use xycoords='data'")
        textcoords = textcoords or xycoords
        opts = _text_opts(kw)
        px, py = (float(v) for v in xy)
        if xytext is None:
            tx, ty, coords = px, py, _TEXT_COORDS[xycoords]
        elif textcoords in ("offset points", "offset pixels"):
            scale = PT if textcoords == "offset points" else 1.0
            tx, ty, coords = px, py, _TEXT_COORDS[xycoords]
            opts["dx"] = opts.get("dx", 0.0) + float(xytext[0]) * scale
            opts["dy"] = opts.get("dy", 0.0) + float(xytext[1]) * scale
        elif textcoords in _TEXT_COORDS:
            tx, ty, coords = float(xytext[0]), float(xytext[1]), _TEXT_COORDS[textcoords]
        else:
            raise ValueError(f"annotate(): textcoords {textcoords!r} is not one sextant has; use 'data', "
                             "'axes fraction', 'offset points' or 'offset pixels'")
        if arrowprops is None:
            i, entry = self._add("text", "text", str(text), tx, ty, coords=coords, **opts)
            return Text(self, i, entry, (0, 1, 2))
        i, entry = self._add("text", "annotate", px, py, str(text), tx, ty, coords=coords,
                             arrow=_arrow_opts(arrowprops), **opts)
        return Text(self, i, entry, (2, 3, 4))

    # --- scales and tick labels ------------------------------------------------------

    def _scale(self, axis, value, kw):
        for k in ("subs", "nonpositive", "nonposx", "nonposy"):  # log clips v <= 0 by itself
            kw.pop(k, None)
        opts = {k: float(kw.pop(k)) for k in ("base", "linthresh", "linscale") if k in kw}
        if kw:
            raise TypeError(f"set_{axis}scale(): unsupported keyword(s) {sorted(kw)}")
        if not isinstance(value, str) or value not in ("linear", "log", "symlog"):
            raise ValueError(f"set_{axis}scale(): sextant draws 'linear', 'log' and 'symlog' axes, not {value!r}")
        self._call(f"set_{axis}scale", value, **opts)

    def set_xscale(self, value, **kw):
        self._scale("x", value, kw)

    def set_yscale(self, value, **kw):
        self._scale("y", value, kw)

    def get_xscale(self):
        return self.core.xscale().value

    def get_yscale(self):
        return self.core.yscale().value

    def _log_plot(self, axes, args, kw):
        scale = {k: kw.pop(k) for k in ("base", "subs", "nonpositive") if k in kw}
        for axis in axes:
            self._scale(axis, "log", dict(scale))
        return self.line(*args, **kw)

    def semilogx(self, *args, **kw):
        """plot() on a log x axis: base=, subs= and nonpositive= go to the scale."""
        return self._log_plot("x", args, kw)

    def semilogy(self, *args, **kw):
        return self._log_plot("y", args, kw)

    def loglog(self, *args, **kw):
        return self._log_plot("xy", args, kw)

    @property
    def xaxis(self):
        return _Axis(self, "x")

    @property
    def yaxis(self):
        return _Axis(self, "y")

    @property
    def spines(self):
        return _Spines(self)

    def minorticks_on(self):
        self._set_style(minor_xticks=True, minor_yticks=True)

    def minorticks_off(self):
        self._set_style(minor_xticks=False, minor_yticks=False)

    def tick_params(self, axis="both", which="major", reset=False, **kw):
        """Per axis: labelbottom/labeltop (x) and labelleft/labelright (y) show or hide
        the tick labels (sextant draws each axis' labels on one side); colors (marks and
        labels), color (marks), labelcolor, labelsize, length and width style them.
        which='minor' sets the minor marks' length (their color and width are the
        axis'); 'both' sets both lengths. Directions, padding and per-side marks are
        not in sextant."""
        del reset
        if axis not in ("x", "y", "both"):
            raise ValueError(f"tick_params(): axis must be 'x', 'y' or 'both', not {axis!r}")
        if which not in ("major", "minor", "both"):
            raise ValueError(f"tick_params(): which must be 'major', 'minor' or 'both', not {which!r}")
        axes = ("x", "y") if axis == "both" else (axis,)
        fields = {}
        for keys, field, own in ((("labelbottom", "labeltop"), "show_xtick_labels", "x"),
                                 (("labelleft", "labelright"), "show_ytick_labels", "y")):
            for k in keys:
                if k in kw:
                    value = kw.pop(k)
                    if own in axes:
                        fields[field] = bool(value)
        colors = kw.pop("colors", None)
        mark = _pop(kw, "color", default=colors)
        label = _pop(kw, "labelcolor", default=colors)
        size = kw.pop("labelsize", None)
        length = kw.pop("length", None)
        width = kw.pop("width", None)
        if kw:
            raise TypeError(f"tick_params(): unsupported keyword(s) {sorted(kw)}; sextant takes "
                            "labelbottom, labeltop, labelleft, labelright, colors, color, labelcolor, "
                            "labelsize, length, width")
        for a in axes:
            if mark is not None:
                fields[f"{a}tick_color"] = to_rgba(mark)
            if label is not None:
                fields[f"{a}label_color"] = to_rgba(label)
            if size is not None:
                fields[f"{a}label_fontsize"] = _fontsize(size)
            if width is not None:
                fields[f"{a}tick_linewidth"] = float(width) * PT
            if length is not None:
                if which in ("major", "both"):
                    fields[f"{a}tick_length"] = float(length) * PT
                if which in ("minor", "both"):
                    fields[f"minor_{a}tick_length"] = float(length) * PT
        if fields:
            self._set_style(**fields)

    def label_outer(self, remove_inner_ticks=False):
        """Hide the x tick labels unless this axes is in the grid's bottom row, and the
        y tick labels unless it is in the first column."""
        del remove_inner_ticks
        if self._span is None:
            return
        (rows, cols), first, last = self._span
        fields = {}
        if (last - 1) // cols != rows - 1:
            fields["show_xtick_labels"] = False
        if (first - 1) % cols != 0:
            fields["show_ytick_labels"] = False
        if fields:
            self._set_style(**fields)

    # --- shared axes and twins ---------------------------------------------------------

    def sharex(self, other):
        """Share the x axis with `other` (matplotlib 3.3+): one set of limits and one
        scale, panned and zoomed together; inner tick labels hide by themselves."""
        self.core.sharex(other.core)  # not in the record: cla() keeps membership
        self.figure._changed()

    def sharey(self, other):
        self.core.sharey(other.core)
        self.figure._changed()

    def _twin(self, which):
        core = getattr(self.core, f"twin{which}")()
        ax = axes_for(core)
        if ax is None:
            ax = Axes(self.figure, core)
            ax._host, ax._shared, ax._span = self, which, self._span
            self._twins[which] = ax
            self.figure.axes.append(ax)
        return ax

    def twinx(self):
        """A second axes over this one's frame, sharing the x axis, its y axis on the
        right. sextant gives one twinx per axes: a second call returns the same one.
        The cell has one title and one legend, which lists both axes' series."""
        return self._twin("x")

    def twiny(self):
        return self._twin("y")

    # --- aspect ---------------------------------------------------------------------------

    def set_aspect(self, aspect, adjustable=None, anchor=None, share=False):
        """'auto', 'equal' or a number (y unit over x unit on screen). adjustable is
        'box' (shrink the frame; the default) or 'datalim' (widen the limits)."""
        del share
        if anchor not in (None, "C"):
            raise ValueError("set_aspect(): sextant centres the frame; anchor must be 'C'")
        if isinstance(aspect, str):
            if aspect not in ("auto", "equal"):
                raise ValueError(f"set_aspect(): 'auto', 'equal' or a number, not {aspect!r}")
            ratio = 0.0 if aspect == "auto" else 1.0
        else:
            ratio = float(aspect)
        if adjustable is None:
            adjustable = self.core.adjustable().value
        if adjustable not in ("box", "datalim"):
            raise ValueError(f"set_aspect(): adjustable must be 'box' or 'datalim', not {adjustable!r}")
        self._call("set_aspect", ratio, adjustable)

    def get_aspect(self):
        ratio = self.core.aspect()
        return "auto" if ratio == 0 else ratio

    def set_adjustable(self, adjustable, share=False):
        del share
        self.set_aspect(self.core.aspect() or "auto", adjustable)

    def get_adjustable(self):
        return self.core.adjustable().value

    def axis(self, arg=None, /, *, emit=True, **kw):
        """axis('equal' | 'scaled' | 'square' | 'image' | 'auto' | 'on' | 'off'),
        axis([xmin, xmax, ymin, ymax]) or axis(xmin=..., ...). Returns the limits."""
        del emit
        if isinstance(arg, bool):
            arg = "on" if arg else "off"
        if isinstance(arg, str):
            if arg == "equal":
                self.set_aspect("equal", "datalim")
            elif arg in ("scaled", "image"):
                self.set_aspect("equal", "box")
            elif arg == "square":
                self.set_aspect("equal", "box")
                (x0, x1), (y0, y1) = self.core.xlim(), self.core.ylim()
                edge = max(abs(x1 - x0), abs(y1 - y0))
                self._call("set_xlim", x0, x0 + math.copysign(edge, x1 - x0))
                self._call("set_ylim", y0, y0 + math.copysign(edge, y1 - y0))
            elif arg == "auto":
                self.set_aspect("auto")
            elif arg in ("on", "off"):
                self._call("set_axis_visible", arg == "on")  # on a twin, its own axis only
            else:
                raise ValueError(f"axis(): {arg!r} is not one sextant has ('equal', 'scaled', 'square', "
                                 "'image', 'auto', 'on', 'off')")
        elif arg is not None:
            xmin, xmax, ymin, ymax = arg
            kw = dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, **kw)
        lims = {k: kw.pop(k) for k in ("xmin", "xmax", "ymin", "ymax") if k in kw}
        if kw:
            raise TypeError(f"axis(): unsupported keyword(s) {sorted(kw)}")
        if "xmin" in lims or "xmax" in lims:
            self.set_xlim(lims.get("xmin"), lims.get("xmax"))
        if "ymin" in lims or "ymax" in lims:
            self.set_ylim(lims.get("ymin"), lims.get("ymax"))
        return (*self.core.xlim(), *self.core.ylim())

    def set_axis_off(self):
        """Hide the axis -- spines, ticks, tick labels, axis titles, grid and background;
        the data, title and legend stay."""
        self._call("set_axis_visible", False)

    def set_axis_on(self):
        self._call("set_axis_visible", True)

    @property
    def axison(self):
        return self.core.axis_visible()


def _printf_format(fmt):
    """A tick-label format sextant takes (printf, one double) from matplotlib's: a
    FormatStrFormatter's '%.1f', or a StrMethodFormatter's '{x:.1f}' (also as a str)."""
    if "{" not in fmt:
        return fmt  # already printf
    out, conversions = [], 0
    for literal, field, spec, _ in string.Formatter().parse(fmt):
        out.append(literal.replace("%", "%%"))
        if field is None:
            continue
        if field != "x":
            raise ValueError(f"tick format {fmt!r}: sextant formats the value {{x}} only, not {{{field}}}")
        m = re.fullmatch(r"([+\- ]?)(#?)(0?)(\d*)(\.\d+)?([eEfFgGd]?)", spec or "")
        if not m:
            raise ValueError(f"tick format {fmt!r}: {spec!r} has no printf equivalent")
        sign, alt, zero, width, prec, kind = m.groups()
        if kind == "d":
            kind, prec = "f", ".0"
        out.append("%" + sign + alt + zero + width + (prec or "") + (kind or "g"))
        conversions += 1
    if conversions != 1:
        raise ValueError(f"tick format {fmt!r}: exactly one {{x}} field")
    return "".join(out)


_SPINE_DEFAULT = (0.3, 0.3, 0.3, 1.0)  # AxesStyle::spine_color
_EDGES = ("left", "right", "bottom", "top")


class _Spine:
    """ax.spines[edge]. sextant's frame is the cell's: visibility (and the shared line
    width) belong to the cell, so a twin's spines are its host's. A colour goes to the
    edge's own axis when that edge is one of a twinned cell's axes (left/bottom: the
    host's, right/top: the twin's), else to the one `spine_color` of all four edges."""

    def __init__(self, axes, edge):
        self.axes = axes
        self.edge = edge

    def _cell(self):
        return self.axes._host or self.axes

    def _colour_target(self):
        """(adapter, field) the colour of this edge lives in."""
        host = self._cell()
        e = self.edge
        axis = "y" if e in ("left", "right") else "x"
        twin = host._twins.get("x" if axis == "y" else "y")
        if twin is None:
            return host, "spine_color"
        return (host if e in ("left", "bottom") else twin), f"{axis}axis_color"

    def set_visible(self, b):
        self._cell()._set_style(**{f"spine_{self.edge}": bool(b)})

    def get_visible(self):
        return self._cell()._style.get(f"spine_{self.edge}", True)

    def set_linewidth(self, w):
        self._cell()._set_style(spine_linewidth=float(w) * PT)

    def get_linewidth(self):
        return self._cell()._style.get("spine_linewidth", 1.0) / PT

    def set_color(self, color):
        ax, field = self._colour_target()
        ax._set_style(**{field: to_rgba(color)})

    set_edgecolor = set_color

    def get_edgecolor(self):
        ax, field = self._colour_target()
        host = self._cell()
        return ax._style.get(field) or host._style.get("spine_color", _SPINE_DEFAULT)

    get_color = get_edgecolor

    def set_alpha(self, alpha):
        r, g, b, _ = self.get_edgecolor()
        self.set_color((r, g, b, float(alpha)))


class _SpineGroup:
    """ax.spines[:] / ax.spines[[...]]: a call goes to every spine in it."""

    def __init__(self, spines):
        self._spines = spines

    def __getattr__(self, name):
        if not name.startswith(("set_", "get_")):
            raise AttributeError(name)
        if name.startswith("get_"):
            return getattr(self._spines[0], name)

        def call(*args, **kw):
            for s in self._spines:
                getattr(s, name)(*args, **kw)
        return call

    def __iter__(self):
        return iter(self._spines)

    def __len__(self):
        return len(self._spines)


class _Spines:
    """ax.spines: a mapping 'left' | 'right' | 'bottom' | 'top' -> _Spine."""

    def __init__(self, axes):
        self._axes = axes

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key != slice(None):
                raise ValueError("spines: only the full slice [:] is a selection")
            return _SpineGroup([self[e] for e in _EDGES])
        if isinstance(key, (list, tuple)):
            return _SpineGroup([self[e] for e in key])
        if key not in _EDGES:
            raise KeyError(f"{key!r}: a spine is one of {_EDGES}")
        return _Spine(self._axes, key)

    def __getattr__(self, name):
        if name in _EDGES:
            return self[name]
        raise AttributeError(name)

    def __iter__(self):
        return iter(_EDGES)

    def __len__(self):
        return 4

    def __contains__(self, key):
        return key in _EDGES

    def keys(self):
        return list(_EDGES)

    def values(self):
        return [self[e] for e in _EDGES]

    def items(self):
        return [(e, self[e]) for e in _EDGES]


class _AxisLabel:
    """ax.xaxis.label: the axis title's colour and size (its text is set_xlabel's)."""

    def __init__(self, axis):
        self._axis = axis

    def _field(self, what):
        return f"{self._axis._axis}title_{what}"

    def set_color(self, color):
        self._axis.axes._set_style(**{self._field("color"): to_rgba(color)})

    def get_color(self):
        ax = self._axis.axes
        return ax._owner(self._axis._axis)._style.get(self._field("color"), (0.15, 0.15, 0.15, 1.0))

    def set_fontsize(self, size):
        self._axis.axes._set_style(**{self._field("fontsize"): _fontsize(size)})

    set_size = set_fontsize

    def get_fontsize(self):
        ax = self._axis.axes
        return ax._owner(self._axis._axis)._style.get(self._field("fontsize"), 16.5) / PT

    get_size = get_fontsize

    def get_text(self):
        return getattr(self._axis.axes.core, self._field("")[:-1])()


class _Axis:
    """ax.xaxis / ax.yaxis: the tick-label formatter, the scale, the title's colour
    and size (`.label`) and the tick parameters."""

    def __init__(self, axes, axis):
        self.axes = axes
        self._axis = axis

    @property
    def label(self):
        if self._axis not in ("x", "y"):
            raise AttributeError("label: sextant has the x and y axis titles only")
        return _AxisLabel(self)

    def set_tick_params(self, which="major", reset=False, **kw):
        self.axes.tick_params(axis=self._axis, which=which, reset=reset, **kw)

    def set_major_formatter(self, formatter):
        """A format string ('{x:.1f}', '%.1f'), a FormatStrFormatter or a
        StrMethodFormatter; the default ScalarFormatter/LogFormatter restore sextant's.
        A FuncFormatter has no equivalent: sextant formats labels in C++."""
        name = type(formatter).__name__
        if isinstance(formatter, str):
            fmt = _printf_format(formatter)
        elif name in ("FormatStrFormatter", "StrMethodFormatter"):
            fmt = _printf_format(formatter.fmt)
        elif name in ("ScalarFormatter", "LogFormatter", "LogFormatterSciNotation", "LogFormatterMathtext"):
            fmt = ""
        else:
            raise TypeError(f"set_major_formatter(): sextant takes a format string, FormatStrFormatter or "
                            f"StrMethodFormatter, not {name}")
        self.axes._owner(self._axis)._call(f"set_{self._axis}tick_format", fmt)

    def get_scale(self):
        return getattr(self.axes, f"get_{self._axis}scale")()


def _values(c, n):
    """scatter's c: per-point values to colormap, or None for a colour."""
    if c is None or isinstance(c, str):
        return None
    a = np.asarray(c)
    if a.dtype.kind in "fiu" and a.ndim == 1 and a.size == n and not (isinstance(c, tuple) and n not in (3, 4)):
        return a.astype(float)
    return None


def _area_style(method, kw, edge, step):
    """fill_between()'s and stackplot()'s shared keywords as FillOptions fields, plus
    "_linewidth" (the outline's px) for the handle, which the caller pops. matplotlib
    draws no outline unless edgecolor (or color=) is given; alpha fades it, in both.
    edge 'face' is each fill's own colour."""
    _drop_ignored(kw)
    alpha = kw.pop("alpha", None)
    lw = _pop(kw, "linewidth", "linewidths", "lw")
    ls = _pop(kw, "linestyle", "linestyles", "ls")
    opts = dict(kw)
    linewidth = (1.0 if lw is None else float(np.ravel(lw)[0])) * PT
    if alpha is not None:
        opts["alpha"] = float(alpha)
    if edge is not None and not (isinstance(edge, str) and edge.lower() == "none") and linewidth > 0:
        if not (isinstance(edge, str) and edge == "face"):
            opts["edgecolor"] = to_rgba(edge)
        opts["linewidth"] = linewidth
    if ls is not None:
        opts["linestyle"] = _linestyle(ls)
    if step is not None:
        if step not in ("pre", "post", "mid"):
            raise ValueError(f"{method}(): step must be 'pre', 'post' or 'mid', not {step!r}")
        opts["drawstyle"] = "steps-" + step
    opts["_linewidth"] = linewidth
    return opts


def _exclude(where, n, method):
    """matplotlib's `where` as sextant's exclude mask (its inverse); None for none."""
    if where is None:
        return None
    w = np.asarray(where, dtype=bool).ravel()
    if w.size != n:
        raise ValueError(f"{method}(): where has {w.size} values, x has {n}")
    return ~w if not w.all() else None


def _quiver_args(args):
    """quiver([X, Y], U, V, [C]) as flat x, y, u, v and c (None without C), as
    matplotlib reads them: no X, Y is U's index grid; 1-D X, Y spanning a 2-D U's
    columns and rows are a meshgrid."""
    if len(args) not in (2, 3, 4, 5):
        raise TypeError(f"quiver(): quiver([X, Y], U, V, [C]) takes 2 to 5 arrays, got {len(args)}")
    xy, rest = (args[:2], args[2:]) if len(args) >= 4 else (None, args)
    U = np.asarray(rest[0], dtype=float)
    V = np.asarray(rest[1], dtype=float)
    nr, nc = (1, U.size) if U.ndim < 2 else U.shape[:2]
    if xy is None:
        X, Y = np.meshgrid(np.arange(nc, dtype=float), np.arange(nr, dtype=float))
    else:
        X, Y = (np.asarray(a, dtype=float).ravel() for a in xy)
        if len(X) == nc and len(Y) == nr:
            X, Y = np.meshgrid(X, Y)
    x, y = X.ravel(), Y.ravel()
    u = np.broadcast_to(U.ravel(), x.shape).astype(float) if U.size == 1 else U.ravel()
    v = np.broadcast_to(V.ravel(), x.shape).astype(float) if V.size == 1 else V.ravel()
    c = None if len(rest) < 3 or rest[2] is None else _vec(rest[2])
    return x, y, u, v, c


def _quiver_amp(x, y, u, v, scale, units):
    """sextant's amp (data units of arrow per unit of vector) for matplotlib's scale in
    `units`. Unset scale: sextant's own rule, matplotlib's mean arrow length with the
    anchors' extent standing for the view's, computed here so that a replay after
    set_UVC() keeps it."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(u) & np.isfinite(v)
    xe = float(np.ptp(x[ok])) if ok.any() else 0.0
    ye = float(np.ptp(y[ok])) if ok.any() else 0.0
    wide = xe if xe > 0 else ye if ye > 0 else 1.0
    high = ye if ye > 0 else xe if xe > 0 else 1.0
    if scale is None:
        lengths = np.hypot(u[ok], v[ok])
        lengths = lengths[(lengths > 0) & np.isfinite(lengths)]
        if not lengths.size:
            return 1.0
        return wide / (1.8 * max(10.0, math.sqrt(int(ok.sum()))) * float(lengths.mean()))
    per = {"xy": 1.0, "x": 1.0, "y": 1.0, "width": wide, "height": high}.get(units)
    if per is None:
        raise ValueError(f"quiver(): sextant draws arrows in data units; a scale in units {units!r} needs "
                         "scale_units 'xy', 'x', 'y', 'width' or 'height'")
    scale = float(scale)
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError(f"quiver(): scale must be finite and > 0, not {scale!r}")
    return per / scale


def _plot_groups(args, data):
    """plot([x], y, [fmt], [x2], y2, [fmt2], ...) -> [(x, y, fmt)]."""
    args = list(args)
    if data is not None:
        args = [data[a] if isinstance(a, str) and a in data else a for a in args]
    out = []
    while args:
        first = args.pop(0)
        if isinstance(first, str):
            raise ValueError(f"plot(): format string {first!r} without data")
        if args and not isinstance(args[0], str):
            x, y = first, args.pop(0)
        else:
            x, y = None, first
        fmt = args.pop(0) if args and isinstance(args[0], str) else ""
        out.append((x, y, fmt))
    return out


class Axes3D(_AxesBase):
    """A 3D axes, matplotlib's mplot3d style, over a core sextant.Axes3D (``.core``)."""

    _SERIES = {"line3d", "scatter3d", "surface", "surface_tri", "bar3d", "quiver3d"}
    _GRID_ON = True

    @_takes_visible
    def line(self, xs, ys, zs=0, *args, zdir="z", **kw):
        """line(xs, ys, zs=0, [fmt], **kwargs): mplot3d's plot(), named after sextant's
        line3d(); plot is an alias. Returns a list of one Path3D."""
        _drop_ignored(kw)
        if zdir != "z":
            raise ValueError("Axes3D.plot(): only zdir='z'")
        fmt = args[0] if args and isinstance(args[0], str) else _pop(kw, "fmt", default="")
        ls, mk, fc = parse_fmt(fmt)
        xs = _vec(xs)
        ys = _vec(ys)
        zs = np.broadcast_to(np.asarray(zs, dtype=float), xs.shape).astype(float)
        color = to_rgba(_pop(kw, "color", "c", default=fc)) or self._next_color()
        label = kw.pop("label", None)
        lw = _pop(kw, "linewidth", "lw")
        opts = dict(kw, color=color, linewidth=(lw if lw is not None else 1.5) * PT)
        kw.pop("linestyle", None), opts.pop("linestyle", None), opts.pop("ls", None)
        if label is not None:
            opts["name"] = str(label)
        if mk is not None and ls is None:
            opts.pop("linewidth")
            opts.update(marker=_marker(mk), size=6.0 * PT)
            i, entry = self._add("scatter3d", "scatter3d", xs, ys, zs, **opts)
            return [Path3D(self, "scatter3d", i, entry, label)]
        i, entry = self._add("line3d", "line3d", xs, ys, zs, **opts)
        return [Path3D(self, "line3d", i, entry, label)]

    plot = line

    @_takes_visible
    def scatter(self, xs, ys, zs=0, zdir="z", s=None, c=None, depthshade=True, marker=None, cmap=None,
                vmin=None, vmax=None, alpha=None, label=None, colorbar=False, **kw):
        _drop_ignored(kw)
        for k in ("linewidths", "edgecolors", "norm"):
            kw.pop(k, None)
        xs, ys = _vec(xs), _vec(ys)
        zs = np.broadcast_to(np.asarray(zs, dtype=float), xs.shape).astype(float)
        opts = dict(kw, **Axes._marker_size(s, len(xs)), depthshade=0.4 if depthshade else 0.0)
        if alpha is not None:
            opts["alpha"] = alpha
        if marker is not None:
            opts["marker"] = _marker(marker)
        if label is not None:
            opts["name"] = str(label)
        color = _pop(opts, "color")
        values = _values(c, len(xs))
        if values is not None:
            lo, hi = _auto_limits(values, vmin, vmax)
            opts.update(vmin=lo, vmax=hi, colorbar=bool(colorbar))
            cmap = _rc_cmap(cmap)
            if cmap is not None:
                opts["cmap"] = cmap
            i, entry = self._add("scatter3d", "scatter3d", xs, ys, zs, colors=values, **opts)
        else:
            c = c if c is not None else color
            opts["color"] = to_rgba(c) if c is not None else self._next_color(patches=True)
            i, entry = self._add("scatter3d", "scatter3d", xs, ys, zs, **opts)
        handle = Path3D(self, "scatter3d", i, entry, label)
        if values is not None:
            self.figure._mappable = handle
        return handle

    def _surface_opts(self, cmap, color, alpha, edgecolor, linewidth, label, vmin, vmax, kw):
        _drop_ignored(kw)
        kw.pop("shade", None), kw.pop("norm", None), kw.pop("facecolors", None)
        opts = dict(kw)
        if alpha is not None:
            opts["alpha"] = alpha
        if edgecolor is not None and to_rgba(edgecolor) != (0.0, 0.0, 0.0, 0.0):
            opts.update(edges=True, edgecolor=to_rgba(edgecolor))
        if linewidth is not None:
            opts["edge_linewidth"] = float(linewidth) * PT
        if label is not None:
            opts["name"] = str(label)
        if cmap is None:
            opts["color"] = to_rgba(color) if color is not None else self._next_color(patches=True)
        else:
            opts["cmap"] = cmap
            if vmin is not None and vmax is not None:  # else the surface's own range
                opts.update(vmin=vmin, vmax=vmax)
        return opts

    @_takes_visible
    def plot_surface(self, X, Y, Z, cmap=None, color=None, alpha=None, edgecolor=None, linewidth=None,
                     label=None, vmin=None, vmax=None, **kw):
        opts = self._surface_opts(cmap, color, alpha, edgecolor, linewidth, label, vmin, vmax, kw)
        X, Y, Z = (np.asarray(a, dtype=float) for a in (X, Y, Z))
        grid = _rectilinear(X, Y, Z)
        if grid is not None:
            u, v, heights = grid
            if cmap is not None:
                opts["colormap"] = True
            i, entry = self._add("surface", "surface", "xy", u, v, heights, **opts)
            handle = Surface3D(self, "surface", i, entry, label)
        else:
            # A parametric surface (a sphere, a torus): its quad mesh as triangles.
            r, c = Z.shape
            idx = np.arange(r * c).reshape(r, c)
            a, b, d, e = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
            tri = np.concatenate([np.stack([a, b, e], -1).reshape(-1, 3), np.stack([a, e, d], -1).reshape(-1, 3)])
            colors = Z.ravel() if cmap is not None else None
            i, entry = self._add("surface_tri", "surface_tri", X.ravel(), Y.ravel(), Z.ravel(), tri,
                                 colors=colors, **opts)
            handle = Surface3D(self, "surface_tri", i, entry, label)
        if cmap is not None:
            self.figure._mappable = handle
        return handle

    @_takes_visible
    def plot_wireframe(self, X, Y, Z, color=None, linewidth=None, label=None, **kw):
        color = to_rgba(_pop(kw, "colors", default=color)) or self._next_color()
        # The faces invisible (their colour's alpha 0; `alpha` would hide the wires too).
        return self.plot_surface(X, Y, Z, color=color[:3] + (0.0,), edgecolor=color,
                                 linewidth=linewidth if linewidth is not None else 1.0, label=label, **kw)

    @_takes_visible
    def plot_trisurf(self, x, y, z, triangles=None, cmap=None, color=None, alpha=None, edgecolor=None,
                     linewidth=None, label=None, vmin=None, vmax=None, **kw):
        opts = self._surface_opts(cmap, color, alpha, edgecolor, linewidth, label, vmin, vmax, kw)
        x, y, z = _vec(x), _vec(y), _vec(z)
        colors = z if cmap is not None else None
        if triangles is None:
            i, entry = self._add("surface_tri", "surface_tri", x, y, z, orient="xy", colors=colors, **opts)
        else:
            i, entry = self._add("surface_tri", "surface_tri", x, y, z, np.asarray(triangles), colors=colors, **opts)
        handle = Surface3D(self, "surface_tri", i, entry, label)
        if cmap is not None:
            self.figure._mappable = handle
        return handle

    @_takes_visible
    def quiver(self, X, Y, Z, U, V, W, *, length=1.0, arrow_length_ratio=None, pivot="tail", normalize=False,
               color=None, colors=None, alpha=None, label=None, **kw):
        """mplot3d's quiver(X, Y, Z, U, V, W): an arrow at (x, y, z) along (u, v, w) times
        length (normalize=True: unit vectors first). arrow_length_ratio sizes the heads (in pixels,
        approximated from the mean arrow's on-screen length). Returns a Quiver3D."""
        _drop_ignored(kw)
        x, y, z, u, v, w = (_vec(a) for a in (X, Y, Z, U, V, W))
        if normalize:
            n = np.sqrt(u * u + v * v + w * w)
            n[n == 0] = 1.0
            u, v, w = u / n, v / n, w / n
        c = color if color is not None else colors
        opts = dict(kw, amp=float(length), pivot=pivot,
                    color=to_rgba(c) if c is not None else self._next_color(patches=True))
        if alpha is not None:
            opts["alpha"] = float(alpha)
        if arrow_length_ratio is not None:
            # sextant sizes heads in pixels: ratio times the mean arrow's length on screen, which
            # is taken as the arrow's share of the longest side of the scene in ~55% of the frame.
            mean = float(np.mean(np.sqrt(u * u + v * v + w * w))) * float(length)
            ends = [np.concatenate([p, p + float(length) * d]) for p, d in ((x, u), (y, v), (z, w))]
            span = max(float(np.ptp(e)) for e in ends) if len(x) else 0.0
            if span > 0 and mean > 0:
                W, H = self.figure._size
                head = max(2.0, float(arrow_length_ratio) * mean / span * 0.55 * min(W, H))
                opts["head_length"], opts["head_width"] = head, head * 0.9
        if label is not None:
            opts["name"] = str(label)
        i, entry = self._add("quiver3d", "quiver3d", x, y, z, u, v, w, **opts)
        return Quiver3D(self, "quiver3d", i, entry, label)

    @_takes_visible
    def bar3d(self, x, y, z, dx, dy, dz, color=None, alpha=None, shade=True, edgecolor=None, label=None, **kw):
        _drop_ignored(kw)
        n = len(np.atleast_1d(x))
        x, y, z, dx, dy, dz = (np.broadcast_to(np.asarray(a, dtype=float), (n,)) for a in (x, y, z, dx, dy, dz))
        cx, cy = x + dx / 2, y + dy / 2
        u, v = np.unique(cx), np.unique(cy)
        if len(u) * len(v) != n:
            raise ValueError("bar3d(): sextant draws bars on a grid; these positions do not form one "
                             "(every x with every y, once)")
        iu, iv = np.searchsorted(u, cx), np.searchsorted(v, cy)
        heights = np.full((len(u), len(v)), np.nan)
        bottoms = np.zeros_like(heights)
        heights[iu, iv], bottoms[iu, iv] = dz, z
        if np.isnan(heights).any():
            raise ValueError("bar3d(): a grid position appears twice")
        su = float(np.min(np.diff(u))) if len(u) > 1 else float(dx[0])
        sv = float(np.min(np.diff(v))) if len(v) > 1 else float(dy[0])
        opts = dict(kw, width=float(np.mean(dx)) / su, depth=float(np.mean(dy)) / sv,
                    shading=0.45 if shade else 0.0,
                    color=to_rgba(color) if color is not None else self._next_color(patches=True))
        if alpha is not None:
            opts["alpha"] = alpha
        if edgecolor is not None:
            opts.update(edges=True, edgecolor=to_rgba(edgecolor))
        if label is not None:
            opts["name"] = str(label)
        bottoms_arg = bottoms if np.any(bottoms != 0) else None
        i, entry = self._add("bar3d", "bar3d", "xy", u, v, heights, bottoms=bottoms_arg, **opts)
        return BarContainer(self, "bar3d", i, entry, label)

    # --- text ---

    @_takes_visible
    def text(self, x, y, z, s, zdir=None, **kw):
        """Text at the data point (x, y, z), at a fixed pixel size. zdir is not
        offered (sextant's 3D text always faces the viewer)."""
        if zdir is not None:
            raise ValueError("text(): sextant's 3D text always faces the viewer; zdir is not offered")
        opts = _text_opts(kw, kw.pop("fontdict", None))
        i, entry = self._add("text", "text", str(s), float(x), float(y), float(z), **opts)
        return Text(self, i, entry, (0, 1, 2, 3))

    @_takes_visible
    def text2D(self, x, y, s, fontdict=None, *, transform=None, **kw):
        """Text at (x, y), fractions of the axes, whatever the camera does."""
        if self._coords(transform, "text2D") not in (None, "fraction"):
            raise ValueError("text2D(): sextant places 2D text in ax.transAxes only")
        opts = _text_opts(kw, fontdict)
        i, entry = self._add("text", "text2d", str(s), float(x), float(y), **opts)
        return Text(self, i, entry, (0, 1, 2))

    # --- decoration ---

    def set_zlabel(self, zlabel, fontsize=None, **kw):
        self._title("set_ztitle", zlabel, fontsize, kw)

    def get_zlabel(self):
        return self.core.ztitle()

    def set_zlim(self, bottom=None, top=None, **kw):
        return self._lim("z", bottom, top, kw)

    def get_zlim(self):
        return self.core.zlim()

    def set_zticks(self, ticks, labels=None, **kw):
        self._ticks("z", ticks, labels, kw)

    def view_init(self, elev=None, azim=None, roll=None, vertical_axis="z"):
        if roll not in (None, 0) or vertical_axis != "z":
            raise ValueError("view_init(): sextant's camera has no roll and z is up")
        cam = self.core.camera()
        self._call("set_view", cam["azimuth"] if azim is None else float(azim),
                   cam["elevation"] if elev is None else float(elev))

    def set_proj_type(self, proj_type, focal_length=None):
        mode = {"persp": "perspective", "ortho": "orthographic"}.get(proj_type)
        if mode is None:
            raise ValueError("set_proj_type(): 'persp' or 'ortho'")
        self._call("set_projection", mode)
        if focal_length is not None:
            self._call("set_fov", math.degrees(2 * math.atan(1 / float(focal_length))))

    def set_box_aspect(self, aspect, zoom=1):
        if aspect is None:
            aspect = (4, 4, 3)  # matplotlib's default box
        self._call("set_box_aspect", tuple(float(a) for a in aspect))


def _rectilinear(X, Y, Z):
    """plot_surface's X, Y, Z as sextant's (u, v, heights (len(u), len(v))) when they
    are a grid -- meshgrid with 'xy' or 'ij' indexing, or 1-D axes -- else None."""
    if X.ndim == 1 and Y.ndim == 1 and Z.shape == (len(Y), len(X)):
        return X, Y, Z.T
    if X.shape != Z.shape or Y.shape != Z.shape or Z.ndim != 2 or min(Z.shape) < 2:
        return None
    if np.all(X == X[0:1, :]) and np.all(Y == Y[:, 0:1]):  # 'xy': x along columns
        u, v, h = X[0], Y[:, 0], Z.T
    elif np.all(X == X[:, 0:1]) and np.all(Y == Y[0:1, :]):  # 'ij': x along rows
        u, v, h = X[:, 0], Y[0], Z
    else:
        return None
    if np.all(np.diff(u) > 0) and np.all(np.diff(v) > 0):
        return u, v, h
    return None


_THETA_DIRECTION = {"counterclockwise": 1, "anticlockwise": 1, "clockwise": -1}


class PolarAxes(_AxesBase):
    """A polar axes, matplotlib's projection='polar', over a core sextant.PolarAxes
    (``.core``). theta is in radians where matplotlib's is (the data, set_xlim(),
    set_xticks(), set_theta_offset()) and in degrees where its is (set_thetagrids(),
    set_thetamin(), set_rlabel_position(), the zero location's offset)."""

    _SERIES = {"line", "scatter", "scatter_z", "bar", "fill"}
    _GRID_ON = True
    _LINE = "plot"

    # The 2D plotting calls, as Axes's: the core's take (theta, r) where Axes's take (x, y).
    line = plot = Axes.line
    step = Axes.step
    _line = Axes._line
    scatter = Axes.scatter
    _marker_size = staticmethod(Axes._marker_size)
    _bar_series = Axes._bar_series
    _err = staticmethod(Axes._err)
    fill_between = Axes.fill_between
    _fill_between = Axes._fill_between
    text = Axes.text
    annotate = Axes.annotate
    _c0 = Axes._c0

    @_zorder(1)
    @_takes_visible
    def bar(self, x, height, width=0.8, bottom=None, *, align="center", **kw):
        """Ring sectors at the angles x (radians), `width` radians wide, from `bottom`
        out by `height`. No error bars."""
        if kw.get("yerr") is not None or kw.get("xerr") is not None:
            raise ValueError("bar(): sextant draws no error bars on polar bars")
        return self._bar_series("bar", x, height, width, bottom, align, "vertical", **kw)

    @_zorder(1)
    @_takes_visible
    def fill(self, *args, data=None, **kw):
        """matplotlib's fill(theta, r, [color], ...): a filled polygon per group, its edges
        straight in (theta, r). Returns a list of Polygon."""
        out = []
        for t, r, fmt in _plot_groups(args, data):
            if t is None:
                raise ValueError("fill(): give both theta and r")
            pkw = dict(kw)
            if fmt:
                pkw.setdefault("color", fmt)
            if not any(k in pkw for k in ("color", "c", "facecolor", "fc")):
                pkw["facecolor"] = self._next_color(patches=True)
            out.append(Polygon(np.column_stack([_vec(t), _vec(r)]), **pkw)._add_to(self))
        return out

    def errorbar(self, *args, **kw):
        raise ValueError("errorbar(): sextant draws no error bars on a polar axes")

    # --- titles ---

    def set_xlabel(self, *args, **kw):
        raise ValueError("set_xlabel(): sextant's polar axes have no axis titles")

    set_ylabel = set_xlabel

    # --- r ---

    def set_rlim(self, bottom=None, top=None, **kw):
        """The r limits (bottom at the centre); rmin=/rmax= too, as matplotlib's."""
        bottom = _pop(kw, "rmin", default=bottom)
        top = _pop(kw, "rmax", default=top)
        return self._lim("r", bottom, top, kw)

    set_ylim = set_rlim

    def get_ylim(self):
        return self.core.rlim()

    def set_rmin(self, rmin):
        self._call("set_rmin", float(rmin))

    def set_rmax(self, rmax):
        self._call("set_rmax", float(rmax))

    def get_rmin(self):
        return self.core.rlim()[0]

    def get_rmax(self):
        return self.core.rlim()[1]

    def invert_yaxis(self):
        lo, hi = self.core.rlim()
        self._call("set_rlim", hi, lo)

    def set_rorigin(self, rorigin):
        raise ValueError("set_rorigin(): sextant draws r from the centre")

    def set_rscale(self, value, **kw):
        if value != "linear":
            raise ValueError(f"set_rscale(): sextant's polar r axis is linear, not {value!r}")

    set_yscale = set_rscale

    def get_rscale(self):
        return "linear"

    get_yscale = get_rscale

    @property
    def yaxis(self):
        return _Axis(self, "r")

    def set_rlabel_position(self, value):
        """The angle (degrees) the r labels sit along."""
        self._call("set_rlabel_position", float(value))

    def get_rlabel_position(self):
        return self.core.rlabel_position()

    def _ticks(self, axis, ticks, labels, kw):
        kw.pop("minor", None)
        if kw:
            raise TypeError(f"set_{axis}ticks(): unsupported keyword(s) {sorted(kw)}")
        ticks = _vec(ticks)
        setattr(self, f"_{axis}ticks", ticks)
        setattr(self, f"_{axis}ticklabels", [str(s) for s in labels] if labels is not None else None)
        labels = [str(s) for s in labels] if labels is not None else []
        if axis == "x":  # theta: radians here, degrees in the core
            self._call("set_thetagrids", np.degrees(ticks), labels)
        else:
            self._call("set_rticks", ticks, labels)
        # An empty list hides the labels, as matplotlib's (sextant's [] means automatic).
        show = bool(len(ticks))
        if self._style.get(f"show_{axis}ticks", True) != show:
            self._set_style(**{f"show_{axis}ticks": show})

    set_rticks = _AxesBase.set_yticks

    def set_rgrids(self, radii, labels=None, angle=None, fmt=None, **kw):
        """Rings at `radii` labelled `labels`, the labels along `angle` (degrees), `fmt`
        formatting them."""
        self.set_yticks(radii, labels, **kw)
        if angle is not None:
            self.set_rlabel_position(angle)
        if fmt is not None:
            self.yaxis.set_major_formatter(fmt)

    def set_thetagrids(self, angles, labels=None, fmt=None, **kw):
        """Spokes at `angles` (degrees) labelled `labels`, or each angle through `fmt`
        ('%d°')."""
        angles = _vec(angles)
        if labels is None and fmt is not None:
            labels = [fmt % a for a in angles]
        self.set_xticks(np.radians(angles), labels, **kw)

    # --- theta ---

    def set_thetalim(self, *args, **kw):
        """set_thetalim(min, max) in radians, or thetamin=/thetamax= in degrees, as
        matplotlib's. Returns the span in radians."""
        lo, hi = self.core.thetalim()
        if len(args) == 1:
            args = tuple(args[0])
        if len(args) == 2:
            lo = lo if args[0] is None else math.degrees(float(args[0]))
            hi = hi if args[1] is None else math.degrees(float(args[1]))
        elif args:
            raise TypeError("set_thetalim(): (min, max) in radians")
        lo = float(kw.pop("thetamin", lo))
        hi = float(kw.pop("thetamax", hi))
        if kw:
            raise TypeError(f"set_thetalim(): unsupported keyword(s) {sorted(kw)}")
        self._call("set_thetalim", lo, hi)
        return math.radians(lo), math.radians(hi)

    def set_xlim(self, left=None, right=None, **kw):
        left = _pop(kw, "left", "xmin", default=left)
        right = _pop(kw, "right", "xmax", default=right)
        kw.pop("emit", None), kw.pop("auto", None)
        if left is not None and right is None and np.ndim(left) == 1:
            left, right = left
        return self.set_thetalim(left, right, **kw)

    def get_xlim(self):
        return tuple(math.radians(v) for v in self.core.thetalim())

    def set_thetamin(self, thetamin):
        self._call("set_thetamin", float(thetamin))

    def set_thetamax(self, thetamax):
        self._call("set_thetamax", float(thetamax))

    def get_thetamin(self):
        return self.core.thetalim()[0]

    def get_thetamax(self):
        return self.core.thetalim()[1]

    def set_theta_zero_location(self, loc, offset=0.0):
        """theta = 0 at 'N', 'NW', 'W', 'SW', 'S', 'SE', 'E' or 'NE', turned a further
        `offset` degrees counterclockwise."""
        self._call("set_theta_zero_location", str(loc), float(offset))

    def set_theta_offset(self, offset):
        """The screen angle of theta = 0, radians counterclockwise from east."""
        self._call("set_theta_offset", math.degrees(float(offset)))

    def get_theta_offset(self):
        return math.radians(self.core.theta_offset())

    def set_theta_direction(self, direction):
        """1 or 'counterclockwise', -1 or 'clockwise'."""
        if isinstance(direction, str):
            if direction.lower() not in _THETA_DIRECTION:
                raise ValueError(f"set_theta_direction(): {direction!r} is not 'clockwise' or 'counterclockwise'")
            direction = _THETA_DIRECTION[direction.lower()]
        self._call("set_theta_direction", int(direction))

    def get_theta_direction(self):
        return self.core.theta_direction()


# --- figure ---------------------------------------------------------------------------

class MplEvent:
    """A sextant Event in matplotlib's shape, as fig.canvas.mpl_connect() delivers it.
    The sextant event is ``.sextant_event``."""

    def __init__(self, figure, name, e):
        self.name = name
        self.canvas = figure.canvas
        self.guiEvent = None
        self.sextant_event = e
        self.x = e.x
        self.y = figure._size[1] - e.y  # matplotlib counts from the bottom
        host = axes_for(e.inaxes)
        self.inaxes = host
        self.xdata = e.xdata if e.has_data else None
        self.ydata = e.ydata if e.has_data else None
        twins = getattr(host, "_twins", None)
        if twins:
            # matplotlib reports the topmost axes, the last twin made, in its own
            # coordinates; sextant reports the cell's host, with the twins' as x2/y2.
            which, self.inaxes = list(twins.items())[-1]
            if e.has_data:
                if which == "x":
                    self.ydata = e.y2data
                else:
                    self.xdata = e.x2data
        self.key = e.key or None
        self.dblclick = e.double_click
        self.button = MOUSE_BUTTONS.get(e.button) if e.kind in ("mouse_down", "mouse_up", "pick") else None
        self.step = 0.0
        if e.kind == "scroll":
            self.step = e.scroll_y
            self.button = "up" if e.scroll_y > 0 else "down"
        if e.kind == "resize":
            self.width, self.height = e.width, e.height
        if e.kind == "pick":
            # The object's own axes: a twin's objects are its own. The event names it by id;
            # the artists are keyed by (kind, index), which survives a log replay (new ids).
            ax = host._twins.get({1: "x", 2: "y"}[e.pick_twin]) if twins and e.pick_twin else host
            loc = ax.core.find_object(e.pick_object) if ax and e.pick_plane == 0 else None
            self.artist = ax._artists.get((_PICK_KINDS.get(str(loc[0]), str(loc[0])), loc[1])) if loc else None
            if isinstance(self.artist, _Pie):  # a press picks one slice
                self.artist = self.artist.wedges[e.pick_index]
            self.ind = np.array([e.pick_index])
            self.mouseevent = self


class Canvas:
    """fig.canvas: redraw, event delivery and mpl_connect()."""

    def __init__(self, figure):
        self.figure = figure

    def draw(self):
        self.figure._refresh()

    draw_idle = draw

    def flush_events(self):
        self.figure._refresh()
        try:
            _sextant.poll_events()
        except RuntimeError:  # macOS, off the main thread
            self.figure.core.dispatch_events()

    def mpl_connect(self, name, fn):
        fig = self.figure
        return fig.core.connect(name, lambda e: fn(MplEvent(fig, name, e)))

    def mpl_disconnect(self, cid):
        self.figure.core.disconnect(cid)

    def get_width_height(self):
        return self.figure._size


class Figure:
    """A figure, matplotlib style, over a core sextant.Figure (``.core``)."""

    def __init__(self, figsize=None, dpi=None, num=None, **kw):
        rc = _rc.rcParams
        w, h = figsize if figsize is not None else rc["figure.figsize"]
        self._dpi = float(dpi) if dpi is not None else float(rc["figure.dpi"])
        self._size = (int(round(w * 100)), int(round(h * 100)))
        title = kw.pop("title", f"Figure {num}" if num is not None else "sextant")
        facecolor = kw.pop("facecolor", None)
        if facecolor is None and rc.changed("figure.facecolor"):
            facecolor = rc["figure.facecolor"]
        if rc.changed("mathtext.fontset"):
            kw.setdefault("math_fontset", rc["mathtext.fontset"])
        if facecolor is not None:
            kw["background"] = to_rgba(facecolor)
        self.core = _sextant.Figure(width=self._size[0], height=self._size[1], dpi=96.0 * self._dpi / 100.0,
                                    title=str(title), **kw)
        self.number = num
        self.axes = []
        self.canvas = Canvas(self)
        self._grid = None
        self._shown = False
        self._background = to_rgba(facecolor) if facecolor is not None else (0.93, 0.93, 0.93, 1.0)
        self._mappable = None  # the last colormapped artist: colorbar()'s default
        self._interactive = lambda: False  # set by pyplot

    # --- subplots ---

    def add_subplot(self, *args, projection=None, polar=False, **kw):
        _drop_ignored(kw)
        kw.pop("label", None)
        facecolor = kw.pop("facecolor", None)
        sharex, sharey = kw.pop("sharex", None), kw.pop("sharey", None)
        if kw:
            raise TypeError(f"add_subplot(): unsupported keyword(s) {sorted(kw)}")
        if polar:
            if projection not in (None, "polar"):
                raise ValueError(f"add_subplot(): polar=True and projection={projection!r}")
            projection = "polar"
        if projection in ("3d", "polar") and (sharex is not None or sharey is not None):
            raise ValueError("add_subplot(): sextant shares 2D axes only")
        if not args:
            args = (1, 1, 1)
        elif len(args) == 1:
            pos = args[0]
            if isinstance(pos, int) and 111 <= pos <= 999:
                args = (pos // 100, pos // 10 % 10, pos % 10)
            else:
                raise TypeError("add_subplot(): a three-digit int, or (nrows, ncols, index)")
        nrows, ncols, index = args
        slot = self._slot(int(nrows), int(ncols), index)
        if projection in (None, "rectilinear"):
            core = self.core.add_subplot(*self._grid, slot)
            cls = Axes
        elif projection == "3d":
            core = self.core.add_subplot3d(*self._grid, slot)
            cls = Axes3D
        elif projection == "polar":
            core = self.core.add_subplot_polar(*self._grid, slot)
            cls = PolarAxes
        else:
            raise ValueError(f"add_subplot(): projection={projection!r} is not supported (None, '3d' or 'polar')")
        ax = axes_for(core)
        if ax is None:
            ax = cls(self, core)
            self.axes.append(ax)
        if cls is Axes:
            first, last = (slot, slot) if isinstance(slot, int) else slot
            ax._span = (self._grid, first, last)
        if sharex is not None:
            ax.sharex(sharex)
        if sharey is not None:
            ax.sharey(sharey)
        if facecolor is not None and cls is not Axes3D:
            ax.set_facecolor(facecolor)
        return ax

    def _slot(self, nrows, ncols, index):
        """(nrows, ncols, index) on the grid this figure's first subplot fixed:
        sextant has one grid per figure, so a grid dividing it maps to a span."""
        if self._grid is None:
            self._grid = (nrows, ncols)
        R, C = self._grid
        first, last = (index, index) if isinstance(index, int) else tuple(index)
        if (nrows, ncols) == (R, C):
            return first if first == last else (first, last)
        if R % nrows or C % ncols:
            raise ValueError(f"this figure's subplot grid is {R}x{C} (fixed by its first subplot); "
                             f"a {nrows}x{ncols} position does not fit it")
        fr, fc = R // nrows, C // ncols
        r0, c0 = divmod(first - 1, ncols)
        r1, c1 = divmod(last - 1, ncols)
        a = r0 * fr * C + c0 * fc + 1
        b = ((r1 + 1) * fr - 1) * C + (c1 + 1) * fc
        return a if a == b else (a, b)

    def subplots(self, nrows=1, ncols=1, *, sharex=False, sharey=False, squeeze=True, subplot_kw=None,
                 width_ratios=None, height_ratios=None):
        kw = dict(subplot_kw or {})
        if (kw.get("projection") in ("3d", "polar") or kw.get("polar")) and any(
                how not in (False, None, "none") for how in (sharex, sharey)):
            raise ValueError("subplots(): sextant shares 2D axes only")
        axs = np.empty((nrows, ncols), dtype=object)
        for r in range(nrows):
            for c in range(ncols):
                axs[r, c] = self.add_subplot(nrows, ncols, r * ncols + c + 1, **kw)
        # matplotlib's groups; the inner tick labels hide by themselves, as there.
        for axis, how in (("x", sharex), ("y", sharey)):
            if how is True:
                how = "all"
            if how in (False, None, "none"):
                continue
            ref = {"all": lambda r, c: axs[0, 0], "col": lambda r, c: axs[0, c],
                   "row": lambda r, c: axs[r, 0]}.get(how)
            if ref is None:
                raise ValueError(f"subplots(): share{axis} must be True, False, 'all', 'col', 'row' or 'none', "
                                 f"not {how!r}")
            for r in range(nrows):
                for c in range(ncols):
                    if ref(r, c) is not axs[r, c]:
                        getattr(axs[r, c], f"share{axis}")(ref(r, c))
        if width_ratios is not None:
            self.core.set_col_ratios(list(width_ratios))
        if height_ratios is not None:
            self.core.set_row_ratios(list(height_ratios))
        if squeeze:
            return axs.item() if axs.size == 1 else axs.squeeze()
        return axs

    def gca(self):
        return self.axes[-1] if self.axes else self.add_subplot(1, 1, 1)

    def get_children(self):
        return list(self.axes)

    # --- decoration and output ---

    def suptitle(self, t, fontsize=None, **kw):
        _drop_ignored(kw)
        _no_parse_math(kw, "suptitle")
        if fontsize is None:
            self.core.suptitle(t)
        else:
            self.core.suptitle(t, float(fontsize) * PT)
        self._changed()

    def colorbar(self, mappable=None, ax=None, label="", orientation="vertical", **kw):
        mappable = mappable if mappable is not None else self._mappable
        if mappable is None:
            raise RuntimeError("colorbar(): no colormapped object (imshow, scatter with c=, a surface with cmap)")
        mappable._entry[2]["colorbar"] = True
        if label:
            mappable._entry[2]["name"] = str(label)
        axes = mappable.axes
        axes._log.append(["set_colorbar_style", [], {"anchor": "bottom" if orientation == "horizontal" else "right"}])
        axes._rebuild()
        return Colorbar(mappable, label, orientation)

    def set_facecolor(self, color):
        self._background = to_rgba(color)
        self.core.set_background(self._background)
        self._changed()

    def get_facecolor(self):
        return self._background

    def savefig(self, fname, dpi=None, format=None, **kw):
        for k in ("bbox_inches", "pad_inches", "edgecolor", "metadata", "pil_kwargs", "backend",
                  "orientation", "papertype"):
            kw.pop(k, None)
        # facecolor= / transparent= recolor the figure background for this file only.
        facecolor, transparent = kw.pop("facecolor", "auto"), kw.pop("transparent", None)
        if transparent:
            self.core.set_background((0.0, 0.0, 0.0, 0.0))
        elif facecolor not in (None, "auto"):
            self.core.set_background(to_rgba(facecolor))
        try:
            return self._savefig(fname, dpi, format, kw)
        finally:
            if transparent or facecolor not in (None, "auto"):
                self.core.set_background(self._background)

    def _savefig(self, fname, dpi, format, kw):
        if dpi == "figure":
            dpi = None
        fmt = format
        if fmt is None and (isinstance(fname, str) or hasattr(fname, "__fspath__")):
            fmt = str(fname).rsplit(".", 1)[-1]  # a path: its extension (sextant checks it)
        if (fmt or "png").lower().lstrip(".") == "png":  # a file object without format: PNG
            kw["dpi"] = 96.0 * (float(dpi) if dpi is not None else self._dpi) / 100.0
        return self.core.savefig(fname, format=format, **kw)

    def show(self, block=None):
        self._shown = True
        self.core.show(block=block)

    def set_size_inches(self, w, h=None):
        if h is None:
            w, h = w
        self._size = (int(round(w * 100)), int(round(h * 100)))
        self.core.resize(*self._size)
        self._changed()

    def get_size_inches(self):
        return np.array(self._size, dtype=float) / 100.0

    def get_figwidth(self):
        return self._size[0] / 100.0

    def get_figheight(self):
        return self._size[1] / 100.0

    def set_figwidth(self, w):
        self.set_size_inches(w, self.get_figheight())

    def set_figheight(self, h):
        self.set_size_inches(self.get_figwidth(), h)

    def get_dpi(self):
        return self._dpi

    def set_dpi(self, dpi):
        raise NotImplementedError("set_dpi(): sextant fixes the dpi when the figure is made (figure(dpi=...))")

    def get_edgecolor(self):
        return (1.0, 1.0, 1.0, 1.0)

    def get_visible(self):
        return True

    @property
    def dpi(self):
        return self._dpi

    # sextant lays the decorations out itself, so matplotlib's layout calls are accepted and change nothing.
    def tight_layout(self, *args, **kw):
        """Accepted, no effect: sextant sizes decorations itself."""

    def subplots_adjust(self, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None):
        """Accepted, no effect: sextant sizes decorations itself (see core set_margins())."""

    def align_labels(self, axs=None):
        """Accepted, no effect."""

    align_xlabels = align_ylabels = align_titles = align_labels

    def set_tight_layout(self, tight):
        """Accepted, no effect."""

    set_constrained_layout = set_tight_layout

    def get_tight_layout(self):
        return False

    get_constrained_layout = get_tight_layout

    def sca(self, ax):
        """Make `ax` this figure's current axes (what gca() returns)."""
        if ax not in self.axes:
            raise ValueError("sca(): the axes is not in this figure")
        self.axes.remove(ax)
        self.axes.append(ax)
        return ax

    def clear(self):
        for ax in self.axes:
            ax.cla()

    clf = clear

    # --- keeping an open window current ---

    def _changed(self):
        if self._shown and self._interactive():
            self._refresh()

    def _refresh(self):
        if self.core.is_open():
            self.core.refresh()

    def _repr_png_(self):
        return self.core._repr_png_()

    def _repr_svg_(self):
        return self.core._repr_svg_()


def pause_loop(interval):
    """Deliver events and keep windows pumped for `interval` seconds."""
    end = time.monotonic() + max(float(interval), 0.0)
    while True:
        try:
            _sextant.poll_events()
        except RuntimeError:  # macOS off the main thread: nothing to pump here
            pass
        left = end - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(left, 0.01))


def interactive_session():
    return hasattr(sys, "ps1") or bool(sys.flags.interactive)
