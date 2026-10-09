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
import sys
import time
import warnings
import weakref

import numpy as np

from . import _sextant

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

# matplotlib marker -> sextant marker (sextant draws seven shapes).
_MARKERS = {".": "circle", ",": "circle", "o": "circle", "8": "circle", "h": "circle", "H": "circle",
            "v": "triangle", "^": "triangle", "<": "triangle", ">": "triangle",
            "1": "triangle", "2": "triangle", "3": "triangle", "4": "triangle",
            "s": "square", "p": "square", "+": "plus", "P": "plus", "|": "plus", "_": "plus",
            "x": "cross", "X": "cross", "*": "diamond", "D": "diamond", "d": "diamond"}
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
            "aspect", "filternorm", "resample", "rstride", "cstride", "ccount", "rcount",
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


def _marker(m):
    if m is None:
        return None
    if m in ("", " ", "None", "none"):
        return "none"
    return _MARKERS.get(m, m)  # sextant names ("circle", ...) pass through


def _linestyle(ls):
    return None if ls is None else _LINESTYLES.get(ls, ls)


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

    def _set_options(self, **kw):
        self._entry[2].update(kw)
        self.axes._rebuild()

    def _set_args(self, method, *args):
        getattr(self.axes.core, method)(self._index, *args)
        self._entry[1][:len(args)] = args
        self.axes.figure._changed()


class Line2D(_Artist):
    """ax.plot()'s return: a line, markers (a scatter over it), or both."""

    def __init__(self, axes, kind, index, entry, label, marker=None):
        super().__init__(axes, kind, index, entry, label)
        self._markers = marker  # (index, entry) of the scatter drawing the markers

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


class PathCollection(_Artist):
    """ax.scatter()'s return."""

    def get_offsets(self):
        core = self.axes.core
        d = core.scatter_data(self._index) if self._kind == "scatter" else core.scatter_z_data(self._index)
        return np.column_stack([d.x, d.y])

    def set_offsets(self, xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        if self._kind == "scatter":
            self._set_args("set_scatter_data", xy[:, 0], xy[:, 1])
        else:
            z = self.axes.core.scatter_z_data(self._index).z
            self._set_args("set_scatter_z_data", xy[:, 0], xy[:, 1], z)

    def get_array(self):
        return self.axes.core.scatter_z_data(self._index).z if self._kind == "scatter_z" else None

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
        d = self.axes.core.heatmap_data(self._index)
        self._set_args("set_heatmap_data", np.asarray(a, dtype=float), d.xrange, d.yrange)

    def set_clim(self, vmin=None, vmax=None):
        if isinstance(vmin, (tuple, list)):
            vmin, vmax = vmin
        self._set_options(vmin=vmin, vmax=vmax)

    def set_cmap(self, cmap):
        self._set_options(cmap=cmap)

    def get_extent(self):
        d = self.axes.core.heatmap_data(self._index)
        return (*d.xrange, *d.yrange)


class BarContainer(_Artist):
    """ax.bar()/hist()'s bars."""

    @property
    def datavalues(self):
        return self.axes.core.bar_data(self._index).height

    def set_heights(self, heights):
        """Not matplotlib (there, each bar is a Rectangle): new heights, same x."""
        x = self.axes.core.bar_data(self._index).x
        self._set_args("set_bar_data", x, _vec(heights))


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


def _text_opts(kw, fontdict=None):
    """matplotlib Text keywords -> TextOptions fields. sextant's own names pass through."""
    if fontdict:
        kw = {**fontdict, **kw}
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

class _AxesBase:
    def __init__(self, figure, core):
        self.figure = figure
        self.core = core
        self._log = []  # [core method, args, kwargs]: everything this axes was told
        self._artists = {}  # (kind, index) -> artist, for pick events
        self._line_cycle = 0
        self._patch_cycle = 0
        self._grid_on = isinstance(self, Axes3D)
        self._xticks = self._yticks = self._zticks = None
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

    def _set_style(self, **fields):
        """Fields of the axes style, merged into what this adapter already set (sextant's
        set_axes_style() replaces the whole struct)."""
        self._style.update(fields)
        self._call("set_axes_style", **self._style)

    def _add(self, kind, method, *args, **kw):
        """A plotting call; returns (index, entry)."""
        index = getattr(self.core, f"{kind}_count")()
        return index, self._call(method, *args, **kw)

    def _rebuild(self):
        self.core.cla()
        for method, args, kw in self._log:
            getattr(self.core, method)(*args, **kw)
        self.figure._changed()

    def _next_color(self, patches=False):
        if patches:
            i, self._patch_cycle = self._patch_cycle, self._patch_cycle + 1
        else:
            i, self._line_cycle = self._line_cycle, self._line_cycle + 1
        return _hex(TAB10[i % 10])

    def cla(self):
        self.core.cla()
        self._log.clear()
        self._artists.clear()
        self._line_cycle = self._patch_cycle = 0
        self._grid_on = isinstance(self, Axes3D)
        self._style.clear()
        self.figure._changed()

    clear = cla

    # --- decoration ---

    def _title(self, method, label, fontsize, kw):
        if isinstance(fontsize, dict):  # matplotlib's positional fontdict
            kw = {**fontsize, **kw}
            fontsize = None
        _drop_ignored(kw)
        for k in ("fontdict", "loc", "pad", "labelpad", "fontweight", "weight", "y"):
            kw.pop(k, None)
        size = _pop(kw, "fontsize", "size", default=fontsize)
        if kw:
            raise TypeError(f"{method}(): unsupported keyword(s) {sorted(kw)}")
        args = (label,) if size is None else (label, float(size) * PT)
        self._call(method, *args)

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

    def _ticks(self, axis, ticks, labels, kw):
        kw.pop("minor", None)
        if kw:
            raise TypeError(f"set_{axis}ticks(): unsupported keyword(s) {sorted(kw)}")
        ticks = _vec(ticks)
        setattr(self, f"_{axis}ticks", ticks)
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
        if visible is None:
            visible = True if opts else not self._grid_on
        self._grid_on = bool(visible)
        self._call("grid", self._grid_on, **opts)

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

    _SERIES = {"line", "scatter", "scatter_z", "bar"}

    # --- plot --------------------------------------------------------------------

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
            i, entry = self._add("line", "line", x, y, **opts)
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

    def scatter(self, x, y, s=None, c=None, marker=None, cmap=None, vmin=None, vmax=None,
                alpha=None, label=None, colorbar=False, **kw):
        _drop_ignored(kw)
        for k in ("norm", "plotnonfinite"):
            kw.pop(k, None)
        edgecolors = _pop(kw, "edgecolors", "edgecolor", "ec")
        linewidths = _pop(kw, "linewidths", "linewidth", "lw")
        facecolors = _pop(kw, "facecolors", "facecolor", "fc")
        color = _pop(kw, "color", default=None)
        x, y = _vec(x), _vec(y)
        size = self._size(s)
        opts = dict(kw, size=size, alpha=1.0 if alpha is None else alpha)
        # An outline only when asked for (matplotlib's default 'face' outline is invisible here).
        if isinstance(edgecolors, str) and edgecolors.lower() == "none":
            edgecolors, linewidths = None, None
        elif edgecolors is not None or linewidths is not None:
            if edgecolors is not None and not (isinstance(edgecolors, str) and edgecolors == "face"):
                opts["edgecolor"] = to_rgba(edgecolors)
            opts["edge_linewidth"] = (1.5 if linewidths is None else float(np.asarray(linewidths).flat[0])) * PT
        if isinstance(facecolors, str) and facecolors.lower() == "none":
            opts["alpha"] = 0.0  # hollow: only the outline
            facecolors = None
        if marker is not None:
            opts["marker"] = _marker(marker)
        if label is not None:
            opts["name"] = str(label)
        values = _values(c, len(x))
        if values is not None:
            lo, hi = _auto_limits(values, vmin, vmax)
            opts.update(vmin=lo, vmax=hi, colorbar=bool(colorbar))
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
    def _size(s):
        if s is None:
            s = 36.0  # matplotlib's lines.markersize ** 2
        if np.ndim(s) > 0:
            s = np.asarray(s, dtype=float)
            if s.size and not np.all(s == s.flat[0]):
                raise ValueError("scatter(): sextant draws one marker size per series; s must be a scalar")
            s = float(s.flat[0]) if s.size else 36.0
        return math.sqrt(float(s)) * PT  # s is an area in points^2; sextant takes a diameter in px

    def bar(self, x, height, width=0.8, bottom=None, *, align="center", color=None, edgecolor=None,
            linewidth=None, label=None, alpha=None, yerr=None, xerr=None, ecolor=None, capsize=None,
            error_kw=None, tick_label=None, **kw):
        _drop_ignored(kw)
        if bottom is not None and np.any(np.asarray(bottom) != 0):
            raise ValueError("bar(): sextant bars stand on 0; bottom= is not supported")
        x, h = _vec(x), _vec(height)
        if np.ndim(width) > 0:
            w = np.asarray(width, dtype=float)
            if not np.all(w == w.flat[0]):
                raise ValueError("bar(): sextant draws one width per series")
            width = float(w.flat[0])
        spacing = float(np.min(np.diff(np.unique(x)))) if len(np.unique(x)) > 1 else 1.0
        if align == "edge":
            x = x + width / 2
        opts = dict(kw, width=float(width) / spacing,
                    color=to_rgba(color) if color is not None else self._next_color(patches=True))
        opts["edgecolor"] = to_rgba(edgecolor) if edgecolor is not None else opts["color"]
        opts["linewidth"] = float(linewidth) * PT if linewidth is not None else (0.0 if edgecolor is None else PT)
        if alpha is not None:
            opts["alpha"] = alpha
        if label is not None:
            opts["name"] = str(label)
        err = self._err(len(x), xerr, yerr, ecolor, capsize, error_kw, opts)
        i, entry = self._add("bar", "bar", x, h, err=err, **opts)
        if tick_label is not None:
            labels = [tick_label] * len(x) if isinstance(tick_label, str) else list(tick_label)
            self.set_xticks(x, labels)
        return BarContainer(self, "bar", i, entry, label)

    def hist(self, x, bins=10, range=None, density=False, weights=None, cumulative=False,
             histtype="bar", color=None, label=None, alpha=None, rwidth=None, edgecolor=None,
             linewidth=None, **kw):
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
        bars = self.bar(centers, n, width=widths[0] * (rwidth if rwidth is not None else 1.0),
                        color=color, label=label, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth,
                        **kw)
        return n, edges, bars

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

    def imshow(self, X, cmap=None, vmin=None, vmax=None, origin=None, extent=None, alpha=None,
               label=None, colorbar=False, **kw):
        _drop_ignored(kw)
        kw.pop("norm", None)
        a = np.asarray(X, dtype=float)
        if a.ndim != 2:
            raise ValueError("imshow(): sextant draws 2-D scalar data (rows, cols); RGB images are not supported")
        rows, cols = a.shape
        origin = origin or "upper"
        lo, hi = _auto_limits(a, vmin, vmax)
        opts = dict(kw, vmin=lo, vmax=hi, colorbar=bool(colorbar))
        if cmap is not None:
            opts["cmap"] = cmap
        if label is not None:
            opts["name"] = str(label)
        if extent is None:
            # matplotlib's: pixel centres on integers; 'upper' puts row 0 at the
            # top with the y axis pointing down.
            xr, yr = (-0.5, cols - 0.5), (-0.5, rows - 0.5)
            i, entry = self._add("heatmap", "heatmap", a, xr, yr, origin="lower", **opts)
            if origin == "upper":
                self._call("set_ylim", rows - 0.5, -0.5)
        else:
            left, right, bottom, top = extent
            i, entry = self._add("heatmap", "heatmap", a, (left, right), (bottom, top), origin=origin, **opts)
        if alpha is not None:
            warnings.warn("imshow(): alpha is not supported by sextant heatmaps; ignored", stacklevel=2)
        handle = AxesImage(self, "heatmap", i, entry, label)
        self.figure._mappable = handle
        return handle

    def pcolormesh(self, *args, cmap=None, vmin=None, vmax=None, shading=None, **kw):
        _drop_ignored(kw)
        if len(args) == 1:
            C = np.asarray(args[0], dtype=float)
            x = np.arange(C.shape[1] + 1, dtype=float)
            y = np.arange(C.shape[0] + 1, dtype=float)
        elif len(args) == 3:
            x, y, C = (np.asarray(a, dtype=float) for a in args)
            x = x[0] if x.ndim == 2 else x
            y = y[:, 0] if y.ndim == 2 else y
        else:
            raise TypeError("pcolormesh(C) or pcolormesh(X, Y, C)")
        rows, cols = C.shape
        if len(x) == cols:  # centres, as shading='nearest'/'auto' reads them
            dx = np.diff(x)
            x = np.concatenate([[x[0] - dx[0] / 2], x[:-1] + dx / 2, [x[-1] + dx[-1] / 2]])
        if len(y) == rows:
            dy = np.diff(y)
            y = np.concatenate([[y[0] - dy[0] / 2], y[:-1] + dy / 2, [y[-1] + dy[-1] / 2]])
        for edges, name in ((x, "X"), (y, "Y")):
            steps = np.diff(edges)
            if not np.allclose(steps, steps[0]):
                raise ValueError(f"pcolormesh(): sextant draws a uniform mesh; {name} is not evenly spaced")
        return self.imshow(C, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower",
                           extent=(x[0], x[-1], y[0], y[-1]), **kw)

    # --- text ----------------------------------------------------------------------

    def text(self, x, y, s, fontdict=None, *, transform=None, **kw):
        """Text at (x, y) in data coordinates, or in the axes' fraction with
        transform=ax.transAxes. Returns a Text."""
        coords = self._coords(transform, "text") or "data"
        opts = _text_opts(kw, fontdict)
        i, entry = self._add("text", "text", str(s), float(x), float(y), coords=coords, **opts)
        return Text(self, i, entry, (0, 1, 2))

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


def _values(c, n):
    """scatter's c: per-point values to colormap, or None for a colour."""
    if c is None or isinstance(c, str):
        return None
    a = np.asarray(c)
    if a.dtype.kind in "fiu" and a.ndim == 1 and a.size == n and not (isinstance(c, tuple) and n not in (3, 4)):
        return a.astype(float)
    return None


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

    _SERIES = {"line3d", "scatter3d", "surface", "surface_tri", "bar3d"}

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

    def scatter(self, xs, ys, zs=0, zdir="z", s=None, c=None, depthshade=True, marker=None, cmap=None,
                vmin=None, vmax=None, alpha=None, label=None, colorbar=False, **kw):
        _drop_ignored(kw)
        for k in ("linewidths", "edgecolors", "norm"):
            kw.pop(k, None)
        xs, ys = _vec(xs), _vec(ys)
        zs = np.broadcast_to(np.asarray(zs, dtype=float), xs.shape).astype(float)
        opts = dict(kw, size=Axes._size(s), depthshade=0.4 if depthshade else 0.0)
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

    def plot_wireframe(self, X, Y, Z, color=None, linewidth=None, label=None, **kw):
        color = to_rgba(_pop(kw, "colors", default=color)) or self._next_color()
        return self.plot_surface(X, Y, Z, color=color, alpha=0.0, edgecolor=color,
                                 linewidth=linewidth if linewidth is not None else 1.0, label=label, **kw)

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

    def text(self, x, y, z, s, zdir=None, **kw):
        """Text at the data point (x, y, z), at a fixed pixel size. zdir is not
        offered (sextant's 3D text always faces the viewer)."""
        if zdir is not None:
            raise ValueError("text(): sextant's 3D text always faces the viewer; zdir is not offered")
        opts = _text_opts(kw, kw.pop("fontdict", None))
        i, entry = self._add("text", "text", str(s), float(x), float(y), float(z), **opts)
        return Text(self, i, entry, (0, 1, 2, 3))

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
        self.inaxes = axes_for(e.inaxes)
        self.xdata = e.xdata if e.has_data else None
        self.ydata = e.ydata if e.has_data else None
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
            ax = self.inaxes
            self.artist = ax._artists.get((str(e.pick_kind), e.pick_object)) if ax and e.pick_plane < 0 else None
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
        w, h = figsize if figsize is not None else (6.4, 4.8)
        self._dpi = float(dpi) if dpi is not None else 100.0
        self._size = (int(round(w * 100)), int(round(h * 100)))
        title = kw.pop("title", f"Figure {num}" if num is not None else "sextant")
        facecolor = kw.pop("facecolor", None)
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

    def add_subplot(self, *args, projection=None, **kw):
        _drop_ignored(kw)
        kw.pop("label", None)
        facecolor = kw.pop("facecolor", None)
        if kw:
            raise TypeError(f"add_subplot(): unsupported keyword(s) {sorted(kw)}")
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
        else:
            raise ValueError(f"add_subplot(): projection={projection!r} is not supported (None or '3d')")
        ax = axes_for(core)
        if ax is None:
            ax = cls(self, core)
            self.axes.append(ax)
        if facecolor is not None and cls is Axes:
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
        if sharex or sharey:
            warnings.warn("subplots(): sharex/sharey are not supported by sextant; ignored", stacklevel=2)
        kw = dict(subplot_kw or {})
        axs = np.empty((nrows, ncols), dtype=object)
        for r in range(nrows):
            for c in range(ncols):
                axs[r, c] = self.add_subplot(nrows, ncols, r * ncols + c + 1, **kw)
        if width_ratios is not None:
            self.core.set_col_ratios(list(width_ratios))
        if height_ratios is not None:
            self.core.set_row_ratios(list(height_ratios))
        if squeeze:
            return axs.item() if axs.size == 1 else axs.squeeze()
        return axs

    def gca(self):
        return self.axes[-1] if self.axes else self.add_subplot(1, 1, 1)

    # --- decoration and output ---

    def suptitle(self, t, fontsize=None, **kw):
        _drop_ignored(kw)
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

    @property
    def dpi(self):
        return self._dpi

    def tight_layout(self, *args, **kw):
        """No-op: sextant sizes decorations itself."""

    def subplots_adjust(self, *args, **kw):
        """No-op: sextant sizes decorations itself (see core set_margins())."""

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
