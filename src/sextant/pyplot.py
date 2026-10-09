"""matplotlib.pyplot over sextant: ``import sextant.pyplot as plt``.

The implicit current figure and axes, and the functions that act on them,
over matplotlib-shaped adapters (sextant/_mpl.py): ``plt.plot(x, y, 'r--')``,
``fig, ax = plt.subplots()``, ``ax.set_xlabel(...)``, ``plt.show()``. The
sextant objects underneath are ``fig.core`` / ``ax.core``.

A subset: what sextant can draw. Figures created here stay registered until
closed -- by plt.close(), or by the user closing a window plt.show() opened.
In a Jupyter kernel, figures are displayed inline (PNG) after each cell and
then closed, as matplotlib-inline does.
"""

import numpy as np

from . import _interactive, _sextant
from ._mpl import Axes, Axes3D, Figure, axes_for, interactive_session, pause_loop

__all__ = [
    "Axes", "Axes3D", "Figure", "annotate", "axes", "axes_for", "bar", "cla", "clf", "close", "colorbar", "connect",
    "disconnect", "draw", "errorbar", "figure", "gca", "gcf", "get_fignums", "grid", "hist", "imshow",
    "ioff", "ion", "isinteractive", "legend", "line", "pause", "pcolormesh", "plot", "savefig", "sca", "scatter",
    "show", "subplot", "subplots", "suptitle", "text", "title", "xlabel", "xlim", "xticks", "ylabel", "ylim",
    "yticks",
]

_figs = {}  # number -> Figure, in creation order
_current = None
_ion = False


# --- interactive mode ---------------------------------------------------------------

def isinteractive():
    """True with ion(), at a prompt (REPL, ``python -i``) or in IPython/Jupyter:
    show() does not block and changes reach an open window at once."""
    return _ion or interactive_session() or _interactive._ipython() is not None


def ion():
    global _ion
    _ion = True


def ioff():
    global _ion
    _ion = False


# --- figures --------------------------------------------------------------------------

def _prune():
    """Forget figures whose window the user closed, as matplotlib does."""
    global _current
    for num, fig in list(_figs.items()):
        if fig._shown and not fig.core.is_open():
            del _figs[num]
            if fig is _current:
                _current = None


def figure(num=None, figsize=None, dpi=None, clear=False, **kw):
    """Make figure `num` (an int, or a str label) current, creating it if needed."""
    global _current
    _prune()
    if isinstance(num, Figure):
        _current = num
        return num
    key = num
    if key is not None and key in _figs:
        fig = _figs[key]
    else:
        if key is None:
            key = max((k for k in _figs if isinstance(k, int)), default=0) + 1
        title = kw.pop("title", key if isinstance(key, str) else f"Figure {key}")
        fig = Figure(figsize=figsize, dpi=dpi, num=key, title=title, **kw)
        fig._interactive = isinteractive
        _figs[key] = fig
    if clear:
        fig.clear()
    _current = fig
    return fig


def gcf():
    _prune()
    return _current if _current is not None else figure()


def gca():
    return gcf().gca()


def sca(ax):
    """Make `ax` (and its figure) current."""
    global _current
    _current = ax.figure
    ax.figure.axes.remove(ax)
    ax.figure.axes.append(ax)


def subplot(*args, **kw):
    ax = gcf().add_subplot(*args, **kw)
    sca(ax)
    return ax


def axes(projection=None):
    return subplot(1, 1, 1, projection=projection)


def subplots(nrows=1, ncols=1, *, figsize=None, dpi=None, num=None, **kw):
    fig = figure(num=num, figsize=figsize, dpi=dpi)
    return fig, fig.subplots(nrows, ncols, **kw)


def close(fig=None):
    """Close the current figure, `fig` (a Figure, number or label) or "all"."""
    global _current
    if fig == "all":
        targets = list(_figs.values())
    elif fig is None:
        targets = [_current] if _current is not None else []
    elif isinstance(fig, Figure):
        targets = [fig]
    else:
        targets = [_figs[fig]] if fig in _figs else []
    for f in targets:
        f.core.close()
        for k, v in list(_figs.items()):
            if v is f:
                del _figs[k]
        if f is _current:
            _current = None


def clf():
    gcf().clear()


def cla():
    gca().cla()


def get_fignums():
    return sorted(k for k in _figs if isinstance(k, int))


# --- plotting on the current axes ------------------------------------------------------

def line(*args, **kw):
    """matplotlib's plot(), named after sextant's line(); plot is an alias."""
    return gca().line(*args, **kw)


plot = line


def scatter(*args, **kw):
    return gca().scatter(*args, **kw)


def bar(*args, **kw):
    return gca().bar(*args, **kw)


def hist(*args, **kw):
    return gca().hist(*args, **kw)


def errorbar(*args, **kw):
    return gca().errorbar(*args, **kw)


def imshow(*args, **kw):
    return gca().imshow(*args, **kw)


def pcolormesh(*args, **kw):
    return gca().pcolormesh(*args, **kw)


def text(x, y, s, *args, **kw):
    return gca().text(x, y, s, *args, **kw)


def annotate(text, xy, *args, **kw):
    return gca().annotate(text, xy, *args, **kw)


def title(label, **kw):
    return gca().set_title(label, **kw)


def xlabel(label, **kw):
    return gca().set_xlabel(label, **kw)


def ylabel(label, **kw):
    return gca().set_ylabel(label, **kw)


def xlim(*args, **kw):
    ax = gca()
    return ax.get_xlim() if not args and not kw else ax.set_xlim(*args, **kw)


def ylim(*args, **kw):
    ax = gca()
    return ax.get_ylim() if not args and not kw else ax.set_ylim(*args, **kw)


def xticks(ticks=None, labels=None, **kw):
    ax = gca()
    if ticks is None and labels is None:
        return ax._xticks
    ax.set_xticks(ticks, labels, **kw)


def yticks(ticks=None, labels=None, **kw):
    ax = gca()
    if ticks is None and labels is None:
        return ax._yticks
    ax.set_yticks(ticks, labels, **kw)


def grid(visible=None, **kw):
    gca().grid(visible, **kw)


def legend(*args, **kw):
    gca().legend(*args, **kw)


def colorbar(mappable=None, ax=None, **kw):
    fig = mappable.figure if mappable is not None else gcf()
    return fig.colorbar(mappable, ax=ax, **kw)


def suptitle(t, **kw):
    gcf().suptitle(t, **kw)


def savefig(*args, **kw):
    return gcf().savefig(*args, **kw)


def connect(name, fn):
    return gcf().canvas.mpl_connect(name, fn)


def disconnect(cid):
    gcf().canvas.mpl_disconnect(cid)


# --- showing ------------------------------------------------------------------------

def _kernel_shell():
    ip = _interactive._ipython()
    return ip if ip is not None and type(ip).__name__ != "TerminalInteractiveShell" else None


def show(block=None):
    """Show every figure. Blocks until all their windows are closed, unless the
    session is interactive (then returns at once). In a Jupyter kernel: display
    the figures inline and close them."""
    if _kernel_shell() is not None:
        _flush_inline()
        return
    _prune()
    if block is None:
        block = not isinteractive()
    for fig in list(_figs.values()):
        if fig.core.is_open():
            fig._refresh()
        else:
            fig.show(block=False)
    if block:
        _sextant.run()
        _prune()


def pause(interval):
    """Show the figures without blocking, bring open windows up to date and run
    their events for `interval` seconds -- the animation loop's step."""
    for fig in list(_figs.values()):
        if not fig.core.is_open() and not fig._shown:
            fig.show(block=False)
        fig._refresh()
    pause_loop(interval)


def draw():
    gcf()._refresh()


# --- Jupyter -----------------------------------------------------------------------

_inline_registered = False


def _flush_inline(*_):
    """After a cell (or at show()): display every figure with something on it, as
    PNG, and close them all."""
    if not _figs:
        return
    from IPython.display import display

    for fig in list(_figs.values()):
        if fig.axes:
            display(fig.core)
    close("all")


def _install_inline():
    global _inline_registered
    ip = _kernel_shell()
    if ip is not None and not _inline_registered:
        ip.events.register("post_execute", _flush_inline)
        _inline_registered = True


_install_inline()
