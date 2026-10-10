# sextant for Python: `sextant.pyplot`

```python
import sextant.pyplot as plt
```

`sextant.pyplot` runs matplotlib-style code on sextant: the implicit current figure and axes, `plt.plot(x, y,
"r--")`, `fig, ax = plt.subplots()`, `ax.set_xlabel()`, `plt.show()`. Often the import line is the only
change a matplotlib script needs.

It covers what sextant can draw. A call or keyword sextant has no equivalent for **raises** (`AttributeError`,
`TypeError` or `ValueError` saying why) instead of being silently ignored, so a script never produces a plot
that quietly differs from what it asked for.

This page lists what is supported and where it differs from matplotlib. The core API underneath is in
[api.md](api.md) and [3d.md](3d.md).

- [A first example](#a-first-example)
- [Figures and axes](#figures-and-axes)
- [Plotting (2D)](#plotting-2d)
- [Plotting (3D)](#plotting-3d)
- [Decoration](#decoration)
- [Text and annotations](#text-and-annotations)
- [Colours, sizes and format strings](#colours-sizes-and-format-strings)
- [Changing what is drawn](#changing-what-is-drawn)
- [Showing, animation and interactive mode](#showing-animation-and-interactive-mode)
- [Events](#events)
- [Saving](#saving)
- [Jupyter](#jupyter)
- [Reaching the core objects](#reaching-the-core-objects)
- [Not supported](#not-supported)

---

## A first example

```python
import numpy as np
import sextant.pyplot as plt

x = np.linspace(0, 10, 100)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, np.sin(x), "r-", label="sin")
ax.scatter(x[::10], np.cos(x[::10]), label="cos, sampled")
ax.set_xlabel("x")
ax.set_title("waves")
ax.legend()
plt.savefig("waves.png")
plt.show()
```

The window that opens is sextant's: pan, zoom, hover readouts, and the Cosmetic and Data panels
([api.md](api.md#what-the-window-offers)).

---

## Figures and axes

| Call | Notes |
|---|---|
| `plt.figure(num=None, figsize=None, dpi=None, clear=False)` | `num` an int or a str label; an existing number makes that figure current |
| `plt.subplots(nrows=1, ncols=1, *, figsize=, dpi=, num=, width_ratios=, height_ratios=, squeeze=True, subplot_kw=)` | Returns `(fig, ax)` or `(fig, array of axes)` as matplotlib does |
| `plt.subplot(nrows, ncols, index)`, `plt.subplot(221)`, `plt.axes(projection=None)` | |
| `fig.add_subplot(nrows, ncols, index, projection=None)` | `projection="3d"` for an `Axes3D`; `index` may be a `(first, last)` span |
| `plt.gcf()`, `plt.gca()`, `plt.sca(ax)` | |
| `plt.close(fig=None)` | The current figure, a `Figure`, a number or label, or `"all"` |
| `plt.clf()`, `plt.cla()`, `fig.clear()`, `ax.cla()` / `ax.clear()` | |
| `plt.get_fignums()` | |
| `fig.set_size_inches(w, h)`, `fig.get_size_inches()`, `fig.dpi` | |
| `fig.tight_layout()`, `fig.subplots_adjust()` | Accepted and ignored: sextant measures its own layout |

**One subplot grid per figure.** sextant lays a figure out on one grid, fixed by its first subplot. A later
position on a grid that divides it evenly is mapped onto it: on a 2×2 figure, `add_subplot(2, 1, 2)` is the
bottom row (cells 3–4). A position that does not fit is a `ValueError`.

`sharex`/`sharey` are not supported (a warning; the axes stay independent).

---

## Plotting (2D)

Each is a method of the axes and a `plt.` function acting on the current axes.

| Call | Returns | Notes |
|---|---|---|
| `plot([x], y, [fmt], [x2], y2, [fmt2], ..., **kw)` | list of `Line2D` | Format strings, several groups, 2-D `y` (one line per column), `data=` |
| `line(...)` | the same | sextant's name for `plot`; the two are the same function |
| `scatter(x, y, s=None, c=None, marker=None, cmap=, vmin=, vmax=, alpha=, label=, colorbar=False, edgecolors=, linewidths=, facecolors=)` | `PathCollection` | `c` a colour, or one value per point to colour-map. `edgecolors`/`linewidths` draw an outline (matplotlib's default `'face'` outline is not drawn); `facecolors='none'` makes hollow markers |
| `bar(x, height, width=0.8, *, align="center", color=, edgecolor=, linewidth=, label=, alpha=, yerr=, xerr=, ecolor=, capsize=, error_kw=, tick_label=)` | `BarContainer` | |
| `hist(x, bins=10, range=None, density=False, weights=None, cumulative=False, histtype="bar", color=, label=, alpha=, rwidth=, edgecolor=, linewidth=)` | `(n, bins, BarContainer)` | Binned with `numpy.histogram`, so `n` and `bins` match matplotlib's |
| `errorbar(x, y, yerr=None, xerr=None, fmt="", ecolor=, elinewidth=, capsize=, label=)` | `Line2D` | `yerr`/`xerr` a scalar, `(N,)`, or `(2, N)` lower/upper; `fmt="none"` for bars alone |
| `imshow(X, cmap=, vmin=, vmax=, origin=None, extent=None, label=, colorbar=False)` | `AxesImage` | 2-D scalar data only |
| `pcolormesh([X, Y,] C, cmap=, vmin=, vmax=, shading=)` | `AxesImage` | A uniform mesh (cell edges or centres) |

Behaviour kept from matplotlib:

- `imshow`: `origin="upper"` (the default) puts row 0 at the top with the y axis pointing down; pixel centres
  sit on integers; `extent=(left, right, bottom, top)` places the image.
- `vmin`/`vmax` default to the data's range (in the core API they default to 0..1).
- `plot` with only a marker in its format (`"o"`) draws markers without a line; a format with both draws both.
- `legend(loc=...)` takes matplotlib's names and codes, mapped onto sextant's corners inside the frame (`"best"`
  is upper right).
- Lines and patches (scatter, bars, surfaces) take colours from separate `tab10` cycles, as in matplotlib.

Unknown keywords are passed on to the core call, so sextant's own options work in pyplot code too:
`ax.plot(x, y, loop=True)`, `ax.scatter(x, y, hint_labels=names)`. Anything that is neither matplotlib's nor
sextant's fails there, listing sextant's options. A set of matplotlib keywords that only tune rendering details
(`zorder`, `antialiased`, `picker`, `rasterized`, `clip_on`, `rstride`, `cstride`, …) is accepted and ignored.

---

## Plotting (3D)

`fig.add_subplot(projection="3d")` or `plt.subplots(subplot_kw={"projection": "3d"})` gives an `Axes3D` with
mplot3d's calls:

| Call | Notes |
|---|---|
| `ax.plot(xs, ys, zs=0, [fmt], **kw)` / `ax.line(...)` | A path; with only a marker in `fmt`, markers |
| `ax.scatter(xs, ys, zs=0, s=None, c=None, depthshade=True, marker=, cmap=, vmin=, vmax=, alpha=, label=, colorbar=False)` | `c` per point colour-maps |
| `ax.plot_surface(X, Y, Z, cmap=, color=, alpha=, edgecolor=, linewidth=, label=, vmin=, vmax=)` | A meshgrid (`"xy"` or `"ij"` indexing) or 1-D axes becomes a sextant surface; any other mesh (a sphere, a torus) becomes a triangle mesh |
| `ax.plot_wireframe(X, Y, Z, color=, linewidth=)` | A surface with transparent faces and drawn edges |
| `ax.plot_trisurf(x, y, z, triangles=None, cmap=, color=, ...)` | Without `triangles`, a Delaunay triangulation in xy |
| `ax.bar3d(x, y, z, dx, dy, dz, color=, alpha=, shade=True, edgecolor=, label=)` | The bars must form a grid (every x with every y, once) |
| `ax.text(x, y, z, s, **kw)`, `ax.text2D(x, y, s, **kw)` | `text2D` takes fractions of the axes (`ax.transAxes`); see [Text and annotations](#text-and-annotations). 3D text always faces the viewer (no `zdir`) |
| `ax.set_zlabel`, `set_zlim`, `get_zlim`, `set_zticks` | |
| `ax.view_init(elev=None, azim=None)` | No `roll`; z is up |
| `ax.set_proj_type("persp" or "ortho", focal_length=None)` | |
| `ax.set_box_aspect(aspect)` | `None` is matplotlib's default box (4, 4, 3) |

---

## Decoration

On axes (2D and 3D): `set_title`, `set_xlabel`, `set_ylabel` (with `fontsize=`), `get_title`, `get_xlabel`,
`get_ylabel`; `set_xlim`/`set_ylim` (positional, a pair, or `left=`/`right=`/`bottom=`/`top=`),
`get_xlim`/`get_ylim`, `invert_xaxis`, `invert_yaxis`; `set_xticks(ticks, labels=None)`,
`set_yticks`, `set_xticklabels`/`set_yticklabels` (after `set_xticks`/`set_yticks`); `grid(visible=None,
color=, linestyle=, linewidth=, alpha=)`; `legend(labels=None, loc=None, fontsize=None, frameon=None)`;
`ax.set_facecolor(color)` (2D); `ax.set(title=..., xlabel=..., xlim=..., ...)`. `set_xticks([])` hides the
ticks and their labels, as in matplotlib (the core API's `set_xticks([])` means automatic ticks).

On the figure: `fig.suptitle(t, fontsize=None)`, `fig.colorbar(mappable=None, ax=None, label="",
orientation="vertical")`, `fig.set_facecolor(color)`/`get_facecolor()`. `facecolor=` on `plt.figure()`
colours the figure as it is made, and on `fig.add_subplot()` the axes.

**Math.** Titles, labels, legend entries, tick labels and texts take `$...$` math, a subset of matplotlib's
mathtext ([api.md](api.md#math-in-text)). `plt.figure(mathtext=False)` turns it off for a figure;
`parse_math=False` turns it off for one `text()` or `annotate()` (on a title or label it raises, pointing to
the figure's switch). `usetex=True` raises: sextant draws its own math, with no LaTeX. Math that does not parse
is drawn as written, with a `RuntimeWarning`.

The same through `plt`: `title`, `xlabel`, `ylabel`, `xlim`, `ylim` (no arguments: get), `xticks`, `yticks`,
`grid`, `legend`, `suptitle`, `colorbar`.

**Colorbars** belong to the colour-mapped object in sextant: `plt.colorbar()` (or `fig.colorbar(im)`) turns
on the colorbar of the last (or given) colour-mapped object, and `label=` titles it. `orientation="horizontal"`
puts the axes' colorbars below the frame. Passing `colorbar=True` to `imshow`, `scatter` or a surface does the
same at once.

---

## Text and annotations

```python
ax.text(5, 0.8, "peak", ha="center", fontsize=12)
ax.text(0.02, 0.95, "run 41", transform=ax.transAxes, va="top")
ax.annotate("first peak", xy=(t0, y0), xytext=(30, 20), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
```

`text(x, y, s, fontdict=None, *, transform=None, **kw)` and `annotate(text, xy, xytext=None, xycoords="data",
textcoords=None, arrowprops=None, **kw)`, on the axes and through `plt`. Both return a `Text`.

- **Where:** `transform` is `ax.transData` (the default) or `ax.transAxes`. `xycoords` is `"data"` or
  `"axes fraction"` (an arrow needs `"data"`); `textcoords` adds `"offset points"` and `"offset pixels"`.
- **Text keywords:** `fontsize`/`size` (points, or `"small"`, `"large"`, ...), `color`, `alpha`,
  `ha`/`horizontalalignment`, `va`/`verticalalignment`, `rotation` (degrees, `"horizontal"`, `"vertical"`),
  `linespacing`, `clip_on`, `fontdict`, and `bbox` (`boxstyle`, `pad`, `facecolor`/`fc`, `edgecolor`/`ec`,
  `linewidth`/`lw`, `alpha`); every box draws square. sextant's own [TextOptions](reference.md#textoptions)
  names pass through.
- **`arrowprops`:** an `arrowstyle` among `-`, `->`, `-|>`, `<-`, `<|-`, `<->`, `<|-|>`, `|-|`, `-[`, `]-`,
  `]-[`, `fancy`, `simple`, `wedge` (the last three draw a filled head), with `head_length`/`head_width`;
  without one, matplotlib's default arrow (`width`, `headwidth`, `headlength`). Also `color`, `linewidth`,
  `linestyle`, `shrinkA`/`shrinkB`, and `connectionstyle="arc3,rad=..."`; other connection styles raise.

A text keeps a constant size on screen, never widens the limits, and is hidden while its data position is out
of view (`clip_on=True` cuts it at the frame instead).

---

## Colours, sizes and format strings

**Colours** accept what matplotlib accepts: `"C0"`–`"C9"`, `"tab:blue"` and the other tab10 names, the
single letters `b g r c m y k w`, `"#rgb"`, `"#rgba"`, `"#rrggbb"`, `"#rrggbbaa"`, grey levels as strings
(`"0.5"`), RGB/RGBA tuples, `"none"`, and the common CSS names with matplotlib's values (`"green"` is
`#008000`). Other names are looked up in matplotlib's table when matplotlib is installed.

**Sizes are in points**, as in matplotlib: a `figsize` in inches is the figure size at 100 dpi, and
`linewidth`, `markersize`, `fontsize`, `capsize` and `elinewidth` are points. Scatter `s` is a marker area in
points², default 36. One marker size per scatter series: a varying `s` raises.

**dpi**: `plt.figure()` saves 640×480 by default, and `dpi=200` (on the figure or in `savefig`) doubles it, as
in matplotlib.

**Format strings** combine a colour, a marker and a line style in any order (`"r--"`, `"o"`, `"k:"`,
`"C2-."`). sextant draws seven marker shapes, so matplotlib's markers map onto them: `o . , 8 h H` circle,
`s p` square, `^ v < > 1 2 3 4` triangle, `D d *` diamond, `x X` cross, `+ P | _` plus.

---

## Changing what is drawn

The returned handles have the usual setters:

| Handle | Methods |
|---|---|
| `Line2D` | `get_data`, `set_data`, `get_xdata`, `set_xdata`, `get_ydata`, `set_ydata`, `get_color`, `set_color`, `set_linewidth`, `set_linestyle`, `set_alpha`, `get_label`, `set_label` |
| `PathCollection` (scatter) | `get_offsets`, `set_offsets`, `get_array`, `set_array`, `set_clim`, `set_cmap` |
| `AxesImage` (imshow, pcolormesh) | `get_array`, `set_data`, `set_clim`, `set_cmap`, `get_extent` |
| `BarContainer` (bar, hist, bar3d) | `datavalues`, `set_heights(heights)` (sextant's; matplotlib changes bars one by one) |
| `Path3D` (3D plot, scatter) | `get_data_3d`, `set_data_3d` |
| `Surface3D` | `set_cmap` |
| `Text` (text, annotate, 3D text) | `get_text`, `set_text`, `get_position`, `set_position`, `set_color`, `set_fontsize`, `set_rotation` |

Data setters (`set_data`, `set_ydata`, `set_offsets`, `set_array`, …) update the plot in place and are cheap:
use them for animation. Style setters (`set_color`, `set_linewidth`, `set_clim`, `legend(labels)`,
`colorbar()`) redraw the axes from scratch, which resets a view panned or zoomed in the window.

---

## Showing, animation and interactive mode

`plt.show(block=None)` shows every figure. In a script it blocks until all their windows are closed; at an
interactive prompt (REPL, `python -i`, IPython) it returns at once and the windows stay live. Figures stay
registered until `plt.close()`, or until the user closes a window `show()` opened.

**Interactive mode.** `plt.ion()` (or a prompt) makes `show()` non-blocking and sends every change to an open
window immediately. Otherwise changes reach an open window at `plt.draw()`, `plt.pause()`,
`fig.canvas.draw()`, `fig.canvas.flush_events()` or `plt.show()`. `plt.ioff()` and `plt.isinteractive()` as
in matplotlib.

**Animation** is the usual `pause` loop:

```python
x = np.linspace(0, 2 * np.pi, 200)
fig, ax = plt.subplots()
(line,) = ax.plot(x, np.sin(x))
ax.set_ylim(-1.2, 1.2)
for k in range(300):
    line.set_ydata(np.sin(x + k / 10))
    plt.pause(0.02)              # shows the window, updates it, handles events
```

`plt.pause(t)` shows figures not yet shown, brings open windows up to date and processes events for `t`
seconds. On macOS it is also what keeps the windows responding.

---

## Events

```python
def onclick(event):
    if event.inaxes is ax:
        print(event.button, event.xdata, event.ydata)

cid = fig.canvas.mpl_connect("button_press_event", onclick)
fig.canvas.mpl_disconnect(cid)
```

Event names: `button_press_event`, `button_release_event`, `motion_notify_event`, `scroll_event`,
`key_press_event`, `key_release_event`, `resize_event`, `close_event`, `pick_event`. `plt.connect()` and
`plt.disconnect()` act on the current figure.

The event follows matplotlib's conventions: `x`, `y` in pixels with `y` measured from the bottom; `inaxes` the
pyplot axes or `None`; `xdata`/`ydata` `None` away from data; `button` 1 (left), 2 (middle), 3 (right), and
`"up"`/`"down"` with `step` for the wheel; `key`; `dblclick`. A `pick_event` has `artist` (the handle that was
clicked, for objects drawn through pyplot) and `ind` (the point index, as an array). sextant's own event is
`event.sextant_event`.

Callbacks run while the program waits in `plt.show()`, `plt.pause()` or `fig.canvas.flush_events()`, or at
an interactive prompt.

---

## Saving

```python
plt.savefig("figure.png")
fig.savefig("figure.svg")
fig.savefig("figure@2x.png", dpi=200)
fig.savefig(buffer, format="png")          # a file object
```

`fname` is a path or a file object; `format` is `"png"` or `"svg"` (by default the extension). `dpi` applies
to PNG. `facecolor=` and `transparent=True` recolour the figure background for that file only (a transparent
PNG or SVG). matplotlib's `bbox_inches`, `pad_inches`, `metadata` and similar keywords are accepted and
ignored. Saving needs no window.

---

## Jupyter

In a Jupyter kernel, pyplot behaves like matplotlib's inline backend: after each cell, every figure with
something on it is displayed as a PNG and closed. `plt.show()` in a cell displays at once. To get a live
window from a notebook instead, use the core API's `Figure.show()` ([api.md](api.md#interactive-sessions)).

---

## Reaching the core objects

Every pyplot object wraps a core sextant object, available as `.core`:

```python
fig, ax = plt.subplots()
ax.plot(x, y)
ax.core.set_axes_style(origin_x=0, origin_y=0)   # anything from the core API
fig.core.set_col_ratios([2, 1])
```

Calls made directly on `.core` are not replayed when a style setter redraws the axes
([Changing what is drawn](#changing-what-is-drawn)); make them after the pyplot calls, or prefer data setters.

`plt.axes_for(core_axes)` maps a core `Axes`/`Axes3D` (for example a core event's `inaxes`) back to its pyplot
axes.

---

## Not supported

These raise rather than draw something different:

- `axhline`, `axvline`, `fill_between`, `contour`/`contourf` (use a heatmap's
  `contours=`), `twinx`, log scales, `stackplot`, `pie`, `boxplot`, `violinplot`, `quiver`, `step`;
- `imshow` of RGB(A) images (scalar data only) and `alpha=` on images (a warning; ignored);
- a scatter with per-point sizes, bars with `bottom=` or varying widths, a `hist` with unequal bins or
  `histtype="step"`;
- a non-uniform `pcolormesh` mesh, a `bar3d` whose positions do not form a grid;
- `view_init(roll=)`, `vertical_axis` other than `"z"`, `Axes3D.plot(zdir=)` other than `"z"`;
- `legend(handles, labels)` (pass `label=` when plotting, or `legend(labels)`);
- subplot positions that do not fit the figure's grid ([Figures and axes](#figures-and-axes)).
