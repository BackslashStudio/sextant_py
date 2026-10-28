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
- [Polar axes](#polar-axes)
- [Draw order](#draw-order)
- [Decoration](#decoration) · [spines](#spines)
- [Scales, shared axes, twins and aspect](#scales-shared-axes-twins-and-aspect)
- [Text and annotations](#text-and-annotations)
- [Colours, sizes and format strings](#colours-sizes-and-format-strings)
- [Changing what is drawn](#changing-what-is-drawn) · [hiding artists](#hiding-artists) ·
  [reading back](#reading-back)
- [Settings: rcParams](#settings-rcparams)
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
| `plt.subplots(nrows=1, ncols=1, *, figsize=, dpi=, num=, facecolor=, width_ratios=, height_ratios=, squeeze=True, subplot_kw=)` | Returns `(fig, ax)` or `(fig, array of axes)` as matplotlib does |
| `plt.subplot(nrows, ncols, index)`, `plt.subplot(221)`, `plt.axes(projection=None, polar=False)` | |
| `fig.add_subplot(nrows, ncols, index, projection=None, polar=False)` | `projection="3d"` for an `Axes3D`, `"polar"` (or `polar=True`) for a [polar axes](#polar-axes); `index` may be a `(first, last)` span |
| `plt.gcf()`, `plt.gca()`, `plt.sca(ax)`, `fig.sca(ax)` | |
| `plt.fignum_exists(num)`, `plt.get_figlabels()`, `plt.figaspect(arg)` | |
| `plt.close(fig=None)` | The current figure, a `Figure`, a number or label, or `"all"` |
| `plt.clf()`, `plt.cla()`, `fig.clear()`, `ax.cla()` / `ax.clear()` | |
| `plt.get_fignums()` | |
| `fig.set_size_inches(w, h)`, `fig.get_size_inches()`, `fig.get_figwidth()`, `fig.get_figheight()`, `fig.dpi`, `fig.get_dpi()` | |
| `fig.tight_layout()`, `fig.subplots_adjust()`, `fig.align_labels()`, `plt.tight_layout()`, `plt.subplots_adjust()` | Accepted and ignored: sextant measures its own layout |

**One subplot grid per figure.** sextant lays a figure out on one grid, fixed by its first subplot. A later
position on a grid that divides it evenly is mapped onto it: on a 2×2 figure, `add_subplot(2, 1, 2)` is the
bottom row (cells 3–4). A position that does not fit is a `ValueError`.

`subplots(sharex=, sharey=)` takes `True`/`"all"`/`"col"`/`"row"`/`False`/`"none"` as matplotlib does, and
`add_subplot(sharex=, sharey=)` and `ax.sharex(other)` / `ax.sharey(other)` share one axis; see
[Scales, shared axes, twins and aspect](#scales-shared-axes-twins-and-aspect).

---

## Plotting (2D)

Each is a method of the axes and a `plt.` function acting on the current axes.

| Call | Returns | Notes |
|---|---|---|
| `plot([x], y, [fmt], [x2], y2, [fmt2], ..., **kw)` | list of `Line2D` | Format strings, several groups, 2-D `y` (one line per column), `data=` |
| `line(...)` | the same | sextant's name for `plot`; the two are the same function |
| `scatter(x, y, s=None, c=None, marker=None, cmap=, vmin=, vmax=, alpha=, label=, colorbar=False, edgecolors=, linewidths=, facecolors=)` | `PathCollection` | `c` a colour, or one value per point to colour-map; `s` one area in points² or one per point (a bubble chart). `edgecolors`/`linewidths` draw an outline (matplotlib's default `'face'` outline is not drawn); `facecolors='none'` makes hollow markers |
| `step(x, y, [fmt], *, where="pre", **kw)` | list of `Line2D` | `where` is `"pre"`, `"mid"` or `"post"`; also `plot(drawstyle="steps-mid")`. Markers stay on the points |
| `bar(x, height, width=0.8, bottom=None, *, align="center", color=, edgecolor=, linewidth=, label=, alpha=, yerr=, xerr=, ecolor=, capsize=, error_kw=, tick_label=)` | `BarContainer` | `bottom` stacks (one number or one per bar) |
| `barh(y, width, height=0.8, left=None, *, align="center", ...)` | `BarContainer` | Horizontal bars; `xerr` is the length error, `tick_label` names the y ticks |
| `hist(x, bins=10, range=None, density=False, weights=None, cumulative=False, bottom=None, histtype="bar", orientation="vertical", color=, label=, alpha=, rwidth=, edgecolor=, linewidth=)` | `(n, bins, BarContainer)` | Binned with `numpy.histogram`, so `n` and `bins` match matplotlib's |
| `fill_between(x, y1, y2=0, where=None, interpolate=False, step=None, **kw)`, `fill_betweenx(y, x1, x2=0, ...)` | `FillBetweenPolyCollection` | `where` leaves points out; `step` is `"pre"`, `"mid"` or `"post"`; `facecolor`/`color`, `edgecolor`, `alpha`, `linewidth`, `label` |
| `stackplot(x, *ys, labels=(), colors=None, baseline="zero", **kw)` | list of `FillBetweenPolyCollection` | One per layer; `baseline` is `"zero"`, `"sym"`, `"wiggle"` or `"weighted_wiggle"`; `hatch` raises |
| `quiver([X, Y], U, V, [C], scale=, scale_units=, units=, angles=, width=, headwidth=, headlength=, pivot=, cmap=, clim=, norm=, color=, alpha=, label=)` | `Quiver` | An arrow per point, black by default; `C` colour-maps (with `colorbar=` or `fig.colorbar(q)`). See below |
| `axhline(y=0, xmin=0, xmax=1)`, `axvline(x=0, ymin=0, ymax=1)` | `RefLine` | Reference lines across the frame; `color`, `linewidth`, `linestyle`, `alpha`, `label` |
| `axhspan(ymin, ymax, xmin=0, xmax=1)`, `axvspan(xmin, xmax, ymin=0, ymax=1)` | `Span` | Bands; opaque unless `alpha=`, no outline unless `edgecolor` (or `color=` with `linewidth`) |
| `errorbar(x, y, yerr=None, xerr=None, fmt="", ecolor=, elinewidth=, capsize=, label=)` | `Line2D` | `yerr`/`xerr` a scalar, `(N,)`, or `(2, N)` lower/upper; `fmt="none"` for bars alone |
| `imshow(X, cmap=, norm=, vmin=, vmax=, origin=None, extent=None, alpha=, label=, colorbar=False)` | `AxesImage` | 2-D scalar data, or an RGB(A) image `(rows, cols, 3 or 4)` (floats in [0, 1] or `uint8`), which `alpha=` fades |
| `pcolormesh([X, Y,] C, cmap=, norm=, vmin=, vmax=, shading=)` | `AxesImage` | Cell edges or centres, evenly spaced or not (1-D, or a meshgrid of them); a curvilinear mesh raises |
| `tripcolor(x, y, C, [triangles], shading="gouraud", cmap=, norm=, vmin=, vmax=)`, `tripcolor(triang, C)` | `AxesImage` | One value per vertex, interpolated across each triangle; no `triangles` = Delaunay. `shading="flat"` raises |
| `hist2d(x, y, bins=10, range=None, density=False, weights=None, cmin=None, cmax=None, cmap=, norm=, vmin=, vmax=)` | `(counts, xedges, yedges, AxesImage)` | Binned with `numpy.histogram2d`; `bins` an int, a pair, or edges |
| `contour([X, Y,] Z, [levels], colors=, cmap=, norm=, vmin=, vmax=, linewidths=, alpha=)`, `contourf(...)` | `ContourSet` | Lines at, or bands between, the levels; `origin`/`extent` for `contour(Z)` as `imshow`'s. `levels` an int or values |
| `clabel(cs, levels=None, fontsize=None)` | | Writes the levels on a `contour`'s lines |
| `pie(x, explode=, labels=, colors=, autopct=, pctdistance=0.6, labeldistance=1.1, startangle=0, radius=1, counterclock=True, wedgeprops=, textprops=, center=(0, 0), normalize=True)` | `(wedges, texts)`, plus `autotexts` with `autopct` | The axes takes equal aspect; `wedgeprops` takes `width` (a ring), `edgecolor`, `linewidth`, `alpha`; `textprops` `fontsize`, `color` |

Behaviour kept from matplotlib:

- `imshow`: `origin="upper"` (the default) puts row 0 at the top with the y axis pointing down; pixel centres
  sit on integers; `extent=(left, right, bottom, top)` places the image.
- `norm=` takes the norms of `sextant.colors` (`Normalize`, `LogNorm`, `TwoSlopeNorm`, `BoundaryNorm`, which
  mirror matplotlib's), or `"linear"`/`"log"`, on `imshow`, `pcolormesh`, `tripcolor`, `hist2d`, `scatter`,
  `quiver` and the contours. `hist2d(norm=LogNorm())` leaves empty bins blank, as matplotlib.
- `contour`/`contourf` draw on a grid (1-D `X`, `Y` or a meshgrid of them). Non-solid `linestyles`, `extend`,
  `hatches` and a curvilinear grid raise.
- `pie`: `explode` is in data units, as matplotlib; `shadow`, `hatch`, `rotatelabels` and a callable `autopct`
  raise.
- `vmin`/`vmax` default to the data's range (in the core API they default to 0..1).
- `plot` with only a marker in its format (`"o"`) draws markers without a line; a format with both draws both.
- Draw order follows `zorder`, with matplotlib's defaults: see [Draw order](#draw-order).
- A hidden artist (`set_visible(False)`) also leaves the automatic limits; matplotlib's still count.
- `legend(loc=...)` takes matplotlib's names and codes, mapped onto sextant's corners inside the frame (`"best"`
  is upper right).
- Lines and patches (scatter, bars, surfaces) take colours from separate `tab10` cycles, as in matplotlib.
  `ax.set_prop_cycle(color=[...])` (or `set_prop_cycle(cycler(color=[...]))`, or `None` for the default)
  replaces the list and restarts it; colours only.
- `axhline`/`axvline`/`axhspan`/`axvspan` take the first cycle colour and do not advance it, as in matplotlib.
- `quiver` follows matplotlib's argument forms (no `X`, `Y`: the index grid of `U`; 1-D `X`, `Y` spanning a 2-D
  `U`: a meshgrid). Arrows are drawn in data units (`angles="uv"` is drawn as `"xy"`, the same at equal aspect),
  and without `scale` sextant's own rule picks the length and keeps it, so `set_UVC()` frames stay comparable.
  `scale_units="dots"`/`"inches"`, per-arrow `angles`, an outline and `quiverkey` are not supported.

Unknown keywords are passed on to the core call, so sextant's own options work in pyplot code too:
`ax.plot(x, y, loop=True)`, `ax.scatter(x, y, hint_labels=names)`. Anything that is neither matplotlib's nor
sextant's fails there, listing sextant's options. A set of matplotlib keywords that only tune rendering details
(`antialiased`, `picker`, `rasterized`, `clip_on`, `rstride`, `cstride`, …) is accepted and ignored.

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
| `ax.quiver(X, Y, Z, U, V, W, length=1, arrow_length_ratio=, pivot="tail", normalize=False, color=, alpha=, label=)` | An arrow per point, `length` times the vector (`normalize=True`: all `length` long). The arrows are flat and face the eye |
| `ax.text(x, y, z, s, **kw)`, `ax.text2D(x, y, s, **kw)` | `text2D` takes fractions of the axes (`ax.transAxes`); see [Text and annotations](#text-and-annotations). 3D text always faces the viewer (no `zdir`) |
| `ax.set_zlabel`, `set_zlim`, `get_zlim`, `set_zticks` | |
| `ax.view_init(elev=None, azim=None)` | No `roll`; z is up |
| `ax.set_proj_type("persp" or "ortho", focal_length=None)` | |
| `ax.set_box_aspect(aspect)` | `None` is matplotlib's default box (4, 4, 3) |

---

## Polar axes

```python
fig = plt.figure()
ax = fig.add_subplot(projection="polar")
theta = np.linspace(0, 4 * np.pi, 400)
ax.plot(theta, theta / (4 * np.pi))
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_rmax(1.2)
```

`projection="polar"` (or `polar=True`) on `add_subplot`, `subplot`, `subplots(subplot_kw=)` or `plt.axes`, and
`plt.polar(theta, r)`, give a polar axes over the core [`PolarAxes`](api.md#polar-axes). Its data is (θ, r),
θ in radians; angles are in degrees where matplotlib's are (`set_thetagrids`, `set_thetamin`/`set_thetamax`,
`set_rlabel_position`, the zero location's offset).

| Call | Notes |
|---|---|
| `plot`, `step`, `scatter`, `bar`, `fill`, `fill_between`, `text`, `annotate` | As on a 2D axes, in (θ, r); a `bar` is a ring sector `width` radians wide, without error bars |
| `set_rlim`, `set_rmin`, `set_rmax`, `set_ylim`, `set_rticks`, `set_rgrids`, `set_rlabel_position` | `plt.rgrids(radii, labels=None, angle=None)` |
| `set_thetagrids(angles, labels=None)`, `set_xticks`, `set_thetamin`, `set_thetamax`, `set_thetalim`, `set_xlim` | `plt.thetagrids(...)`; a sector when the span is under 360° |
| `set_theta_zero_location(loc, offset=0)`, `set_theta_offset`, `set_theta_direction` | `loc` `"N"`, `"NW"`, `"W"`, …; direction `1`/`-1` or `"clockwise"` |
| `grid`, `set_title`, `legend`, `set_facecolor` | |

`set_rorigin`, a log r axis and error bars raise: sextant draws a linear r from the centre.

**`fill` in polar differs from matplotlib.** sextant runs a polygon's edges straight in (θ, r), so it bends
with the circles, where matplotlib draws them straight on screen. A closed curve given as many points looks
the same, but few points (a radar chart) do not: the fill follows arcs and spirals between them. For a region
from the centre out to a curve, use `fill_between(theta, 0, r)`: it fills inside the curve (its edges still
follow (θ, r), as matplotlib's own polar `fill_between`). Give a radar chart more points per side for straight
edges.

---

## Draw order

Objects draw in ascending `zorder`, equal ones in call order, with matplotlib's defaults: images and meshes
(`imshow`, `pcolormesh`, `tripcolor`, `hist2d`, `contourf`) 0, patches and collections (`bar`, `fill_between`,
`scatter`, spans, `pie`) 1, lines (`plot`, `contour`, reference lines, error bars) 2. So a line stays above an
`imshow` drawn after it. Every 2D plotting call takes `zorder=`, and the handles have `set_zorder(z)` and
`get_zorder()`; a `zorder` must be finite. Texts always draw above the data (their `get_zorder()` reads 3). A
3D scene draws by depth and ignores `zorder`.

---

## Decoration

On axes (2D and 3D): `set_title`, `set_xlabel`, `set_ylabel` (with `fontsize=` and `color=`), `get_title`, `get_xlabel`,
`get_ylabel`; `set_xlim`/`set_ylim` (positional, a pair, or `left=`/`right=`/`bottom=`/`top=`),
`get_xlim`/`get_ylim`, `invert_xaxis`, `invert_yaxis`; `set_xticks(ticks, labels=None)`,
`set_yticks`, `set_xticklabels`/`set_yticklabels` (after `set_xticks`/`set_yticks`); `grid(visible=None,
color=, linestyle=, linewidth=, alpha=)`; `legend(labels=None, loc=None, fontsize=None, frameon=None)`;
`ax.set_facecolor(color)` (2D); `ax.set(title=..., xlabel=..., xlim=..., ...)`; `ax.tick_params(axis="both",
which="major", ...)` with `colors`, `color`, `labelcolor`, `labelsize`, `length`, `width` and
`labelbottom`/`labeltop`/`labelleft`/`labelright` (`which="minor"` sets the minor mark length only; other
keywords such as `direction`, `pad` and `rotation` raise), also as `ax.xaxis.set_tick_params(...)`;
`ax.xaxis.label.set_color()`/`set_fontsize()`; `ax.label_outer()`. `set_xticks([])` hides the
ticks and their labels, as in matplotlib (the core API's `set_xticks([])` means automatic ticks).

On the figure: `fig.suptitle(t, fontsize=None)`, `fig.colorbar(mappable=None, ax=None, label="",
orientation="vertical")`, `fig.set_facecolor(color)`/`get_facecolor()`. `facecolor=` on `plt.figure()`
colours the figure as it is made, and on `fig.add_subplot()` the axes.

**Math.** Titles, labels, legend entries, tick labels and texts take `$...$` math, a large subset of
matplotlib's mathtext, fractions, roots and accents included ([api.md](api.md#math-in-text)).
`plt.figure(mathtext=False)` turns it off for a figure; `plt.figure(math_fontset="cm")` (or `"stix"`,
`"stixsans"`, `"dejavusans"`, `"dejavuserif"`, `"custom"` with `math_font_path=`) chooses the font math is drawn
in (or `rcParams["mathtext.fontset"]`, read when a figure is made);
`parse_math=False` turns it off for one `text()` or `annotate()` (on a title or label it raises, pointing to
the figure's switch). `usetex=True` raises: sextant draws its own math, with no LaTeX. Math that does not parse
is drawn as written, with a `RuntimeWarning`.

The same through `plt`: `title`, `xlabel`, `ylabel`, `xlim`, `ylim` (no arguments: get), `xticks`, `yticks`,
`grid`, `legend`, `suptitle`, `colorbar`.

**Colorbars** belong to the colour-mapped object in sextant: `plt.colorbar()` (or `fig.colorbar(im)`) turns
on the colorbar of the last (or given) colour-mapped object, and `label=` titles it. `orientation="horizontal"`
puts the axes' colorbars below the frame. Passing `colorbar=True` to `imshow`, `scatter` or a surface does the
same at once. A `contourf`'s colorbar is stepped at its levels; `fig.colorbar(h[3])` takes `hist2d`'s image.

### Spines

```python
ax.spines[["top", "right"]].set_visible(False)
ax.spines["left"].set_color("tab:red")
```

`ax.spines["left"]` (`"right"`, `"top"`, `"bottom"`), a list of them or `[:]` take `set_visible`,
`set_linewidth`, `set_color`/`set_edgecolor` and `set_alpha`. sextant draws one frame colour per axes, so
recolouring a spine whose side has no axis of its own recolours all four. `set_position` and `set_bounds`
raise.

---

## Scales, shared axes, twins and aspect

| Call | Notes |
|---|---|
| `ax.set_xscale(value, **kw)`, `set_yscale`, `get_xscale()`, `get_yscale()`, `plt.xscale()`, `plt.yscale()` | `"linear"`, `"log"`, `"symlog"` with `base`, `linthresh`, `linscale`; `subs` and `nonpositive` are dropped (sextant's log clips by itself) |
| `semilogx(...)`, `semilogy(...)`, `loglog(...)` | Axes method and `plt.` function |
| `ax.xaxis.set_major_formatter(f)`, `ax.yaxis...` | A format string (`"%.1f"`), `FormatStrFormatter` or `StrMethodFormatter` (`"{x:.2f}"`). `ScalarFormatter`/`LogFormatter` restore the default; a `FuncFormatter` raises `TypeError` |
| `minorticks_on()`, `minorticks_off()`, `grid(which="minor"\|"both")` | sextant's minor grid takes the major grid's style, so `which="minor"` alone turns the grid on |
| `subplots(sharex=, sharey=)`, `add_subplot(sharex=, sharey=)`, `ax.sharex(other)`, `ax.sharey(other)` | One set of limits and one scale per group; inner tick labels hide by themselves. There is no unsharing |
| `ax.twinx()`, `ax.twiny()`, `plt.twinx(ax=None)`, `plt.twiny(ax=None)` | The twin is in `fig.axes` and `plt.` makes it current. sextant has one twin per side, so a second call returns the same one |
| `ax.set_aspect(aspect, adjustable=None)`, `get_aspect()`, `set_adjustable()`, `get_adjustable()` | `aspect` is `"auto"`, `"equal"` or a number; `anchor` only `"C"` |
| `ax.axis(arg)`, `plt.axis(arg)` | `"equal"` (limits widen), `"scaled"`/`"image"` (frame shrinks), `"square"`, `"auto"`, `"on"`/`"off"`, a `[xmin, xmax, ymin, ymax]` list; `"tight"` raises |
| `ax.set_axis_off()`, `set_axis_on()`, `ax.axison` | Off hides the frame, ticks, labels, grid and background, as matplotlib |

Differences that are sextant's design:

- **A twin shares the cell.** Its title, legend (which lists both axes' series, in call order), colorbars and
  shared axis belong to the host: `set_title()`, `legend()` and the shared axis' label, ticks and format called
  on the twin act on the host. Objects of both axes draw in call order, not the twin on top. A twin's own axis
  is hidden by `axis("off")` on the twin only.
- An event over a twinned axes reports `inaxes` as the last twin made, with that twin's `xdata`/`ydata`.
- `imshow` keeps its frame unless you pass `aspect=`, whatever matplotlib's `aspect="equal"` default.
- `fill_between(interpolate=True)` runs a region on only where the curves really cross.

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
points², default 36, one value or one per point.

**dpi**: `plt.figure()` saves 640×480 by default, and `dpi=200` (on the figure or in `savefig`) doubles it, as
in matplotlib.

**Format strings** combine a colour, a marker and a line style in any order (`"r--"`, `"o"`, `"k:"`,
`"C2-."`). sextant draws sixteen marker shapes, and matplotlib's codes map onto them: `o` circle, `s` square,
`^ v < >` the triangles, `D d` diamond, `*` star, `p` pentagon, `h H` hexagon, `x` and `+` the thin line
crosses, `X` and `P` the filled ones, `| _` the line markers. A few have only a stand-in: `. , 8` draw circles
and `1 2 3 4` triangles.

---

## Changing what is drawn

The returned handles have the usual setters:

| Handle | Methods |
|---|---|
| `Line2D` | `get_data`, `set_data`, `get_xdata`, `set_xdata`, `get_ydata`, `set_ydata`, `get_color`, `set_color`, `set_linewidth`, `set_linestyle`, `set_alpha`, `get_drawstyle`, `set_drawstyle`, `get_label`, `set_label` |
| `PathCollection` (scatter) | `get_offsets`, `set_offsets`, `get_array`, `set_array`, `get_sizes`, `set_sizes`, `set_clim`, `set_cmap` |
| `FillBetweenPolyCollection` (fill_between, stackplot) | `set_data(t, f1, f2, *, where=None)`, `get_data` (sextant's), `get_facecolor`, `set_facecolor`, `get_edgecolor`, `set_edgecolor`, `set_color`, `set_linewidth`, `set_linestyle`, `set_alpha` |
| `Quiver` | `X`, `Y`, `U`, `V`, `N`, `set_UVC(U, V, C=None)`, `get_offsets`, `set_offsets`, `get_array`, `set_array`, `set_clim`, `set_cmap`, `set_color`, `set_alpha` |
| `RefLine` (axhline, axvline) | `get_xdata`, `set_xdata`, `get_ydata`, `set_ydata`, `get_color`, `set_color`, `set_linewidth`, `set_linestyle`, `set_alpha` |
| `Span` (axhspan, axvspan) | `get_span`, `set_span(lo, hi)` (sextant's), `get_facecolor`, `set_facecolor`, `set_edgecolor`, `set_linewidth`, `set_alpha` |
| `AxesImage` (imshow, pcolormesh, tripcolor, hist2d) | `get_array`, `set_data`, `set_clim`, `set_cmap`, `get_extent` |
| `ContourSet` (contour, contourf) | `levels`, `set_clim`, `set_cmap` |
| `PieWedge` (pie) | `theta1`, `theta2`, `r`, `center`, `set_facecolor`, `set_alpha` |
| `BarContainer` (bar, hist, bar3d) | `datavalues`, `set_heights(heights)` (sextant's; matplotlib changes bars one by one) |
| `Path3D` (3D plot, scatter) | `get_data_3d`, `set_data_3d` |
| `Surface3D` | `set_cmap` |
| `Text` (text, annotate, 3D text) | `get_text`, `set_text`, `get_position`, `set_position`, `set_color`, `set_fontsize`, `set_rotation` |

Data setters (`set_data`, `set_ydata`, `set_offsets`, `set_array`, `set_UVC`, …) update the plot in place and
are cheap: use them for animation. Style setters (`set_color`, `set_linewidth`, `set_clim`, `legend(labels)`,
`colorbar()`) redraw the axes from scratch, which resets a view panned or zoomed in the window.

Every handle also has `set_zorder()`/`get_zorder()` ([Draw order](#draw-order)).

### Hiding artists

Every handle has `set_visible(b)` / `get_visible()`, and the plotting calls take `visible=`. Hiding is cheap
like a data setter. For a `plot()` with markers it hides both the line and its markers; for `hist()` or
`stackplot()`, every bar or layer. Unlike matplotlib, a hidden artist leaves the automatic limits.

### Reading back

`ax.lines`, `ax.texts`, `ax.patches`, `ax.collections`, `ax.images` and `ax.get_children()` list the handles,
`ax.get_legend_handles_labels()` and `ax.get_legend()` the legend's, `ax.get_facecolor()` the background.
`get_xbound()`/`set_xbound()` (and y) read and set the limits low to high, `xaxis_inverted()` tells.
`get_xticks()` and `get_xticklabels()` return ticks set with `set_xticks()`; automatic ticks are chosen at draw
time, so for them they raise a `RuntimeError`. `plt.setp(artist, **props)`, `plt.getp(artist, name)` and
`plt.findobj(match=None)` work on these handles.

---

## Settings: rcParams

```python
plt.rcParams["axes.grid"] = True
with plt.rc_context({"figure.figsize": (8, 4), "image.cmap": "magma"}):
    fig, ax = plt.subplots()
```

`plt.rcParams`, `plt.rc(group, **kw)`, `plt.rc_context()` (also a decorator) and `plt.rcdefaults()` hold
matplotlib's keys. sextant reads `figure.figsize`, `figure.dpi`, `figure.facecolor`, `axes.facecolor`,
`axes.grid`, `axes.prop_cycle` (colours), `image.cmap` and `mathtext.fontset`, when a figure or axes is made;
setting another key stores it and warns once.

**Colormaps:** `plt.colormaps` (also `plt.cm`), `plt.get_cmap(name)`, and the current image's `plt.gci()`,
`plt.sci(im)`, `plt.clim(vmin, vmax)`, `plt.set_cmap(name)` and the per-map shortcuts (`plt.viridis()`).

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

- `boxplot`, `violinplot`, `quiverkey`, `hatch=`;
- bars with varying widths, a `hist` with unequal bins or `histtype="step"`;
- a curvilinear `pcolormesh` or `contour` mesh, a `bar3d` whose positions do not form a grid;
- in polar axes: `set_rorigin`, a log r axis, error bars;
- `view_init(roll=)`, `vertical_axis` other than `"z"`, `Axes3D.plot(zdir=)` other than `"z"`;
- a `FuncFormatter`, `usetex=True`;
- `legend(handles, labels)` (pass `label=` when plotting, or `legend(labels)`);
- subplot positions that do not fit the figure's grid ([Figures and axes](#figures-and-axes)).
