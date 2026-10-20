# sextant for Python: API guide

```python
import sextant
```

This guide covers the core API: figures, 2D plots, saving, the window, live updates and events. 3D has its
own page, **[3d.md](3d.md)**. Every keyword option, enum and data class is tabulated in
**[reference.md](reference.md)**. matplotlib-style code (`import sextant.pyplot as plt`) is in
**[pyplot.md](pyplot.md)**. Installing is in **[install.md](install.md)**.

The core API follows sextant's C++ API name for name, so the C++ documentation applies too, read through the
[conventions](#conventions) below.

**Contents**

- [The model](#the-model)
- [Conventions](#conventions): [options](#options-are-keyword-arguments) · [enums](#enums-are-strings) ·
  [colours](#colours) · [arrays](#arrays) · [tuples](#tuples) · [typing](#typing)
- [2D plot types](#2d-plot-types): [line](#line) · [scatter](#scatter) · [scatter_z](#scatter_z) ·
  [bar](#bar) · [hist](#hist) · [heatmap and imshow](#heatmap-and-imshow) · [fill](#fill) ·
  [stack](#stack) · [quiver](#quiver) · [axline and axspan](#axline-and-axspan)
- [Error bars](#error-bars)
- [Contours](#contours)
- [Colormaps](#colormaps)
- [Titles, legends, colorbars and ticks](#titles-legends-colorbars-and-ticks)
- [Scales, shared and twin axes](#scales-shared-and-twin-axes): [log and symlog](#log-and-symlog-axes) ·
  [tick formats and minor ticks](#tick-formats-and-minor-ticks) · [shared axes](#shared-axes) ·
  [twin axes](#twin-axes) · [equal aspect](#equal-aspect)
- [Text and annotations](#text-and-annotations)
- [Math in text](#math-in-text)
- [Styling](#styling)
- [Layout](#layout): [subplots and spans](#subplots-and-spans) · [sizing a figure](#sizing-a-figure)
- [Saving and rendering](#saving-and-rendering)
- [Jupyter and IPython display](#jupyter-and-ipython-display)
- [The window](#the-window): [show() and waiting](#show-and-waiting) · [interactive sessions](#interactive-sessions) ·
  [what the window offers](#what-the-window-offers)
- [Live updates](#live-updates)
- [Events](#events)
- [Reading back](#reading-back)
- [Threads](#threads)
- [Errors and warnings](#errors-and-warnings)
- [Lifetime](#lifetime)

---

## The model

A `Figure` is a window, or an image file. Inside it is a grid of cells, each holding an `Axes` (a 2D plot
area) or an `Axes3D` (a 3D scene, see [3d.md](3d.md)). An `Axes3D` can in turn hold `Plane2D`s: 2D plot
areas placed in the scene.

```python
import numpy as np
import sextant

x = np.linspace(0, 10, 400)
fig = sextant.Figure(width=900, height=500, title="first plot")
ax = fig.axes()                      # the single axes of a 1x1 grid
ax.line(x, np.sin(x), name="sin").grid().legend()
fig.savefig("first.png")             # no window needed
fig.show()                           # in a script: blocks until the window is closed
```

Every plotting and decorating method of `Axes` returns the axes itself, so calls chain:

```python
(fig.axes()
    .line(x, y, name="signal")
    .scatter(px, py, name="samples")
    .set_title("Run 41")
    .legend()
    .grid())
```

**Data is copied.** Each call copies what it needs, so you can reuse or change your arrays afterwards. To
change what is plotted, use [`set_<kind>_data()`](#live-updates).

**Objects keep their identity.** `fig.axes() is fig.axes()`, and `fig.add_subplot(2, 2, 1)` returns the same
object every time it is asked for that cell. An `Axes` keeps its figure alive: a figure whose `Figure` object
you dropped stays open as long as you hold one of its axes.

---

## Conventions

### Options are keyword arguments

Each plotting call takes its options as keywords. They are the fields of the matching C++ options struct
(`LineOptions` for `line()`, and so on), listed with their defaults in [reference.md](reference.md):

```python
ax.line(t, volts, color="blue", linewidth=2, linestyle="--", name="channel A")
```

A misspelled option is an error, which lists the ones that exist:

```text
TypeError: line() got an unexpected keyword argument 'colour' (LineOptions fields: color, linewidth,
linestyle, name, show_legend, visible, alpha, loop, drawstyle, errorbar, hint_labels)
```

A nested options struct is a dict: `errorbar={"capsize": 3, "color": "black"}`,
`margins={"left": 30, "right": 30}`.

Every method's docstring lists its keywords (`help(sextant.Axes.line)`).

**Calls that set a whole style reset the fields you leave out.** `set_axes_style()`, `legend()`,
`set_colorbar_style()`, `Figure.set_suptitle_style()`, `Figure.set_margins()`, and in 3D `set_box_style()`,
`set_camera()` and `set_default_camera()` each replace the complete style: the fields you do not name return to
their defaults, not to what an earlier call set. Pass everything you want in one call.

### Enums are strings

Wherever sextant takes an enum, pass its name as a string. Names are lower snake case, as listed in
[reference.md](reference.md#enums). Matching ignores case, `_`, `-` and spaces, so `"dash_dot"`, `"DashDot"`
and `"dashdot"` are the same; a few short aliases are taken as well:

| Enum | Aliases |
|---|---|
| `LineStyle` | `"-"` solid, `"--"` dashed, `":"` dotted, `"-."` dashdot, `""` none |
| `MarkerStyle` | matplotlib's codes: `"o"` circle, `"s"` square, `"^"` `"v"` `"<"` `">"` the triangles, `"D"`/`"d"` diamond, `"*"` star, `"p"` pentagon, `"h"` hexagon, `"X"` cross, `"P"` plus, `"x"` thin_cross, `"+"` thin_plus, `"\|"` vline, `"_"` hline, `""` none |
| `DrawStyle` | matplotlib's `"steps"`, `"steps-pre"`, `"steps-mid"`, `"steps-post"` |
| `QuiverPivot` | `"mid"` middle |
| `MathFontset` | matplotlib's `"stixsans"`, `"dejavusans"`, `"dejavuserif"` |
| `Colormap` | `"grey"`; any name with `"_r"` for the reversed map |
| `HAlign` | `"centre"` |
| `Projection` | `"ortho"`, `"persp"` |
| `EventKind` | matplotlib's event names, see [Events](#events) |

The same names are also available as `str`-valued enum classes, for discoverability and autocompletion:
`sextant.LineStyle.DASHED`, `sextant.Colormap.VIRIDIS`, and so on. A member *is* its string
(`sextant.LineStyle.DASHED == "dashed"`), so the two are interchangeable everywhere. Calls that return an enum
return a member. Calling the class resolves any accepted spelling: `sextant.LineStyle("--") is
sextant.LineStyle.DASHED`. (Marker codes are the one case-sensitive spelling: `"x"` and `"X"` are different
shapes.)

### Colours

A colour is one of:

- a name: `"red"`, `"blue"`, `"green"`, `"orange"`, `"purple"`, `"cyan"` (matplotlib's tab10 shades, so
  `"blue"` is `#1f77b4`), `"black"`, `"white"`, `"gray"`/`"grey"`;
- a hex string: `"#rrggbb"` or `"#rrggbbaa"`;
- a tuple or list of 3 or 4 floats in [0, 1]: `(0.2, 0.4, 0.9)`, `(0.2, 0.4, 0.9, 0.5)`.

Colours read back as `(r, g, b, a)` tuples. A plot call's `color` is `None` by default: the series takes the
next colour of the axes' [colour cycle](#colour-cycle). An error bar's `color` may also be `None`, meaning the
series' own colour. (`sextant.pyplot` understands matplotlib's wider set: `"C0"`, `"k"`, `"tab:blue"`, CSS names.)

### Colour cycle

A series plotted without a `color` takes the axes' next colour from a ten-colour cycle (matplotlib's tab10),
for `line`, `scatter`, `bar`, `hist`, `fill` (and each layer of a `stack`), `quiver` without `c`, and
`axline`; a span takes none. Colour-mapped objects take no colour from it. `set_color_cycle()` replaces the
palette and restarts it; `[]` restores the default; `cla()` restarts it. Objects already plotted keep their
colour, and a twin continues its host's cycle.

```python
ax.set_color_cycle(["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"])
ax.line(x, a).line(x, b).line(x, c)         # taken in turn
```

Lines are drawn in the order you plotted them, whatever their kind: a later heatmap covers an earlier line.

### Arrays

Every data argument takes anything numpy can turn into a float64 array: numpy arrays of any numeric dtype,
lists, tuples, ranges, pandas columns. A C-contiguous float64 array is used without a copy into a temporary;
anything else goes through `numpy.asarray(x, dtype=float64)`. An array of the wrong dimensionality is a
`TypeError`; lengths that do not match are a `ValueError`.

2D grids (`heatmap`, `imshow`, and in 3D `bar3d`/`surface` heights) are 2-D arrays. Data read back comes as
numpy arrays that you own.

### Tuples

Small C++ structs are tuples: a range is `(lo, hi)`, a subplot span `(first, last)`, a figure size
`(width, height)`, a 3D point or box aspect `(x, y, z)`.

### Typing

The package ships type information (`py.typed`). A type checker (mypy, pyright) sees every keyword option
with its type, so `ax.line(x, y, colour="red")` or `linestyle="dashy"` is flagged before it runs. Enum
arguments are typed as the enum member or one of its canonical names and aliases; the case- and
separator-insensitive spellings work at run time but are type errors.

---

## 2D plot types

### line

```python
ax.line(x, y=None, *, err=None, **LineOptions) -> Axes
```

```python
ax.line(t, volts, color="blue", linewidth=2.0, linestyle="dashed", name="channel A")
ax.line(samples)                    # one argument: plotted against 0, 1, 2, ...
```

Line styles are `solid`, `dashed`, `dotted`, `dashdot` and `none`. `none` draws no stroke at all, and a series
set to it drops out of the legend. Dashing is honoured in the window, PNG and SVG alike.

`loop=True` adds a closing segment from the last point back to the first. It adds no point, so limits, hover
and the legend are unchanged.

`drawstyle` draws the line as steps (matplotlib's `drawstyle` and `step()`): `"steps_pre"`, `"steps_mid"` or
`"steps_post"` put the riser before, between or after each point. The vertices, hover and picks are still the
data's.

Lines are the kind that scales: a million points pan and zoom smoothly. There is no marker option; draw a
`scatter()` over the same data for a marked series.

### scatter

```python
ax.scatter(x, y, *, err=None, **ScatterOptions) -> Axes
```

```python
ax.scatter(px, py, color="orange", size=24, marker="diamond", alpha=0.6)
```

`size` is the marker's diameter in pixels and stays the same when you zoom. There are sixteen markers, as
matplotlib's `o s ^ v < > D * p h X P x + | _`: `circle`, `square`, `triangle`, `triangle_down`,
`triangle_left`, `triangle_right`, `diamond`, `star`, `pentagon`, `hexagon`, `cross`, `plus`, the line markers
`thin_cross`, `thin_plus`, `vline` and `hline` (stroked `edge_linewidth` thick), and `none`. One colour per
series.

**A size per point** makes a bubble chart: `sizes` holds one diameter in pixels per point (`None`: all
`size`). A NaN or non-positive size hides that point. `scatter_z` and `scatter3d` take it too, and
`ScatterData.sizes` reads and sets it.

```python
ax.scatter(x, y, sizes=8 + 40 * weight, alpha=0.5)
```

**Outlines and hollow markers.** `edge_linewidth` (pixels, 0 by default) draws an outline inside the marker,
so `size` stays its full diameter. `edgecolor` sets its colour (unset: the marker's own) and `edge_alpha` its
opacity, independent of `alpha`. With `alpha=0` only the outline is drawn: a hollow marker. Every shape takes
one, `cross` and `plus` included (both are filled shapes). `scatter_z` and `scatter3d` take the same keywords.

```python
ax.scatter(px, py, color="orange", size=16, alpha=0, edge_linewidth=2)       # hollow
```

### scatter_z

A scatter whose colour carries a third value.

```python
ax.scatter_z(x, y, z, *, err=None, **ScatterZOptions) -> Axes
```

```python
ax.scatter_z(x, y, temperature, vmin=-10, vmax=40, colorbar=True, name="T (°C)")
```

Each point's colour is `z[i]` mapped through `cmap`, `vmin` and `vmax`. **`vmin`/`vmax` default to 0 and 1**,
not to the data's range, so set them for data outside [0, 1]. `colorbar=True` draws a bar explaining the
mapping; `name` titles it and keys the series in the legend (its marker shape, filled white).

### bar

```python
ax.bar(x, height, *, err=None, **BarOptions) -> Axes
```

```python
ax.bar(months, balance, color="blue", width=0.7, name="surplus", edgecolor="black")
```

Bars are centred on `x`. `width` is a fraction of the spacing between bars, so `1.0` makes them touch.
Negative heights draw downward from zero. `orientation="horizontal"` lays the bars along x (matplotlib's
`barh`: `x` then holds the bar positions on y, and `height` the lengths), and `bottom` gives each bar a base
other than 0: one value per bar, or one number, which stacks bars:

```python
ax.bar(q, a, name="A").bar(q, b, bottom=a, name="B")
ax.bar(names, counts, orientation="horizontal")
```

 Name the positions with [`set_xticks()`](#ticks).

### hist

```python
ax.hist(data, bins=10, *, density=False, cumulative=False, **BarOptions) -> Axes
```

```python
ax.hist(a, 40, color="blue", alpha=0.6, density=True, name="A")
ax.hist(b, 40, color="orange", alpha=0.6, density=True, name="B")
```

A histogram is a bar plot whose bars come from binning `data` into `bins` equal bins over its range.
`density=True` normalizes so the bars' area is 1; `cumulative=True` makes each bin count itself and every bin
before it. The bar keywords style the bars, and here `width` (a fraction of the bin width) defaults to `1.0`,
so bins touch. `hist()` takes no error bars. Its bars are read back with `bar_data()`.

### heatmap and imshow

```python
ax.heatmap(data, xrange, yrange, **HeatmapOptions) -> Axes
ax.imshow(data, **HeatmapOptions) -> Axes
```

```python
# A 90x140 grid placed over 400..700 nm by -1..1:
ax.heatmap(field, (400, 700), (-1, 1), cmap="inferno", vmin=0, vmax=1, colorbar=True, name="intensity")

# Or, when the indices are the coordinates:
ax.imshow(field, colorbar=True)
```

`data` is a 2-D array of shape `(rows, cols)`. Values are stored in single precision.

`xrange` and `yrange` place the image in data space. They are the **outer edges** of the mesh, not cell
centres, so a cell is `(hi - lo) / cols` wide. The mesh is uniform. `imshow()` is the case `(0, cols)` by
`(0, rows)`: one unit per cell. A reversed range (`lo > hi`) mirrors that axis; an empty or non-finite one is a
`ValueError`.

`origin="lower"` (the default) draws row 0 at the bottom, `"upper"` at the top. As with `scatter_z`,
**`vmin`/`vmax` default to 0 and 1**.

Automatic limits never pad a heatmap: the image meets the frame. Hovering a cell in the window reports its row,
column and value.

(`sextant.pyplot`'s `imshow` follows matplotlib instead: row 0 at the top, cell centres on integers, colour
limits from the data.)

### fill

The area between two curves, as matplotlib's `fill_between`:

```python
ax.fill(x, y1, y2=0.0, **FillOptions) -> Axes
```

```python
(ax.fill(x, lo, hi, color="blue", alpha=0.3, interpolate=True, name="±σ")
   .line(x, mean, color="blue"))
```

`y2` is a number or one value per point. `orientation="vertical"` fills between two x curves along y
(`fill_betweenx`; the arrays then read `y, x1, x2`). `exclude` is one flag per point that leaves it out (the
inverse of matplotlib's `where`), and a NaN splits the fill; each run left is a region of its own. Where the
two curves cross, `interpolate=True` runs a region on to the crossing instead of stopping at the last point.
`drawstyle` steps the curves, and `edgecolor` with `linewidth` outlines the region. Lengths that disagree are a
`ValueError`.

Automatic limits cover every point of both curves. A fill draws in call order, has a legend key, and is picked
and hovered anywhere inside it (`PickKind.FILL`, with the segment under the cursor in `pick_index`).
`fill_data(i)` / `set_fill_data(i, ...)` read and replace it, and the Data panel edits it. `Plane2D` has
`fill()` too.

### stack

Stacked areas, as matplotlib's `stackplot`: one fill per layer, each from the top of the layers below it.

```python
ax.stack(x, ys, **StackOptions) -> Axes
```

```python
ax.stack(t, layers, baseline="wiggle", names=["a", "b", "c", "d"])
```

`ys` is a 2-D array with one row per layer, or a sequence of 1-D arrays, each one value per `x`. `baseline` is
`"zero"`, `"sym"` (centred on zero), `"wiggle"` or `"weighted_wiggle"` (streamgraphs). `colors` and `names` are
empty or one per layer, and `alpha`, `edgecolor`, `linewidth`, `linestyle` and `drawstyle` apply to all. The
curves are computed once: afterwards each layer is an ordinary fill, read back with `fill_data()` and edited as
one, and editing a layer does not restack the rest. A NaN carries up the stack. `Plane2D` has `stack()` too.

### quiver

An arrow per point, as matplotlib's `quiver` with `angles='xy'` and `scale_units='xy'`:

```python
ax.quiver(x, y, u, v, c=None, **QuiverOptions) -> Axes
```

```python
ax.quiver(x, y, u, v, speed, cmap="viridis", colorbar=True, amp=0.18, pivot="middle").set_aspect(1.0)
```

An arrow runs from `(x, y)` along `(u, v)` times `amp`, **in data units**, so it points the way the data does
only at [equal aspect](#equal-aspect). `amp` (matplotlib's `1/scale`) is chosen from the data when unset and
then kept, so the frames of an animation stay comparable. `pivot` is `"tail"`, `"middle"` or `"tip"`; `width`,
`head_width` and `head_length` are in pixels. With `c` (one value per arrow), each arrow is coloured through
`cmap`, `vmin` and `vmax`, with an optional colorbar; an arrow with a NaN, a zero vector or an end at or below
0 on a log axis is not drawn. Automatic limits cover whole arrows. Hover shows the vector (`PickKind.QUIVER`),
and `quiver_data(i)` / `set_quiver_data(i, ...)` read and replace it. `Plane2D` has `quiver()` too.

### axline and axspan

Reference lines and bands across the frame:

```python
ax.axline(value, **AxLineOptions) -> Axes
ax.axspan(lo, hi, **AxSpanOptions) -> Axes
```

```python
(ax.axspan(0.7, 1.3, color="green", name="±0.3")                          # axhspan
   .axline(1.0, linestyle="dashed", name="mean")                          # axhline
   .axline(6.2, orientation="vertical", name="event"))                    # axvline
```

Horizontal by default (at y = `value`, over y); `orientation="vertical"` puts them on x. `from_` and `to`
(fractions of the frame, 0 to 1; `from_` because `from` is a Python keyword) limit them to part of it, and a
span can be outlined with `edgecolor` and `linewidth`. They stay put through pan and zoom along their length,
draw in call order, have legend keys, and are edited in the Data panel. Automatic limits include their values
(on a log axis only those > 0); a non-finite value is a `ValueError`. They are read back and moved with
`axline_data(i)` / `set_axline_data(i, value)` and `axspan_data(i)` / `set_axspan_data(i, lo, hi)`. They are
decorations: never hovered or picked.

---

## Error bars

An error bar decorates a series. Its **data** is a `sextant.ErrorBar` passed as `err=`; its **style** is the
series' `errorbar=` keyword, a dict of [ErrorBarOptions](reference.md#errorbaroptions). An error bar has two
shapes, either of which can be left out:

- a **capped whisker**, for an observed range (`x_cap_lo`, `x_cap_hi`, `y_cap_lo`, `y_cap_hi`);
- a **box**, for a spread such as one standard deviation (`x_box_lo`, `x_box_hi`, `y_box_lo`, `y_box_hi`).

Every value is an **offset from the point**, not a coordinate, and is read as a size (a negative number counts
as its absolute value). Give one end only and the bar is symmetric; a zero or NaN leaves that side undrawn,
which makes a one-sided bar. Each array holds one value per point.

```python
from sextant import ErrorBar

# A whisker for the observed range and a box for one standard deviation:
ax.line(x, y, err=ErrorBar(y_cap_lo=lo, y_cap_hi=hi, y_box_lo=sigma), color="blue", name="measured")

# One-sided "at least this much": zeros below, arrow caps above.
ax.scatter(x, y, err=ErrorBar(y_cap_lo=np.zeros_like(up), y_cap_hi=up), errorbar={"capstyle": "arrow"})

# Bars hang theirs off the tip:
ax.bar(centers, heights, err=ErrorBar(y_cap_lo=e), errorbar={"color": "black", "capsize": 10})

# x and y together; a box in both directions is a rectangle:
ax.scatter(x, y, err=ErrorBar(x_box_lo=sx, y_box_lo=sy))
```

`line`, `scatter`, `scatter_z` and `bar` take error bars. Without box data in a direction the box is
`boxwidth` pixels across. Automatic limits widen to fit the bars. An unset `color` is the series' own (a bar's
edge colour; black for `scatter_z`).

---

## Contours

A heatmap can trace iso-lines through itself:

```python
ax.heatmap(field, (-3, 3), (-3, 3), cmap="coolwarm", vmin=-8, vmax=8, colorbar=True,
           contours=[-6, -4, -2, 0, 2, 4, 6], contour_color="#222222", contour_labels=True)
```

Levels are values in the data's own units, the numbers on the colorbar. They are sorted and de-duplicated for
you. `contour_labels=True` writes each level on its line. Lines pass through cell centres, so they stop half a
cell inside the image, and a heatmap smaller than 2×2 traces nothing.

---

## Colormaps

Every colour-mapped kind takes `cmap=`: `heatmap`/`imshow`, `scatter_z`, `quiver` with a `c`, and in 3D
`surface`, `surface_tri`, `scatter3d` and `line3d`. There are 26 built-in maps:

| `cmap` | For |
|---|---|
| `"viridis"` | The default. Perceptually uniform, readable in greyscale |
| `"plasma"` | Like viridis with more contrast |
| `"inferno"`, `"magma"` | Dark to bright: data that should glow against black |
| `"cividis"` | Designed for colour-vision deficiency |
| `"turbo"` | A rainbow, for readers used to jet; not perceptually uniform |
| `"gray"`, `"Greys"`, `"hot"`, `"jet"` | Greyscale images, and the classic heat and rainbow maps |
| `"Blues"`, `"Greens"`, `"Reds"`, `"Oranges"`, `"Purples"`, `"YlOrRd"`, `"YlGnBu"` | Sequential, light to dark: counts, densities |
| `"coolwarm"`, `"RdBu"`, `"Spectral"`, `"BrBG"`, `"PiYG"` | Diverging: data with a meaningful centre, such as signed values |
| `"twilight"`, `"twilight_shifted"`, `"hsv"`, `"phase"` | Cyclic: angles and phases, whose ends meet |

The colours match matplotlib's maps of the same names. `vmin` and `vmax` choose the data range the map spans;
values outside it take the end colours, **except with a cyclic map, where they wrap round it**, so an angle reads
right past ±π.

Any name takes `"_r"` for the reversed map (`cmap="RdBu_r"`), in any case. Your own map is registered for the
rest of the process and then used by name:

```python
sextant.register_colormap("sunset", ["blue", "white", "red"])
sextant.register_colormap("clock", colors, cyclic=True)    # wraps instead of clamping
ax.heatmap(field, (0, 1), (0, 1), cmap="sunset_r")
sextant.colormaps()                                        # every name, registered ones last
```

`register_colormap()` spreads two or more colours evenly from `vmin` to `vmax` (alpha included) and returns the
name; a name already taken, empty or ending in `"_r"` is a `ValueError`, and a registered map cannot be removed.
The panel's colormap list shows registered maps too, with a Reversed toggle.

---

## Titles, legends, colorbars and ticks

```python
(ax.set_title("Run 41", 18)
   .set_xtitle("time (s)")
   .set_ytitle("voltage (V)")
   .set_xlim(0, 10)
   .set_ylim(-1, 1)
   .grid()
   .legend())

fig.suptitle("Experiment 7")      # one title over the whole subplot grid
```

"Title" names an axis or the whole axes; "label" is the text at a tick. Font sizes are pixels as drawn.

Limits are automatic, error bars included, until you set them. `xlim()`/`ylim()` read back the limits as
drawn.

**Legend.** A series is listed when it has a non-empty `name` and `show_legend` is left `True`. `legend()`
shows the legend and places it: `anchor` puts it inside a corner of the frame (`"inside_tl"` …
`"inside_br"`, taking no space) or outside it (`"outside_rt"` by default, and eleven others; the plot shrinks
to make room).

```python
ax.legend(anchor="inside_tl", fontsize=12)
```

**Colorbars.** A colorbar appears because a plot object asked for one (`colorbar=True` on `heatmap`,
`imshow` or `scatter_z`), and that object's `name` titles it. Several can share an axes: each gets its own bar,
stacked outward in plot order. `set_colorbar_style()` styles all of an axes' bars and chooses the side
(`anchor="left"`, `"right"`, `"top"`, `"bottom"`).

### Ticks

```python
days = [0, 1, 2, 3, 4]
ax.set_xticks(days, ["Mon", "Tue", "Wed", "Thu", "Fri"])
ax.set_xticks(days)              # positions only; the values are the labels
ax.set_xticks([])                # back to automatic ticks
```

`set_xticks([])` means automatic ticks, not none (`sextant.pyplot`'s follows matplotlib and hides them). To
hide an axis' ticks, see [Styling](#styling).

`cla()` resets the axes completely: plot objects, limits, titles, styles, legend and ticks. To change a
series' data and keep everything else, use [`set_<kind>_data()`](#live-updates).

> **Ordering.** `set_title(text, fontsize)` stores its size in the axes style, so a later `set_axes_style()`
> resets it. Call `set_axes_style()` first.

---

## Scales, shared and twin axes

### Log and symlog axes

```python
ax.set_xscale("log").set_yscale("symlog", linthresh=1.0).grid(True, minor=True)
```

The scales are `"linear"` (the default), `"log"` and `"symlog"`, with `base` (10), and for symlog `linthresh`
(the linear range around 0) and `linscale`, as matplotlib's. `xscale()` / `yscale()` read them back. **Limits
stay in data units**; pan and zoom move whole decades.

On a log axis values <= 0 are clipped off the frame (a bar stands on its bottom edge) and ignored by the
automatic limits, and `set_xlim()` / `set_ylim()` with an end <= 0 is a `ValueError`. Switching to `"log"` drops
set limits that reach 0 back to automatic. Ticks fall on the decades, labelled `10^3`; symlog adds 0 and the
negative decades. A [heatmap](#heatmap-and-imshow) keeps its axes linear (`ValueError` otherwise).

### Tick formats and minor ticks

`set_xtick_format("%.1f")` / `set_ytick_format()` take printf's format for one number (`%f`, `%e` or `%g`,
with flags, width and precision, plus any text, [math](#math-in-text) included): `"%.0f%%"`, `"$%.1f$ ms"`. It
labels the generated ticks and any `set_xticks()` positions given without a label. Empty restores the default;
anything else (no conversion, or two) is a `ValueError`.

Minor ticks are unlabelled marks between the major ones, on by default for log and symlog axes;
`set_axes_style(minor_xticks=, minor_yticks=)` turn them on or off, and `minor_tick_length` sizes them.
`grid(minor=True)` adds grid lines at them, in `minor_color`.

`set_axes_style()` also has per-axis variants of the tick and label fields (`xtick_color`, `ytick_length`,
`zlabel_fontsize`, …), unset by default so the shared field applies, and `show_xtick_labels` /
`show_ytick_labels` to hide an axis' labels while keeping its marks.

### Shared axes

```python
top.line(x, a)
bottom.line(x, b).sharex(top)           # also sharey()
top.set_xlim(2, 8)                      # moves both
```

Axes that share an axis hold **one set of limits and one scale**: `set_xlim()`, `set_xscale()`, pan, zoom and
the panel's fields on any of them move them all, and automatic limits cover all their data. Tick positions,
formats and titles stay per axes. Inner subplots hide their tick labels by themselves, as
`plt.subplots(sharex=True)`, and give the room back; frames along a shared axis stay aligned. There is no
unsharing. Sharing an axes of another figure is a `ValueError`, as is putting a heatmap on a log-scaled group.

### Twin axes

```python
host = fig.axes()
host.line(t, temperature).set_ytitle("temperature [°C]")
twin = host.twinx()                     # shares x, its own y on the right
twin.line(t, rain).set_ytitle("rain [mm/h]")
host.set_axes_style(yaxis_color="red")
twin.set_axes_style(yaxis_color="blue")
```

`twinx()` returns a second axes over the same frame that shares the x axis and has its own y axis on the right;
`twiny()` the same with the axes swapped. It is the same object on every call. Plot into it as into any axes;
the objects of both draw in the order they were plotted, the twin continues the host's colour cycle, and pan,
zoom and reset move both.

The cell's title, legend (which lists both axes' series), colorbars and shared axis stay the host's: on a
`twinx()`, `set_title()`, `legend()`, `set_xtitle()`, `set_xticks()` and `set_xtick_format()` act on the host.
`yaxis_color` colours each y axis' frame edge, so an axis can take its series' colour. Events report the twin's
coordinate as `Event.y2data` (`x2data` for a `twiny()`), and a pick names whose object it found in
`Event.pick_twin`. In the window, a menu-bar selector next to the subplot selector picks which layer the panels
edit.

### Equal aspect

`set_aspect(ratio, adjustable="box")` draws one y unit `ratio` x units long, as matplotlib's: `1.0` keeps a
circle round, `0` (the default) fills the frame. Units are the scales', so on log-log a decade is as long on
both axes.

```python
ax.line(cx, cy, loop=True).set_aspect(1.0)
```

`"box"` (the default) shrinks the frame, centred, with its ticks, titles, legend and colorbars; `"datalim"`
keeps the frame and widens one axis' limits. Pan and zoom keep the ratio. `aspect()` and `adjustable()` read
them back; `cla()` resets them. A negative or non-finite ratio is a `ValueError`, as is `set_aspect()` on a twin
(its host's aspect sets the cell).

---

## Text and annotations

```python
ax.text(s, x, y, *, coords="data", xcoords=None, ycoords=None, **TextOptions) -> Axes
ax.annotate(px, py, s, tx, ty, *, coords="data", xcoords=None, ycoords=None,
            arrow=None, **TextOptions) -> Axes
```

`text()` writes a string over the plot; `annotate()` writes one with an arrow to the data point `(px, py)`.
Each coordinate of the text's position is in data units or, with `"fraction"`, a fraction of the plot frame
from its left (x) or bottom (y) edge. `coords` sets both; `xcoords` and `ycoords` set one each, so a label
can follow the data on one axis and stay put on the other:

```python
(ax.text("threshold", 2.5, 0.95, ycoords="fraction", va="top")
   .text("n = 120", 0.02, 0.04, coords="fraction")
   .annotate(t_peak, x_peak, "first peak", 0.6, 0.8, coords="fraction",
             fontsize=14, background="white", edgecolor="gray", edge_linewidth=1,
             arrow={"arc": 0.2}))
```

The keywords ([TextOptions](reference.md#textoptions)) set the look: `fontsize` (pixels), `color`, `alpha`,
`font_path`; `ha`/`va`, how the block sits against its anchor (`va="baseline"` by default); `rotation` in
degrees; `dx`/`dy`, a nudge in pixels (y up); multi-line text (`"\n"`) with `linespacing`; and a box,
`background` filled and `edge_linewidth` outlined, `pad` pixels out. `arrow` is a dict of
[ArrowOptions](reference.md#arrowoptions): `head` and `tail` (`"none"`, `"open"`, `"filled"`, `"bar"`), their
size, `linewidth`, `color` (unset: the text's), `linestyle`, the gaps left at the text and the point, and
`arc`, which bows the arrow (as matplotlib's `arc3`).

Texts are drawn over the data at a fixed pixel size: they never widen the automatic limits and reserve no
room. A text whose **data** coordinate leaves the view is hidden whole; `clip_to_frame=True` cuts it at the
frame instead. An annotation is hidden while its point is out of view. A non-finite coordinate is a
`ValueError`.

Texts appear in the window, PNG and SVG, have a tab in the Data panel, and read back like plot objects:
`text_count()`, `text_data(i)` (a `TextData`: the string, both coordinates and their `Coords`, the arrow and
its point) and `set_text_data(i, ...)`, either with a `TextData` or as `set_text_data(i, s, x, y)`, which keeps
each coordinate's coords and an annotation's arrow unless told otherwise. 3D has `text()`, `text2d()` and
`annotate()` of its own; see [3d.md](3d.md#text-and-annotations).

---

## Math in text

Every string sextant draws can hold math: titles, axis titles, the suptitle, legend and colorbar names,
custom tick labels and texts, in 2D and 3D. Write it between dollars, as in matplotlib (a raw string keeps the
backslashes):

```python
ax.set_xtitle(r"$\theta$ [rad]")
ax.set_title(r"Damped oscillator: $x(t) = e^{-\gamma t}\sin(\omega t)$")
```

It is a large subset of matplotlib's mathtext, laid out by sextant itself:

- superscripts and subscripts, `x^2`, `x_i`, `x_{i,j}^{2}`, nestable;
- Greek letters and about 150 symbols by name: `\alpha`, `\Delta`, `\pm`, `\times`, `\leq`, `\neq`,
  `\approx`, `\infty`, `\nabla`, `\rightarrow`, ...;
- function names set upright, `\sin`, `\cos`, `\log`, `\exp`, `\max`, ...;
- **fractions and roots**: `\frac{a}{b}` (also `\tfrac`, `\dfrac`), `\binom{n}{k}`, `\sqrt{x}`, `\sqrt[n]{x}`,
  `\overline{}`, `\underline{}`;
- **accents**: `\hat`, `\bar`, `\tilde`, `\dot`, `\ddot`, `\acute`, `\grave`, `\breve`, `\check`, `\mathring`,
  `\vec`, and over a whole body `\widehat`, `\widetilde`, `\overrightarrow`, `\overleftarrow`;
- **sized delimiters**: `\left( … \middle| … \right)` grow to their contents; `\big`, `\Big`, `\bigg`, `\Bigg`
  give a fixed size;
- **big operators with limits**: `\sum`, `\prod`, `\int`, `\iint`, `\oint`, `\bigcup`, … draw larger, and
  `\lim`, `\max`, `\sup` and the like take limits above and below; also `\overset`, `\underset`, `\stackrel`,
  `\substack`, `\operatorname`;
- TeX spacing (`\,`, `\;`, `\quad`, ...) and the spacing around operators;
- fonts: `\mathrm{}` (upright), `\mathit{}` (italic), `\text{}` (upright, spaces kept), `\mathbf{}` (bold),
  and the alphabets `\mathcal{}`, `\mathscr{}`, `\mathbb{}`, `\mathfrak{}`, `\mathsf{}`, `\mathtt{}`.

```python
ax.set_title(r"$\frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$")
ax.text(r"$\int_0^\infty e^{-x^2}\,dx = \frac{\sqrt{\pi}}{2}$", 0.5, 0.5, coords="fraction")
```

Letters draw italic, as in TeX; digits, symbols, Greek capitals and function names stay upright. Symbols the
font lacks come from a system symbol font. The SVG writes math as `<tspan>`s, so it stays text.

**Math font sets.** `sextant.Figure(math_fontset=...)` chooses what the math is drawn in: `"text"` (the
default: the text font, as before), `"stix"`, `"cm"` (Computer Modern), or `"custom"` with
`math_font_path="/path/to/font.otf"`. They use fonts installed on the machine (none is bundled): with none
found the text font is used, and creating the figure issues a `RuntimeWarning`. matplotlib's names
`"stixsans"`, `"dejavusans"` and `"dejavuserif"` are accepted too.

**When a string is math.** As in matplotlib: when it has an even, nonzero number of `$` not preceded by a
backslash. `\$` is a literal dollar, and a string with one `$` or none draws as written. Math that does not
parse (an unknown command, an unclosed brace) is drawn as written, and the call that set it issues a
`RuntimeWarning` (see [Errors and warnings](#errors-and-warnings)).

Turn it off for a whole figure with `sextant.Figure(mathtext=False)`, or for one text with
`parse_math=False`. Automatic tick labels, contour labels and hover text are never math. Not supported:
`\boldsymbol` and `\color` (drawn as written, with a warning), and `usetex`.

---

## Styling

Four calls cover an axes' appearance, each taking the keywords of one options struct:

```python
ax.set_axes_style(spine_color="gray", spine_top=False, spine_right=False,
                  label_fontsize=13, font_path="C:/Windows/Fonts/times.ttf")
ax.grid(True, color=(0.85, 0.85, 0.85), linestyle="dashed")
ax.legend(fontsize=12, frameon=True)
ax.set_colorbar_style(anchor="bottom")
fig.set_suptitle_style(fontsize=26, align="left")
```

`font_path` is an absolute path to a `.ttf`, `.ttc` or `.otf` file. Left empty, sextant uses a font it finds
among the system fonts. The legend, colorbars and suptitle each have their own `font_path` and do not inherit
the axes'.

**Where the axes sit.** By default the x axis runs along the bottom of the frame and the y axis up its left
side. `xaxis_y` and `yaxis_x` move them (`"low"`, `"mid"`, `"high"`), and `origin_x`/`origin_y` put them
through a data value: `set_axes_style(origin_x=0, origin_y=0)` draws axes crossing at the origin. The four
`spine_*` flags turn the frame's edges on and off.

**Backgrounds.** `sextant.Figure(background=...)` (or `fig.set_background(color)`) fills behind the whole
figure: margins, gaps and the suptitle, light gray by default. `set_axes_style(background=...)` fills a 2D plot
area, white by default (a 3D box colours its panes with `set_box_style(pane_color=...)`). Alpha 0 leaves a PNG
or SVG transparent there; the window shows its panel colour instead.

**Hiding ticks.** `set_axes_style(show_xticks=False)` (and `show_yticks`, and `show_zticks` in 3D) drops that
axis' tick marks and labels and gives their room back to the plot; grid lines stay.

```python
fig = sextant.Figure(background="#dde6ef")
fig.axes().set_axes_style(background="#1e1f29", show_xticks=False)
```

The window's side panels (not the plot) take a theme when the figure is made:
`sextant.Figure(theme="dark", panel_width=300)`; `"light"` is the default, `"classic"` the third.

---

## Layout

### Subplots and spans

```python
fig = sextant.Figure(width=1000, height=620, subplot_col_gap=10, subplot_row_gap=10)

fig.add_subplot(2, 3, (1, 3)).line(x, y)     # the whole top row
fig.add_subplot(4).hist(n, 30)               # the grid shape is known by now
fig.add_subplot(5).bar(bx, bh)
fig.add_subplot(6).heatmap(f, (-2, 2), (-2, 2))
```

Cells are numbered `1 .. rows*cols`, row by row, as in matplotlib. A span `(first, last)` covers the rectangle
from its top-left cell to its bottom-right one.

A figure has **one grid shape**, fixed by the first call that gives one (`axes()` counts as 1×1). After that,
`add_subplot(index)` and `add_subplot((first, last))` use it, and a different shape is a `ValueError`. Asking
again for the same cell or span returns the axes already there; a request overlapping an occupied cell is a
`ValueError`. A span is addressed later by its first cell.

`add_subplot3d()` puts an `Axes3D` in a cell under the same rules, so 2D and 3D subplots share a grid; see
[3d.md](3d.md).

**Column and row weights** share the grid out unevenly:

```python
fig.set_col_ratios([2, 1, 1])     # the first column twice as wide
fig.set_row_ratios([1, 2])
fig.set_col_ratios([])            # equal again
```

A weight sizes the whole cell, decorations included. In the window, the boundary between two columns or rows
can be dragged, and a double-click evens it out.

`set_margins(left=, right=, top=, bottom=)` sets the border between the figure edge and the grid (10 pixels
each by default).

The space for titles, tick labels, legends and colorbars is measured from their text, so there is no
`tight_layout()` to call.

### Sizing a figure

Sizes in sextant (`width`/`height`, font sizes, line widths, marker sizes, margins) are **logical pixels**,
1/96 inch. The window draws them at the display's scale, so a figure looks the same on every monitor.

`width`/`height` (default 800×600) and `resize(w, h)` set the **plot area**, which is what `savefig()`
writes. An open window grows by its menu bar and panels, so the plot keeps the size you asked for.

A PNG has `dpi / 96` pixels per logical pixel: at the default `dpi=96`, an 800×600 figure is an 800×600 file;
at `dpi=192` it is 1600×1200 with the same layout.

To size from the data area instead:

```python
w, h = fig.size_for_frame(400, 300)          # figure size giving a 400x300 data frame
fig.resize_to_frame(400, 300)                # ...and apply it
fig.resize_to_frame(400, 300, slot_index=2)  # for subplot 2's frame
```

---

## Saving and rendering

```python
fig.savefig("plot.png")
fig.savefig("plot.svg")
```

**No window is needed**: both formats work before `show()`, after the window has closed, or on a machine with
no display. SVG needs no OpenGL at all.

```python
savefig(fname, *, format=None, width=0, height=0, **options)
```

- `fname` is a path (`str` or `pathlib.Path`) or a file object with a `write()` method. On Windows a path may
  not contain `:` other than a drive letter's (NTFS would read it as a hidden data stream): `OSError`.
- `format` is `"png"` or `"svg"`. Without it, a path's extension decides (anything else is a `ValueError`),
  and a file object gets PNG.
- `width`/`height` render at another size; `0` uses the figure's.
- The keyword options are the format's own: `dpi` and `peel_layers` for PNG, `max_splits` and `max_tests` for
  SVG ([reference](reference.md#pngexportoptions)). An option of the other format is a `TypeError`.
- For SVG it returns an `SvgSaveReport`; for PNG, `None`.

```python
fig.savefig("plot@2x.png", dpi=192)        # twice the pixels, same layout
fig.savefig("small.png", width=640, height=400)

buf = io.BytesIO()
fig.savefig(buf)                            # PNG bytes into memory
with open("plot.svg", "w", encoding="utf-8") as f:
    fig.savefig(f, format="svg")            # a text file gets str
```

PNG is supersampled (`Figure(supersample=2)` by default; 1 to 4) and matches the figure size exactly. SVG is
vector, 3D scenes included.

Without touching a file:

| Call | Returns |
|---|---|
| `fig.render_png(*, width=0, height=0, **PngExportOptions)` | The PNG file as `bytes` |
| `fig.render_svg(*, width=0, height=0, **SvgExportOptions)` | `(svg_text, SvgSaveReport)` |
| `fig.render_rgba(*, width=0, height=0, **PngExportOptions)` | A `(height, width, 4)` `uint8` array, top row first |

`render_rgba()` hands the pixels to numpy, Pillow (`Image.fromarray(a)`), OpenCV (after RGBA→BGRA) or a video
writer without encoding a PNG.

**SvgSaveReport.** A 3D scene whose translucent objects pass through each other is sorted polygon by polygon
for SVG, within a work bound. If the bound is hit the file is still written, with part of it in plain depth
order: `scene_order_exact` is then `False`, `warning` says which bound and the number to beat, and a
`RuntimeWarning` is issued. See [3d.md](3d.md#translucency-and-depth-order).

---

## Jupyter and IPython display

A `Figure` displays itself as the value of a notebook cell:

```python
fig = sextant.Figure(width=600, height=350)
fig.axes().line(x, np.sin(x))
fig            # the cell shows the PNG
```

`sextant.set_repr_formats("png")` is the default. `set_repr_formats("svg")`, `("png", "svg")` or none at all
(`set_repr_formats()`) choose what a figure offers; SVG is opt-in because ordering a large 3D scene for SVG can
be slow on every display.

A notebook can also open real windows with `fig.show()`; see [Interactive sessions](#interactive-sessions).

---

## The window

`fig.show()` opens a window and renders the figure there, on a thread of its own.

### show() and waiting

```python
fig.show(block=None)
```

| `block` | Behaviour |
|---|---|
| `None` (default) | In a script: wait until the window is closed. In an interactive session: return at once |
| `True` | Wait until the window is closed |
| `False` | Return at once; the window stays open |

The window stays open until the user closes it or you call `fig.close()`, even if no variable refers to the
figure any more: `show(False)` inside a function leaves its window open, callbacks and all, after the function
returns, as in matplotlib. At interpreter exit, every window is closed.

Other ways to wait:

| Call | |
|---|---|
| `fig.wait_closed(timeout=None)` | Wait until this window is closed (`True`) or `timeout` seconds pass (`False`) |
| `sextant.run()` | Wait until every open figure is closed |
| `fig.is_open()` | Whether the window is open now |
| `sextant.poll_events()` | Process window events once, without waiting; see [Live updates](#live-updates) |

All waits answer Ctrl+C within a tenth of a second (`KeyboardInterrupt`).

Several windows at once:

```python
a = sextant.Figure(title="raw");      a.axes().line(raw)
b = sextant.Figure(title="filtered"); b.axes().line(smooth)
a.show(False)
b.show(False)
sextant.run()                         # until both are closed
```

A `Figure` is also a context manager that closes its window on exit:

```python
with sextant.Figure() as fig:
    fig.axes().line(y)
    fig.show(False)
    ...                               # the window closes when the block ends
```

### Interactive sessions

At a prompt (the plain REPL, `python -i`, IPython, Jupyter), `show()` returns immediately and the window keeps
working between statements: you can pan and zoom it while typing the next line, and changes you make reach it
with `fig.refresh()`.

- **REPL and `python -i`**: sextant pumps the window while the prompt waits for input.
- **IPython in a terminal**: sextant registers an event-loop hook (`%gui sextant`) on the first `show()`,
  unless another GUI loop (Qt, Tk) is active; then it leaves that alone and the window responds during
  sextant's own waits.
- **Jupyter**: windows open on the machine running the kernel. The first `show()` starts a sextant event loop
  in the kernel (`%gui sextant`), unless another GUI loop is active: the window responds, on macOS too, and
  callbacks connected with `connect()` run between cells, without `wait_closed()` or `poll_events()`. Their
  printed output appears under the last cell run. The loop stops by itself when the last window closes;
  `%gui sextant` starts it by hand (it then stays on until `%gui`).

### What the window offers

The window has a **Cosmetic** panel and a **Data** panel docked beside the plot, and menus:

| | |
|---|---|
| **File → Save** | PNG or SVG (by the name's `.png`/`.svg`) at a chosen size, or a chosen plot-frame size. A failed save says why in the window and is issued as a `RuntimeWarning` |
| **File → Resize to plot frame** | Resize the window so a plot area hits a target size |
| **File → Refit layout** | Re-measure tick labels after panning has widened them |
| **View** | Show or hide either panel |
| **Edit → Navigate** | Pan, zoom, box-zoom (2D), orbit and fly (3D) with the mouse and keys |
| **Edit → Back / Forward / Home** | Step through the views: pans, zooms, boxes and 3D camera moves |
| **Edit → Hints** | Hover for values (on by default) |

Click a subplot to select it; the panels edit the selected subplot. In 2D, with Navigate on, left-drag pans,
the wheel zooms at the cursor, and Shift+drag draws a box to zoom to (a thin box zooms one axis, Esc cancels;
with an aspect ratio the box keeps the frame's shape). On a log axis pan and zoom move whole decades. 3D
navigation is in [3d.md](3d.md#camera).

**Back, Forward and Home** step through the views. Home is the limits and camera *your program set*
(automatic where it set none), and a **double-click goes Home** for its subplot, so it returns to your
`set_xlim()` or `set_view()` rather than to automatic limits; a 3D axes with `set_default_camera()` still
returns to that camera.

**Hints** read out the values under the cursor. Give points extra hover text with `hint_labels`, one string
per point:

```python
ax.line(x, y, hint_labels=["", "", "Peak", "", "Trough"])
```

The **Cosmetic** panel edits titles, limits, ticks, grid, frame, legend, colorbars, suptitle, margins, gaps and
grid weights, live. The **Data** panel has a tab per plot object with its appearance and its data as an
editable table.

---

## Live updates

The window renders from a snapshot of the figure. Change the figure from Python, then publish the change with
`fig.refresh()`:

```python
fig = sextant.Figure()
ax = fig.axes()
ax.line(x, y, name="u(t)").set_title("simulation")
fig.show(block=False)

while not fig.wait_closed(1 / 60):     # until the user closes the window
    y = step(y)
    ax.set_line_data(0, x, y)          # replace series 0's data, keep everything else
    fig.refresh()                      # publish
```

**`set_<kind>_data(i, ...)` replaces object `i`'s data** and keeps its options, the axes' titles, limits and
styles, and anything edited in the window. It exists for every kind: `set_line_data`, `set_scatter_data`,
`set_scatter_z_data`, `set_bar_data`, `set_heatmap_data`, `set_fill_data`, `set_quiver_data`, and the 3D ones. It takes either the plotting call's
own data arguments or the object `<kind>_data(i)` returns:

```python
ax.set_line_data(0, x, y)
ax.set_scatter_z_data(1, x, y, z)
ax.set_heatmap_data(0, field, (0, 1), (0, 1))

d = ax.line_data(0)
d.y *= 2
ax.set_line_data(0, d)
```

`i` counts within the kind, in plotting order (the first `scatter()` is scatter 0 whatever came before it);
negative indices count from the end. The new data is checked as the plotting call would check it, and a bad
one changes nothing. Error bars and `hint_labels` stay while the point count is unchanged and are dropped
otherwise.

**Hiding an object** keeps it and its data: `set_<kind>_visible(i, False)` (or `visible=False` when plotting)
hides it, `<kind>_visible(i)` reads it. A hidden object is not drawn or exported, has no legend key or colorbar,
takes no hover or pick, and leaves the automatic limits; its colour and draw order stay, so showing it again
puts it back as it was. It exists for every kind, texts included (`set_text_visible`), and the Data panel has a
Visible checkbox per object.

```python
ax.set_line_visible(1, False)
fig.refresh()
```

**Rules:**

- `refresh()` is a `RuntimeError` before `show()` or after the window has closed. Check `fig.is_open()` if you
  are not sure.
- You do not need `refresh()` before `savefig()`: saving uses the figure as you last changed it.
- **Edits made in the window survive `refresh()`**: typed titles, a panned view, retyped data. When you and
  the user change the same thing, the later change wins. `cla()` resets everything, window edits included.
- **macOS**: windows only respond while the main thread is inside sextant. In a loop on the main thread, wait
  with `wait_closed(timeout)` as above, or call `sextant.poll_events()` every iteration (it does nothing on
  Windows and Linux, so the loop stays portable). `poll_events()` raises `RuntimeError` off the main thread on
  macOS.

`fig.frame_stats()` reports render timing since `show()`: `frames`, `total_ms` (render work, excluding
vsync waits), `last_ms`, `max_ms`. Make the figure with `vsync=False` to measure the full cost.

---

## Events

`fig.connect(kind, callback)` calls `callback(event)` for each event of a kind and returns an id for
`fig.disconnect(id)`:

```python
def on_click(e):
    if e.inaxes is ax and e.has_data:
        print(f"clicked ({e.xdata:.3g}, {e.ydata:.3g}) with {e.modifiers}")

cid = fig.connect("mouse_down", on_click)
fig.connect("key_down", lambda e: e.key == "escape" and fig.close())
fig.show()
```

| `kind` | matplotlib name (also accepted) | When |
|---|---|---|
| `"close"` | `"close_event"` | The window closed |
| `"mouse_down"`, `"mouse_up"` | `"button_press_event"`, `"button_release_event"` | A mouse button |
| `"mouse_move"` | `"motion_notify_event"` | The pointer moved over the plot |
| `"scroll"` | `"scroll_event"` | The wheel |
| `"key_down"`, `"key_up"` | `"key_press_event"`, `"key_release_event"` | A key |
| `"resize"` | `"resize_event"` | The plot area changed size |
| `"pick"` | `"pick_event"` | A click on a plot object (a point, bar, cell, …) |

The `Event` fields are listed in [reference.md](reference.md#event). The useful ones:

- `x`, `y`: pixels from the top left of the plot area; `inaxes`: the `Axes` or `Axes3D` under the pointer (the
  same object `add_subplot()` returned), or `None`;
- `has_data`, `xdata`, `ydata` (`zdata`): the pointer in data coordinates, NaN when it is not over data;
- `button` (0 left, 1 right, 2 middle), `double_click`, `modifiers` (`("ctrl", "shift")`, …), `key`
  (`"a"`, `"ctrl+a"`, `"escape"`, `"f5"`, `"left"`, …), `scroll_y`, `width`/`height` for resize;
- `x2data`/`y2data`: over an axes with a `twiny()` / `twinx()`, the twin's coordinate under the pointer (NaN
  without one); `xdata`/`ydata` stay the host's;
- for a pick: `pick_kind` (`"line"`, `"scatter"`, `"bar"`, `"heatmap"`, `"quiver"`, `"fill"`, …), `pick_object`
  (the `i` of `<kind>_data(i)`), `pick_index` (the point, bar or vertex; a fill's segment *i*, between its points
  *i* and *i* + 1), `pick_row`/`pick_col` (a heatmap cell), `pick_plane` (in 3D), `pick_twin` (0 the axes' own,
  1 its `twinx()`, 2 its `twiny()`).

```python
def on_pick(e):
    if e.pick_kind == "scatter":
        d = e.inaxes.scatter_data(e.pick_object)
        print("picked point", e.pick_index, d.x[e.pick_index], d.y[e.pick_index])

fig.connect("pick", on_pick)
```

**What a press picks.** A point, marker, arrow or path vertex within the **pick radius** answers first, the
nearest (`Figure(pick_radius=12)` by default, or `fig.set_pick_radius(px)`); otherwise the topmost bar, fill or
heatmap cell the press is *inside*, whose position is the event's `xdata`/`ydata`. Reference lines, spans and
texts are decorations and are never picked.

`consumed` says what sextant itself did with the input (`"navigate"` for a pan, `"select"` for selecting a
subplot, …); your callback still runs.

**Where callbacks run.** Events are queued by the window and delivered on your thread, while it is inside
`show()` (blocking), `wait_closed()`, `run()`, `poll_events()` or `fig.dispatch_events()`, or waiting at an
interactive prompt. A script that never waits receives no events. A callback can change the figure and call
`refresh()`.

An exception in a callback is reported through `sys.unraisablehook` (printed, by default) and delivery
continues. `KeyboardInterrupt` is raised from the wait that was delivering. A callback lives as long as the
`Figure` object it was connected through.

---

## Reading back

`Axes` (and `Plane2D`, `Axes3D`) report what they hold now, including edits made in the window:

```python
ax.title(); ax.xtitle(); ax.ytitle()
ax.xlim()                      # (lo, hi) as drawn: set, panned, or automatic
ax.xscale(); ax.aspect()       # the scale and the aspect ratio
ax.line_count()
d = ax.line_data(0)            # LineData(x=array(...), y=array(...))
```

Each kind has `<kind>_count()` and `<kind>_data(i)`, returning an object named after the kind (`LineData`,
`ScatterData`, `ScatterZData`, `BarData`, `HeatmapData`, `FillData`, `QuiverData`, `AxLineData`,
`AxSpanData`; in 3D `Bar3DData`, `SurfaceData`, `SurfaceTriData`,
`Scatter3DData`, `Line3DData`) whose attributes are numpy arrays named after the plotting call's arguments.
Texts read back the same way: `text_count()`, `text_data(i)` (`TextData`, in 3D `Text3DData`).
`bar_data` covers `hist()` (bin centres and heights) and `heatmap_data` covers `imshow()`; `HeatmapData.data`
is `(rows, cols)`, rounded to single precision. `ScatterData.sizes` and `BarData.bottom` are `None` when the
whole series shares one `size` or stands on 0; a `stack()` reads back as one `FillData` per layer. Styles, error bars and `hint_labels` are not read back.

The arrays are yours: change them in place and pass the object back to `set_<kind>_data(i, d)`. The data
classes can also be built directly: `sextant.LineData(x, y)`.

---

## Threads

sextant objects can be used from any Python thread; calls on one figure are serialized. Waiting
(`show()`, `wait_closed()`, `run()`) releases the GIL, so other Python threads run meanwhile, and one thread
can update a figure while another waits on it:

```python
import threading

def worker():
    while fig.is_open():
        ax.set_line_data(0, x, acquire())
        fig.refresh()

fig.show(block=False)
threading.Thread(target=worker, daemon=True).start()
fig.wait_closed()
```

On macOS, window events are processed only on the main thread, so keep `show()` and the waiting there and let
other threads do the updating; `poll_events()` raises `RuntimeError` on any other thread. Callbacks run on the thread that delivers them (see [Events](#events)).

---

## Errors and warnings

| Raised | When |
|---|---|
| `TypeError` | An unknown keyword option (the message lists the valid ones); an option or argument of the wrong type, including an unknown enum name or colour; an array of the wrong dimensionality |
| `ValueError` | Mismatched lengths; a non-finite text coordinate; an error-bar array that is not one per point; an empty or non-finite heatmap range; a bad subplot index, a second grid shape, or an occupied cell; a non-finite reference line; a log scale on a
heatmap's axes; a bad tick format; a negative aspect; a non-positive `resize()` size or dpi; an unknown `savefig` format; a 2-D grid of the wrong shape; for 3D see [3d.md](3d.md#errors) |
| `IndexError` | A `<kind>_data(i)` or `set_<kind>_data(i, …)` index out of range |
| `RuntimeError` | `refresh()` before `show()` or after the window closed; `poll_events()` off the main thread on macOS; a window that cannot be created (on Linux without an X display, with a hint about `DISPLAY`) |
| `OSError` | Writing a file failed (`FileNotFoundError`, `PermissionError`, … as the cause dictates); on Windows, a file name with `:` other than a drive letter's |

A call that raises changes nothing.

**Warnings.** Problems that do not stop a call (no usable font found, math that does not parse, an SVG export
that hit its work bound) are issued as `RuntimeWarning`, so the `warnings` module controls them:

```python
import warnings
warnings.simplefilter("error", RuntimeWarning)       # make them exceptions, or:
sextant.set_message_handler(lambda msg: log.info(msg))   # route them elsewhere
sextant.set_message_handler(None)                     # back to warnings
```

The handler may be called from a window thread; an exception from it is reported through
`sys.unraisablehook`.

---

## Lifetime

- A figure lives while any Python object refers to it: the `Figure`, or one of its `Axes`, `Axes3D` or
  `Plane2D`.
- A shown figure also lives while its window is open: sextant holds it, with its callbacks, until the window
  is closed by the user or by `fig.close()`, then lets it go.
- `fig.close()` closes the window but keeps the figure: it can still be saved, changed and shown again.
- At interpreter exit, sextant closes every window first, so a script may end with windows open.
