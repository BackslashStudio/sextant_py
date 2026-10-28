# sextant for Python: reference

Every keyword option, enum, data class and event field of the core API, with defaults. For how they are
used, see the [API guide](api.md) and [3D plots](3d.md).

Sizes are logical pixels (1/96 inch) unless a row says otherwise. *colour* means a colour as described in
[api.md](api.md#colours): a name, `"#rrggbb[aa]"`, or a 3- or 4-tuple of floats in [0, 1]; colour defaults are
written as `(r, g, b)` tuples. Named colours are matplotlib's tab10 shades: `"blue"` is `#1f77b4`,
`"orange"` `#ff7f0e`, and so on. A `color` default of `None` means the axes' next colour from the
[colour cycle](api.md#colour-cycle). Enum values are strings ([Enums](#enums)). *array* means anything
`numpy.asarray(x, float)` accepts.

- [Figure and output](#figure-and-output): [Figure()](#figure-options) · [margins](#figuremargins) ·
  [suptitle style](#suptitleoptions) · [PNG export](#pngexportoptions) · [SVG export](#svgexportoptions) ·
  [SvgSaveReport](#svgsavereport) · [FrameStats](#framestats)
- [2D plot options](#2d-plot-options): [line](#lineoptions) · [scatter](#scatteroptions) ·
  [scatter_z](#scatterzoptions) · [bar and hist](#baroptions) · [heatmap and imshow](#heatmapoptions) ·
  [imshow_rgb](#imageoptions) · [heatmap_rec](#heatmaprecoptions) · [heatmap_tri](#heatmaptrioptions) ·
  [contour and contourf](#contouroptions) · [hist2d](#hist2doptions) · [ColorNorm](#colornorm) ·
  [fill](#filloptions) · [stack](#stackoptions) · [quiver](#quiveroptions) · [axline](#axlineoptions) ·
  [axspan](#axspanoptions) · [shapes](#polyoptions) · [pie](#pieoptions) · [ErrorBar](#errorbar) ·
  [errorbar style](#errorbaroptions)
- [Decoration and style](#decoration-and-style): [set_axes_style](#axesstyle) · [scales](#scaleoptions) ·
  [grid](#gridoptions) · [legend](#legendoptions) · [set_colorbar_style](#colorbaroptions)
- [Text](#text): [TextOptions](#textoptions) · [ArrowOptions](#arrowoptions)
- [3D options](#3d-options): [camera](#camera3d) · [box style](#box3dstyle) · [plane](#plane2doptions) ·
  [bar3d](#bar3doptions) · [surface](#surfaceoptions) · [surface_tri](#surfacetrioptions) ·
  [scatter3d](#scatter3doptions) · [line3d](#line3doptions) · [quiver3d](#quiver3doptions) · [ErrorBar3D](#errorbar3d) ·
  [3D errorbar style](#errorbar3doptions)
- [Enums](#enums)
- [Data classes](#data-classes)
- [Event](#event)
- [Module functions](#module-functions)

---

## Figure and output

### Figure options

`sextant.Figure(**options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `width`, `height` | int | `800`, `600` | Size of the plot area, which is what `savefig()` writes. A window adds its menu bar and panels |
| `title` | str | `"sextant"` | Window title bar |
| `resizable` | bool | `True` | Whether the window can be resized |
| `dpi` | float | `96` | PNG resolution: `dpi / 96` output pixels per logical pixel |
| `subplot_col_gap`, `subplot_row_gap` | float | `0` | Space between subplot cells (a cell includes its decorations) |
| `margins` | dict | 10 each | [FigureMargins](#figuremargins); also `set_margins()` |
| `background` | colour | `(0.93, 0.93, 0.93)` | Fill behind everything (margins, gaps, suptitle); alpha 0 is transparent in PNG and SVG. Also `set_background()` |
| `mathtext` | bool | `True` | Draw `$...$` spans in every string as math; `False` draws every string as written |
| `math_fontset` | `MathFontset` | `"text"` | The font math draws in, from the fonts installed: `"text"` the text's own, `"stix"` (STIX Two Math, STIX Two Text, STIXGeneral), `"cm"` (Latin Modern Math, CMU Serif, Latin Modern Roman), `"custom"` the file at `math_font_path`. None installed: the text font, and a `RuntimeWarning`. Fixed when the figure is made |
| `math_font_path` | str | `""` | The font file for `math_fontset="custom"` |
| `pick_radius` | float | `12` | How near, in pixels, the cursor must come to a point, marker, arrow or path vertex for hover and picks; bars, fills and heatmaps answer inside them. Finite and >= 0; also `set_pick_radius()` |
| `panel_width` | float | `240` | Initial width of the window's side panels. Never exported |
| `supersample` | int | `2` | Supersampling for the window and PNG, 1..4; 1 disables. Cost is quadratic. No effect on SVG |
| `vsync` | bool | `True` | Cap rendering at the display refresh rate. Turn off only to measure |
| `theme` | `PanelTheme` | `"light"` | Look of the window's panels (not the plot) |

### FigureMargins

`margins={...}` in `Figure()`, or `fig.set_margins(**margins)` (unnamed sides reset to 10).

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `left`, `right`, `top`, `bottom` | float | `10` each | Space between the figure edge and the subplot grid |

### SuptitleOptions

`fig.set_suptitle_style(**options)`; the text is `fig.suptitle(text, fontsize=21)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `fontsize` | float | `21` | Also set by `suptitle(text, fontsize)` |
| `color` | colour | `(0.1, 0.1, 0.1)` | |
| `font_path` | str | `""` | Absolute font file path; empty = default font |
| `align` | `HAlign` | `"center"` | `"left"`/`"right"` anchor to the figure edge |
| `offset_x`, `offset_y` | float | `0` | Nudge as drawn; does not enlarge the title band |

### PngExportOptions

Keywords of `savefig()` for PNG, `render_png()` and `render_rgba()`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `peel_layers` | int | `0` (= 8) | Depth-peeling layers for translucent 3D, 1..64 |
| `dpi` | float | `0` (= the figure's `dpi`) | Resolution of this file; same layout at any dpi |

### SvgExportOptions

Keywords of `savefig()` for SVG and `render_svg()`. They only matter for 3D scenes with interpenetrating
translucent geometry.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `max_splits` | int | `0` (= 8 × polygons + 64) | Polygon splits allowed; the bound that binds in practice |
| `max_tests` | int | `0` (= 20,000,000) | Pairwise comparisons allowed; the wall-clock bound |

### SvgSaveReport

Returned by `savefig()` for SVG, and by `render_svg()` beside the text. Read-only.

| Attribute | Type | Meaning |
|---|---|---|
| `scene_order_exact` | bool | `False` if a bound was hit and part of the file is in plain depth order |
| `splits`, `tests` | int | Work done |
| `warning` | str | Which bound was hit and the number to beat; empty when exact |

### FrameStats

Returned by `fig.frame_stats()`; cumulative since `show()`. Read-only.

| Attribute | Type | Meaning |
|---|---|---|
| `frames` | int | Frames rendered |
| `total_ms` | float | Summed render work, excluding the vsync wait |
| `last_ms` | float | The most recent frame |
| `max_ms` | float | The worst single frame |

---

## 2D plot options

These are also the options of the same calls on a `Plane2D`.

### LineOptions

`line(x, y=None, *, err=None, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour or `None` | `None` | `None` = the axes' next [colour-cycle](api.md#colour-cycle) colour, taken at the call |
| `linewidth` | float | `1.5` | |
| `linestyle` | `LineStyle` | `"solid"` | `"none"` draws no stroke and drops the series from the legend |
| `name` | str | `""` | Legend key |
| `show_legend` | bool | `True` | Keyed when this is true and `name` is not empty |
| `visible` | bool | `True` | `False` hides the object as if absent: not drawn or exported, no legend key, colorbar, hover or pick, and out of the automatic limits. Its data stays. Also `set_line_visible()` and the like |
| `alpha` | float | `1` | |
| `loop` | bool | `False` | Add a closing segment from the last point to the first |
| `drawstyle` | `DrawStyle` | `"default"` | `"steps_pre"`, `"steps_mid"`, `"steps_post"` draw the line as steps |
| `errorbar` | dict | | [ErrorBarOptions](#errorbaroptions): style of the error bars passed as `err` |
| `zorder` | float | `0` | Draw order: objects draw in ascending `zorder`, equal ones in call order. Finite. Also `set_zorder(id, z)` |
| `hint_labels` | list of str | `[]` | Extra hover text per point |

### ScatterOptions

`scatter(x, y, *, err=None, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour or `None` | `None` | As in [LineOptions](#lineoptions) |
| `size` | float | `20` | Marker diameter; constant under zoom |
| `sizes` | array or `None` | `None` | One diameter per point (a bubble chart); `None` = `size`. A NaN or non-positive size hides the point. Kept as data: `ScatterData.sizes` |
| `marker` | `MarkerStyle` | `"circle"` | |
| `name` | str | `""` | Legend key |
| `show_legend` | bool | `True` | |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `alpha` | float | `0.8` | Multiplies the own alpha of every colour of the series: fill, outline and error bars; 0 hides it |
| `filled` | bool | `True` | `False` draws the outline alone: a hollow marker |
| `edgecolor` | colour or `None` | `None` | Outline colour; `None` = the fill colour |
| `edge_linewidth` | float | `0` | Outline width, drawn inside the marker; 0 = none. The stroke width of the line markers |
| `errorbar` | dict | | [ErrorBarOptions](#errorbaroptions) |
| `zorder` | float | `0` | As in [LineOptions](#lineoptions) |
| `hint_labels` | list of str | `[]` | |

### ScatterZOptions

`scatter_z(x, y, z, *, err=None, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `cmap` | `Colormap` | `"viridis"` | |
| `size` | float | `20` | |
| `sizes` | array or `None` | `None` | As in [ScatterOptions](#scatteroptions) |
| `marker` | `MarkerStyle` | `"circle"` | |
| `alpha` | float | `0.8` | |
| `filled`, `edgecolor`, `edge_linewidth` | | `True`, `None`, `0` | As in [ScatterOptions](#scatteroptions); `None` = each point's own colour |
| `vmin`, `vmax` | float | `0`, `1` | Data range the colormap spans |
| `norm` | `ColorNorm` or str | `"linear"` | How values land on the colormap: see [ColorNorm](#colornorm) |
| `colorbar` | bool | `False` | Draw a colorbar for this series |
| `name` | str | `""` | Titles the colorbar; also the legend key (marker filled white, black edge) |
| `show_legend` | bool | `True` | |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `errorbar` | dict | | [ErrorBarOptions](#errorbaroptions); unset colour = black |
| `zorder` | float | `0` | As in [LineOptions](#lineoptions) |
| `hint_labels` | list of str | `[]` | Default hover text is `x=.., y=.., z=..` |

### BarOptions

`bar(x, height, *, err=None, **options)`, and the bar style of `hist(data, bins=10, *, density=False,
cumulative=False, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour or `None` | `None` | As in [LineOptions](#lineoptions) |
| `orientation` | `Orientation` | `"vertical"` | `"horizontal"` lays the bars along x (matplotlib's `barh`) |
| `bottom` | array, float or `None` | `None` | Where each bar starts: one value, or one per bar (stacking, waterfalls). Kept as data: `BarData.bottom` |
| `width` | float | `0.8`; `1.0` for `hist()` | Fraction of the bar spacing (for `hist()`, of the bin width) |
| `alpha` | float | `1` | |
| `name` | str | `""` | Legend key |
| `show_legend` | bool | `True` | |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `edgecolor` | colour | `"black"` | |
| `linewidth` | float | `0.5` | Edge width |
| `errorbar` | dict | | [ErrorBarOptions](#errorbaroptions); hung off the bar's tip; unset colour = `edgecolor` |
| `zorder` | float | `0` | As in [LineOptions](#lineoptions) |
| `hint_labels` | list of str | `[]` | |

`hist()`'s own keywords: `density` (normalize so the bars' area is 1) and `cumulative` (each bin counts itself
and every bin before it), both `False`.

### HeatmapOptions

`heatmap(data, xrange, yrange, **options)` and `imshow(data, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `cmap` | `Colormap` | `"viridis"` | |
| `vmin`, `vmax` | float | `0`, `1` | Data range the colormap spans |
| `norm` | `ColorNorm` or str | `"linear"` | How values land on the colormap: see [ColorNorm](#colornorm) |
| `colorbar` | bool | `False` | |
| `name` | str | `""` | Titles the colorbar |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `zorder` | float | `0` | As in [LineOptions](#lineoptions) |
| `origin` | str | `"lower"` | `"lower"` draws row 0 at the bottom, `"upper"` at the top |
| `contours` | array, float or `None` | `None` | Contour levels in data units |
| `contour_color` | colour | `"black"` | |
| `contour_linewidth` | float | `1` | |
| `contour_labels` | bool | `False` | Write each level on its line |
| `contour_fontsize` | float | `10` | |
| `hint_labels` | list of str | `[]` | Row-major: `row * cols + col` |

`xrange`, `yrange` are `(lo, hi)`: the outer cell edges. `lo > hi` mirrors; `lo == hi` is a `ValueError`.

### ImageOptions

`imshow_rgb(data, **options)`. The image carries its own colours: no colormap, colorbar or contours.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `xrange`, `yrange` | `(lo, hi)` or `None` | `None` | Where the image sits, as `heatmap()`'s ranges; `None` = `(0, cols)` and `(0, rows)` |
| `origin` | str | `"upper"` | `"upper"` draws row 0 at the top (as a photo), `"lower"` at the bottom |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `zorder` | float | `0` | As in [LineOptions](#lineoptions) |
| `hint_labels` | list of str | `[]` | Row-major: `row * cols + col` |

### HeatmapRecOptions

`heatmap_rec(data, x_edges, y_edges, **options)`: [HeatmapOptions](#heatmapoptions) without `origin` (row 0 is
at the first y edge).

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `cmap` | `Colormap` | `"viridis"` | |
| `vmin`, `vmax` | float | `0`, `1` | |
| `norm` | `ColorNorm` or str | `"linear"` | How values land on the colormap: see [ColorNorm](#colornorm) |
| `colorbar` | bool | `False` | |
| `name` | str | `""` | Titles the colorbar |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `zorder` | float | `0` | As in [LineOptions](#lineoptions) |
| `contours`, `contour_color`, `contour_linewidth`, `contour_labels`, `contour_fontsize` | | | As in [HeatmapOptions](#heatmapoptions); traced between cell centres |
| `hint_labels` | list of str | `[]` | Row-major: `row * cols + col` |

### HeatmapTriOptions

`heatmap_tri(x, y, values, tri=None, **options)`; `tri` is an `(n, 3)` index array, `None` = Delaunay.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `cmap` | `Colormap` | `"viridis"` | |
| `vmin`, `vmax` | float | `0`, `1` | |
| `norm` | `ColorNorm` or str | `"linear"` | Applied per pixel to the interpolated value; under `"log"` a vertex <= 0 leaves out its triangles |
| `colorbar` | bool | `False` | |
| `name` | str | `""` | Titles the colorbar |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `zorder` | float | `0` | As in [LineOptions](#lineoptions) |
| `hint_labels` | list of str | `[]` | One per vertex; hover shows the nearest vertex's |

### ContourOptions

`contour(x, y, z, **options)` and `contourf(x, y, z, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `levels` | array | `[]` | In z's units, sorted and de-duplicated; at most 16384. Empty = `nlevels` nice values over z's range |
| `nlevels` | int | `7` | |
| `filled` | bool | `False` | Bands between neighbouring levels instead of lines (set by `contourf()`) |
| `cmap` | `Colormap` | `"viridis"` | A line takes its level's colour, a band its midpoint's |
| `norm` | `ColorNorm` or str | `"linear"` | How values land on the colormap: see [ColorNorm](#colornorm) |
| `vmin`, `vmax` | float or `None` | `None` | `None` = the first and last level |
| `line_color` | colour or `None` | `None` | One colour for every line instead of the colormap |
| `linewidth` | float | `1.5` | Lines only |
| `labels` | bool | `False` | Write each level on its line, breaking it to fit |
| `fontsize` | float | `10` | |
| `colorbar` | bool | `False` | A `contourf`'s is stepped, labelled at the levels |
| `name` | str | `""` | Titles the colorbar |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `zorder` | float | `0` | As in [LineOptions](#lineoptions) |

### Hist2dOptions

`hist2d(x, y, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `xbins`, `ybins` | int | `10`, `10` | Even bins |
| `xrange`, `yrange` | `(lo, hi)` or `None` | `None` | `None` = the points' extent (widened by 0.5 each way when it is one value) |
| `x_edges`, `y_edges` | array | `[]` | Bin edges for that axis instead of bins and range; with either set the result is a `heatmap_rec()` |
| `density` | bool | `False` | Count / (points inside * bin area) instead of the count |
| `cmin`, `cmax` | float or `None` | `None` | A bin below / above is empty; `cmin=1` hides empty bins |
| `cmap` | `Colormap` | `"viridis"` | |
| `norm` | `ColorNorm` or str | `"linear"` | How values land on the colormap: see [ColorNorm](#colornorm) |
| `vmin`, `vmax` | float or `None` | `None` | `None` = the bins' extent (under `"log"`, from the smallest above 0) |
| `colorbar` | bool | `False` | |
| `name` | str | `""` | Titles the colorbar |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `zorder` | float | `0` | As in [LineOptions](#lineoptions) |

### ColorNorm

`sextant.ColorNorm`: how values map onto a colormap, as matplotlib's norms. The `norm=` keyword of the
heatmap kinds, `contour`/`contourf`, `hist2d`, `scatter_z`, `quiver` and `quiver3d` takes one, or the strings
`"linear"` and `"log"`.

| Member | Meaning |
|---|---|
| `ColorNorm.linear()` | The default (`Normalize`) |
| `ColorNorm.log()` | `LogNorm`: `vmin` and `vmax` > 0; a value <= 0 has no colour |
| `ColorNorm.two_slope(center)` | `TwoSlopeNorm`: `center` at the map's middle, strictly between `vmin` and `vmax` |
| `ColorNorm.boundaries(b)` | `BoundaryNorm`: one colour per band `[b[i], b[i+1])`, the last closed; 2 to 16384 finite, increasing values; `vmin`/`vmax` unused; a value outside has no colour |
| `kind()`, `center()`, `bounds()` | `NormKind`, the centre, the boundaries |

A norm that does not fit its `vmin`/`vmax` is a `ValueError` at the call.

### FillOptions

`fill(x, y1, y2=0.0, **options)`; `stack()` builds one per layer.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `orientation` | `Orientation` | `"horizontal"` | `"horizontal"` fills between y1(x) and y2(x) (`fill_between`); `"vertical"` between x1(y) and x2(y) (`fill_betweenx`) |
| `color` | colour or `None` | `None` | As in [LineOptions](#lineoptions) |
| `alpha` | float | `1` | |
| `edgecolor` | colour or `None` | `None` | Outline of each region; `None` = `color` |
| `linewidth` | float | `0` | Outline width; 0 = none |
| `linestyle` | `LineStyle` | `"solid"` | Of the outline |
| `drawstyle` | `DrawStyle` | `"default"` | Steps both curves |
| `exclude` | array of bool or `None` | `None` | One flag per point leaving it out (the inverse of matplotlib's `where`); a NaN also splits the fill. Kept as data: `FillData.exclude` |
| `interpolate` | bool | `False` | Run a region on to where the curves cross |
| `name` | str | `""` | Legend key |
| `show_legend` | bool | `True` | |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `zorder` | float | `0` | As in [LineOptions](#lineoptions) |

### StackOptions

`stack(x, ys, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `orientation` | `Orientation` | `"horizontal"` | `"horizontal"` stacks up along y over x; `"vertical"` along x over y |
| `baseline` | `StackBaseline` | `"zero"` | Where the bottom layer starts: `"zero"`, `"sym"`, `"wiggle"`, `"weighted_wiggle"` |
| `colors` | list of colour | `[]` | One per layer; empty = one cycle colour each |
| `names` | list of str | `[]` | One legend name per layer |
| `alpha`, `edgecolor`, `linewidth`, `linestyle`, `drawstyle` | | | As in [FillOptions](#filloptions), for every layer |
| `show_legend`, `visible` | bool | `True` | For every layer |
| `zorder` | float | `0` | For every layer |

### QuiverOptions

`quiver(x, y, u, v, c=None, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour or `None` | `None` | As in [LineOptions](#lineoptions); unused with a `c` per arrow (the cycle is not advanced) |
| `alpha` | float | `1` | Multiplies either colour's own alpha |
| `cmap` | `Colormap` | `"viridis"` | For a `c` per arrow (a cyclic map wraps) |
| `vmin`, `vmax` | float | `0`, `1` | Data range the colormap spans |
| `norm` | `ColorNorm` or str | `"linear"` | How values land on the colormap: see [ColorNorm](#colornorm) |
| `colorbar` | bool | `False` | Only with `c`; `name` titles it |
| `amp` | float or `None` | `None` | Data units of arrow per unit of vector (matplotlib's `1/scale`). `None` = chosen from the data at the call, then kept, so animation frames stay comparable |
| `pivot` | `QuiverPivot` | `"tail"` | Which part of the arrow sits on `(x, y)`: `"tail"`, `"middle"`, `"tip"` |
| `width` | float | `2` | Shaft width in pixels |
| `head_width`, `head_length` | float | `7`, `8` | Head size in pixels; an arrow shorter than its head is all head |
| `name` | str | `""` | Legend key; with `c`, also the colorbar title |
| `show_legend` | bool | `True` | |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `zorder` | float | `0` | As in [LineOptions](#lineoptions) |
| `hint_labels` | list of str | `[]` | Default hover text is `x=.., y=.., u=.., v=..` (plus `c=..`) |

### AxLineOptions

`axline(value, **options)`, `axline(x, y, *, slope, coords="data", **options)` and `axline(x1, y1, x2, y2, *,
coords="data", **options)`; `coords` is a [Coords](#enums), or `xcoords`/`ycoords` per axis.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `orientation` | `Orientation` | `"horizontal"` | A line at y = value (`axhline`), or with `"vertical"` at x = value (`axvline`). Unused by the sloped forms |
| `color` | colour or `None` | `None` | As in [LineOptions](#lineoptions) |
| `linewidth` | float | `1.5` | |
| `linestyle` | `LineStyle` | `"solid"` | `"none"` draws nothing and keys nothing |
| `alpha` | float | `1` | |
| `from_`, `to` | float | `0`, `1` | Where it starts and ends along the frame, as fractions of it (0 = left or bottom edge), whatever the limits. Unused by the sloped forms |
| `name` | str | `""` | Legend key |
| `show_legend` | bool | `True` | |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `zorder` | float | `0` | As in [LineOptions](#lineoptions) |

### AxSpanOptions

`axspan(lo, hi, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `orientation` | `Orientation` | `"horizontal"` | A band over y (`axhspan`), or with `"vertical"` over x (`axvspan`) |
| `color` | colour or `None` | `None` | `None` = `"blue"`; a span takes no colour from the cycle |
| `alpha` | float | `0.25` | |
| `edgecolor` | colour or `None` | `None` | Outline; `None` = `color` |
| `linewidth` | float | `0` | Outline width; 0 = none |
| `from_`, `to` | float | `0`, `1` | As in [AxLineOptions](#axlineoptions) |
| `name` | str | `""` | Legend key |
| `show_legend` | bool | `True` | |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `zorder` | float | `0` | As in [LineOptions](#lineoptions) |

### PolyOptions

`shape_poly(x, y, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour or `None` | `None` | As in [LineOptions](#lineoptions) |
| `alpha` | float | `1` | Multiplies fill and outline |
| `edgecolor` | colour or `None` | `None` | `None` = `color` |
| `linewidth` | float | `0` | Outline width; 0 = none |
| `linestyle` | `LineStyle` | `"solid"` | |
| `space` | `Coords` | `"data"` | What every vertex is measured in |
| `name` | str | `""` | Legend key |
| `show_legend` | bool | `True` | |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `zorder` | float | `0` | As in [LineOptions](#lineoptions) |

### CircleOptions

`shape_circle(cx, cy, r, *, coords="data", xcoords=None, ycoords=None, rcoords=None, **options)`:
[PolyOptions](#polyoptions) without `space`. `coords` sets all three; `xcoords`, `ycoords` (the centre) and
`rcoords` (the radius; as a fraction, of the frame's shorter side) one by one.

### WedgeOptions

`shape_wedge(cx, cy, r, theta1, theta2, *, coords="data", ..., **options)`, angles in degrees counter-clockwise
from +x: [CircleOptions](#circleoptions) plus

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `inner` | float | `0` | The radius of a hole, in the units of `r`: a ring segment. Kept as data: `WedgeData.inner` |

### PieOptions

`pie(values, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `labels` | list of str | `[]` | One per slice; also the legend's entries. Kept as data: `PieData` |
| `explode` | array | `[]` | Each slice's offset from the centre, a fraction of `radius`. Kept as data |
| `colors` | list of colour | `[]` | Slice k is `colors[k % len]`; empty = the colour cycle |
| `autopct` | str | `""` | Each slice's percent drawn on it, a printf conversion (`"%1.1f%%"`); anything else is a `ValueError` |
| `pctdistance` | float | `0.6` | In radii from the centre |
| `labeldistance` | float or `None` | `1.1` | `None` = no labels drawn |
| `startangle` | float | `0` | Degrees counter-clockwise from +x where the first slice starts |
| `counterclock` | bool | `True` | |
| `radius` | float | `1` | Data units |
| `cx`, `cy` | float | `0`, `0` | The centre |
| `normalize` | bool | `True` | `False`: values are fractions of the circle, summing to at most 1 |
| `inner` | float | `0` | A hole's radius, < `radius`: a ring |
| `edgecolor` | colour or `None` | `None` | Slice outline; `None` = the slice's colour |
| `linewidth` | float | `0` | Outline width; 0 = none |
| `alpha` | float | `1` | |
| `fontsize` | float or `None` | `None` | Labels and percents; `None` = the axes' label font size |
| `textcolor` | colour | `"black"` | |
| `name` | str | `""` | The pie's (hover) |
| `show_legend` | bool | `True` | |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `zorder` | float | `0` | As in [LineOptions](#lineoptions) |

### ErrorBar

`sextant.ErrorBar(*, x_cap_lo=None, x_cap_hi=None, x_box_lo=None, x_box_hi=None, y_cap_lo=None,
y_cap_hi=None, y_box_lo=None, y_box_hi=None)`: error-bar **data**, passed as `err=` to `line`, `scatter`,
`scatter_z` and `bar`. Each field is an array with one value per point, or `None`.

| Fields | Meaning |
|---|---|
| `x_cap_lo`, `x_cap_hi`, `y_cap_lo`, `y_cap_hi` | Capped whisker, as offsets below/above the point |
| `x_box_lo`, `x_box_hi`, `y_box_lo`, `y_box_hi` | Box, as offsets below/above the point |

One end given means symmetric; zero or NaN draws nothing on that side; a negative value is read as its size.

### ErrorBarOptions

Error-bar **style**: the `errorbar={...}` keyword of a 2D series.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour or `None` | `None` | `None` = the series' colour (`edgecolor` for bars, black for `scatter_z`) |
| `linewidth` | float | `1` | |
| `capsize` | float | `6` | Total cap length across a whisker end; 0 = no caps |
| `capstyle` | `CapStyle` | `"flat"` | `"arrow"` draws an open chevron pointing away from the point |
| `boxwidth` | float | `10` | Box width across the whisker, for a direction without box data |
| `box_alpha` | float | `0.25` | Box fill opacity, as a fraction of `color`'s; 0 = outline only |

---

## Decoration and style

Titles and limits are plain arguments: `set_title(text, fontsize=18)`, `set_xtitle(text, fontsize=16.5)`,
`set_ytitle(...)` (and `set_ztitle` in 3D), `set_xlim(lo, hi)`, `set_ylim(lo, hi)`,
`set_xticks(positions, labels=[])`, `set_yticks(...)`. An empty `set_xticks([])` means automatic ticks; to hide
them, `show_xticks=False` below.

### AxesStyle

`set_axes_style(**options)` on `Axes` and `Axes3D`. Unnamed keywords reset to their defaults.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `background` | colour | `"white"` | 2D plot-area fill, under the data; alpha 0 shows the figure background. A 3D box uses `pane_color` ([Box3DStyle](#box3dstyle)) |
| `spine_color` | colour | `(0.3, 0.3, 0.3)` | Frame and axis lines |
| `spine_linewidth` | float | `1` | |
| `spine_bottom`, `spine_left`, `spine_top`, `spine_right` | bool | `True` | The four 2D frame edges; ignored in 3D |
| `xaxis_color`, `yaxis_color` | colour or `None` | `None` | A twinned cell: the colour of the frame edge each axis lies on (host y left, `twinx` y right; host x bottom, `twiny` x top), so an axis can take its series' colour. `None`: `spine_color` |
| `xaxis_y`, `xaxis_z` | `AxisPosition` | `"auto"` | Where the x axis sits along y (and z, in 3D) |
| `yaxis_x`, `yaxis_z` | `AxisPosition` | `"auto"` | Where the y axis sits along x (and z) |
| `zaxis_x`, `zaxis_y` | `AxisPosition` | `"auto"` | Where the z axis sits (3D) |
| `origin_x`, `origin_y`, `origin_z` | float or `None` | `None` | Put the other axes through this data value; overrides the positions |
| `frame_margin` | float | `0` | Space around the frame and its labels before an outside legend or colorbar |
| `show_xticks`, `show_yticks`, `show_zticks` | bool | `True` | `False` hides that axis' tick marks and labels and frees their room; grid lines stay. `show_zticks` is 3D only |
| `show_xtick_labels`, `show_ytick_labels` | bool or `None` | `None` | 2D: show or hide an axis' tick labels, keeping its marks. `None` hides them on an inner axes of a share group |
| `tick_color` | colour | `(0.3, 0.3, 0.3)` | |
| `tick_length` | float | `5` | |
| `tick_linewidth` | float | `1` | |
| `minor_xticks`, `minor_yticks` | bool or `None` | `None` | 2D: unlabelled marks between the major ticks. `None` = on for log and symlog axes, off for linear |
| `minor_tick_length` | float | `2.5` | |
| `label_color` | colour | `(0.2, 0.2, 0.2)` | Tick labels |
| `label_fontsize` | float | `11` | |
| `xtick_color`, `ytick_color`, `ztick_color` | colour or `None` | `None` | Per axis, over `tick_color`; `None` takes the shared field. `z` in 3D only |
| `xtick_length`, `ytick_length`, `ztick_length` | float or `None` | `None` | Likewise for `tick_length` |
| `xtick_linewidth`, `ytick_linewidth`, `ztick_linewidth` | float or `None` | `None` | Likewise for `tick_linewidth` |
| `minor_xtick_length`, `minor_ytick_length` | float or `None` | `None` | Likewise for `minor_tick_length` (2D) |
| `xlabel_color`, `ylabel_color`, `zlabel_color` | colour or `None` | `None` | Likewise for `label_color` |
| `xlabel_fontsize`, `ylabel_fontsize`, `zlabel_fontsize` | float or `None` | `None` | Likewise for `label_fontsize` |
| `title_color`, `title_fontsize` | colour, float | `(0.15, 0.15, 0.15)`, `18` | Also set by `set_title()` |
| `xtitle_color`, `xtitle_fontsize` | colour, float | `(0.15, 0.15, 0.15)`, `16.5` | |
| `ytitle_color`, `ytitle_fontsize` | colour, float | `(0.15, 0.15, 0.15)`, `16.5` | |
| `ztitle_color`, `ztitle_fontsize` | colour, float | `(0.15, 0.15, 0.15)`, `16.5` | 3D only |
| `font_path` | str | `""` | Titles and tick labels; absolute path to a .ttf/.ttc/.otf |

### ScaleOptions

`set_xscale(scale, **options)` / `set_yscale(...)`, with `scale` a `Scale`: `"linear"` (the default), `"log"` or
`"symlog"`. Each value used must be finite.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `base` | float | `10` | Log and symlog; > 1 |
| `linthresh` | float | `1` | Symlog: the half-width of the linear part around 0; > 0 |
| `linscale` | float | `1` | Symlog: the length of the linear part, in decades; > 0 |

### GridOptions

`grid(enable=True, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour | `(0.8, 0.8, 0.8)` | |
| `linestyle` | `LineStyle` | `"solid"` | `"none"` draws no grid lines |
| `linewidth` | float | `0.5` | |
| `minor` | bool | `False` | 2D: also a line at each minor tick |
| `minor_color` | colour | `(0.9, 0.9, 0.9)` | Colour of those lines |

### LegendOptions

`legend(**options)`. Unnamed keywords reset to their defaults.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `anchor` | `LegendAnchor` | `"outside_rt"` | Inside a frame corner, or outside the frame |
| `margin` | float | `10` | Distance from the anchor; space reserved for an outside legend |
| `offset_x`, `offset_y` | float | `0` | Move the drawn box only |
| `fontsize` | float | `10` | |
| `frameon` | bool | `True` | Draw the box |
| `text_color` | colour | `(0.15, 0.15, 0.15)` | |
| `frame_color` | colour | `(1, 1, 1, 0.85)` | Box fill |
| `border_color` | colour | `(0.5, 0.5, 0.5)` | |
| `border_linewidth` | float | `1` | |
| `font_path` | str | `""` | Independent of the axes' `font_path` |

### ColorbarOptions

`set_colorbar_style(**options)`; styles every colorbar of the axes. Unnamed keywords reset to their defaults.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `anchor` | `ColorbarAnchor` | `"right"` | Side of the frame; `"top"`/`"bottom"` bars are horizontal |
| `width` | float | `15` | Thickness across the bar |
| `margin` | float | `15` | Space between a bar and the frame, legend or previous bar |
| `offset_x`, `offset_y` | float | `0` | Move the bars as drawn |
| `fontsize` | float | `10` | |
| `text_color` | colour | `(0.2, 0.2, 0.2)` | |
| `border_color` | colour | `(0.3, 0.3, 0.3)` | |
| `border_linewidth` | float | `1` | |
| `font_path` | str | `""` | |

---

## Text

`Axes.text(s, x, y, *, coords="data", xcoords=None, ycoords=None, **options)` and
`annotate(px, py, s, tx, ty, ..., arrow=None, **options)`; in 3D `text()`, `text2d()` and `annotate()`
([3d.md](3d.md#text-and-annotations)). `coords` (and per axis `xcoords`/`ycoords`) is a [`Coords`](#enums):
`"data"`, or `"fraction"` of the plot frame from its left (x) or bottom (y) edge.

### TextOptions

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `fontsize` | float | `12` | Pixels as drawn; constant under zoom and at any 3D distance |
| `color` | colour | `"black"` | |
| `alpha` | float | `1` | Of everything the text draws (text, box, arrow) |
| `font_path` | str | `""` | Empty = the default font |
| `ha` | `HAlign` | `"left"` | Where the block sits against its anchor, across |
| `va` | `VAlign` | `"baseline"` | ... and up/down; `baseline` is the last line's |
| `rotation` | float | `0` | Degrees counter-clockwise, about the anchor |
| `dx`, `dy` | float | `0` | Nudge of the whole text, box included; y up |
| `linespacing` | float | `1.2` | Lines (`"\n"`) are `linespacing` × `fontsize` apart |
| `background` | colour | transparent | Box fill; alpha 0 = no fill |
| `edgecolor` | colour or `None` | `None` | Box outline; `None` = the text colour |
| `edge_linewidth` | float | `0` | 0 = no outline |
| `pad` | float | `4` | Box margin around the text |
| `clip_to_frame` | bool | `False` | Cut at the plot frame; by default a text whose data position leaves the view is hidden whole |
| `parse_math` | bool | `True` | Draw `$...$` as math (only when the figure's `mathtext` is on too) |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |

### ArrowOptions

`annotate(..., arrow={...})`: the arrow from the text to the point, as a dict; `None` takes every default.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `head` | `ArrowHead` | `"filled"` | At the point |
| `tail` | `ArrowHead` | `"none"` | At the text |
| `head_length`, `head_width` | float | `10`, `7` | |
| `linewidth` | float | `1.25` | |
| `color` | colour or `None` | `None` | `None` = the text colour |
| `linestyle` | `LineStyle` | `"solid"` | |
| `gap_text`, `gap_point` | float | `2`, `0` | Pixels left clear at the text's box and before the point |
| `arc` | float | `0` | 0 = straight; otherwise it bows to one side by `arc` × its length (negative: the other side), as matplotlib's `arc3` |

---

## 3D options

### Camera3D

`set_camera(**options)` and `set_default_camera(**options)` (unnamed keywords reset to their defaults);
`camera()` returns a dict with these keys.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `azimuth` | float | `-60` | Degrees about the box's z axis; 0 looks along +x |
| `elevation` | float | `30` | Degrees above the xy plane, clamped to ±89 |
| `target` | `(x, y, z)` | `(0, 0, 0)` | Box-space point the camera orbits |
| `zoom` | float | `1` | Magnifies the fitted picture |
| `projection` | `Projection` | `"orthographic"` | |
| `fov` | float | `45` | Vertical field of view, perspective only, 5..120 |

`set_box_aspect((x, y, z))`: relative side lengths of the box, default `(1, 1, 1)`.

### Box3DStyle

`set_box_style(**options)`. Unnamed keywords reset to their defaults.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `panes` | bool | `True` | Draw the three back panes |
| `pane_color` | colour | `(0.94, 0.94, 0.96)` | |
| `pane_edge_color` | colour | `(0.75, 0.75, 0.78)` | |
| `margin` | float | `0.12` | Fraction of the cell left around the box for labels |

### Plane2DOptions

`plane(orient, offset=0.0, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `alpha` | float | `1` | Whole-plane opacity, multiplying its objects' own; also `set_alpha()` |
| `visible` | bool | `True` | Hides the drawing only; limits, colorbars and legend keys stay. Hiding one object instead is `set_<kind>_visible()` |

### Bar3DOptions

`bar3d(orient, u, v, heights, *, bottoms=None, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour or `None` | `None` | As in [LineOptions](#lineoptions) |
| `alpha` | float | `1` | |
| `width`, `depth` | float | `0.8` | Footprint as a fraction of the grid spacing along u and v |
| `bottom` | float | `0` | Base of every bar, unless `bottoms` is passed |
| `shading` | float | `0.45` | 0 = flat colour, 1 = the dimmest face black |
| `edges` | bool | `False` | Outline the bars |
| `edgecolor` | colour | `"black"` | Its own alpha, times `alpha` |
| `edge_linewidth` | float | `1` | At the bar's depth |
| `name` | str | `""` | Legend key (a swatch of `color`) |
| `show_legend` | bool | `True` | |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `hint_labels` | list of str | `[]` | Per bar, u major like `heights` |

### SurfaceOptions

`surface(orient, u, v, heights, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour or `None` | `None` | Flat colour, unless `colormap`; `None` = the next cycle colour |
| `colormap` | bool | `False` | Colour each cell by height |
| `cmap` | `Colormap` | `"viridis"` | |
| `vmin`, `vmax` | float | `0`, `0` | Equal = the surface's own range |
| `colorbar` | bool | `False` | Only with `colormap` |
| `name` | str | `""` | Colorbar title when colour-mapped, else legend key |
| `show_legend` | bool | `True` | |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `alpha` | float | `1` | |
| `shading` | float | `0.45` | |
| `edges` | bool | `False` | Wireframe along the sampling grid |
| `edgecolor` | colour | `"black"` | Its own alpha, times `alpha` |
| `edge_linewidth` | float | `1` | At the box centre |
| `hint_labels` | list of str | `[]` | Per sample, u major like `heights` |

### SurfaceTriOptions

`surface_tri(x, y, z, triangles=None, *, orient=None, colors=None, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour or `None` | `None` | Flat colour, unless `colors` is passed; `None` = the next cycle colour |
| `cmap` | `Colormap` | `"viridis"` | For `colors` |
| `vmin`, `vmax` | float | `0`, `0` | Equal = the mesh's own range |
| `colorbar` | bool | `False` | Only with `colors` |
| `name` | str | `""` | Colorbar title when colour-mapped, else legend key |
| `show_legend` | bool | `True` | |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `alpha` | float | `1` | |
| `shading` | float | `0.45` | Per face, both sides lit |
| `edges` | bool | `False` | All three edges of every triangle |
| `edgecolor` | colour | `"black"` | Its own alpha, times `alpha` |
| `edge_linewidth` | float | `1` | At the box centre |
| `hint_labels` | list of str | `[]` | Per vertex |

### Scatter3DOptions

`scatter3d(x, y, z, *, colors=None, err=None, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour or `None` | `None` | Flat colour, unless `colors` is passed; `None` = the next cycle colour |
| `size` | float | `20` | Marker diameter, constant at any depth |
| `sizes` | array or `None` | `None` | One diameter per point, as in [ScatterOptions](#scatteroptions) |
| `marker` | `MarkerStyle` | `"circle"` | |
| `alpha` | float | `1` | Multiplies the own alpha of every colour of the series |
| `depthshade` | float | `0` | Darken with distance; 0 = off |
| `filled`, `edgecolor`, `edge_linewidth` | | `True`, `None`, `0` | Marker outline, as in [ScatterOptions](#scatteroptions); darkened with `depthshade` |
| `cmap` | `Colormap` | `"viridis"` | For `colors` |
| `vmin`, `vmax` | float | `0`, `0` | Equal = the series' own range |
| `colorbar` | bool | `False` | Only with `colors` |
| `name` | str | `""` | Legend key; with `colors`, also the colorbar title |
| `show_legend` | bool | `True` | |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `errorbar` | dict | | [ErrorBar3DOptions](#errorbar3doptions) |
| `hint_labels` | list of str | `[]` | |

### Line3DOptions

`line3d(x, y, z, *, colors=None, err=None, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour or `None` | `None` | Flat colour, unless `colors` is passed; `None` = the next cycle colour |
| `linewidth` | float | `1.5` | At the box centre; thinner with distance under perspective |
| `alpha` | float | `1` | |
| `loop` | bool | `False` | Close the path |
| `depthshade` | float | `0` | |
| `cmap` | `Colormap` | `"viridis"` | For `colors` |
| `vmin`, `vmax` | float | `0`, `0` | Equal = the path's own range |
| `colorbar` | bool | `False` | Only with `colors` |
| `name` | str | `""` | Legend key |
| `show_legend` | bool | `True` | |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `errorbar` | dict | | [ErrorBar3DOptions](#errorbar3doptions) |
| `hint_labels` | list of str | `[]` | Per vertex |

### Quiver3DOptions

`quiver3d(x, y, z, u, v, w, c=None, **options)`. The arrows are flat and turn to face the eye.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour or `None` | `None` | Flat colour; `None` = the next cycle colour. Unused with a `c` per arrow |
| `alpha` | float | `1` | |
| `cmap` | `Colormap` | `"viridis"` | For `c` |
| `vmin`, `vmax` | float | `0`, `0` | Equal = the arrows' own `c` range |
| `norm` | `ColorNorm` or str | `"linear"` | How values land on the colormap: see [ColorNorm](#colornorm) |
| `colorbar` | bool | `False` | Only with `c`; `name` titles it |
| `amp` | float or `None` | `None` | Data units of arrow per unit of vector: an arrow runs from (x, y, z) to (x, y, z) + amp * (u, v, w). `None` = chosen from the data at the call, as in 2D |
| `pivot` | `QuiverPivot` | `"tail"` | Which part of the arrow sits on the point |
| `width` | float | `2` | Shaft width, at the box centre |
| `head_width`, `head_length` | float | `7`, `8` | The head, at the box centre; an arrow shorter than its head is all head |
| `depthshade` | float | `0` | As in [Scatter3DOptions](#scatter3doptions) |
| `name` | str | `""` | Legend key |
| `show_legend` | bool | `True` | |
| `visible` | bool | `True` | As in [LineOptions](#lineoptions) |
| `hint_labels` | list of str | `[]` | Per arrow |

### ErrorBar3D

`sextant.ErrorBar3D(*, x_cap_lo=None, ..., z_box_hi=None)`: error-bar data for `scatter3d` and `line3d`,
passed as `err=`. The same rules as [ErrorBar](#errorbar).

| Fields | Meaning |
|---|---|
| `x_cap_lo`, `x_cap_hi`, `y_cap_lo`, `y_cap_hi`, `z_cap_lo`, `z_cap_hi` | Capped whisker offsets |
| `x_box_lo`, `x_box_hi`, `y_box_lo`, `y_box_hi`, `z_box_lo`, `z_box_hi` | Box offsets |

### ErrorBar3DOptions

The `errorbar={...}` keyword of `scatter3d` and `line3d`. Pixel sizes are measured at the box centre.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour or `None` | `None` | `None` = the series' colour, or black with `colors` |
| `linewidth` | float | `1` | 0 draws no error bar at all |
| `capsize` | float | `6` | |
| `capstyle` | `CapStyle` | `"flat"` | |
| `boxwidth` | float | `10` | Block extent on axes without box data |
| `box_alpha` | float | `0.25` | Block faces, as a fraction of `color`'s alpha; 0 drops them. The whole error bar also takes the series' `alpha` |

---

## Enums

Pass the value as a string (case, `_`, `-` and spaces are ignored, except in marker codes), or use the class member
(`sextant.LineStyle.DASHED`). Members are `str` and compare equal to their value.

| Class | Values | Aliases |
|---|---|---|
| `LineStyle` | `solid`, `dashed`, `dotted`, `dashdot`, `none` | `-` `--` `:` `-.` `""` |
| `MarkerStyle` | `none`, `circle`, `square`, `triangle`, `triangle_down`, `triangle_left`, `triangle_right`, `diamond`, `star`, `pentagon`, `hexagon`, `cross`, `plus`, `thin_cross`, `thin_plus`, `vline`, `hline` | `o` `s` `^` `v` `<` `>` `*` `p` `h` `X` `P` `x` `+` `\|` `_` `D` `d` `""` (case-sensitive: `x` is `thin_cross`, `X` is `cross`) |
| `DrawStyle` | `default`, `steps_pre`, `steps_mid`, `steps_post` | `steps`, `steps-pre`, `steps-mid`, `steps-post` |
| `Orientation` | `vertical`, `horizontal` | |
| `StackBaseline` | `zero`, `sym`, `wiggle`, `weighted_wiggle` | |
| `QuiverPivot` | `tail`, `middle`, `tip` | `mid` |
| `Scale` | `linear`, `log`, `symlog` | |
| `Adjustable` | `box`, `datalim` | |
| `MathFontset` | `text`, `stix`, `cm`, `custom` | `stixsans`, `dejavusans`, `dejavuserif` |
| `Colormap` | `viridis`, `plasma`, `inferno`, `magma`, `cividis`, `turbo`, `coolwarm`, `gray`, `Blues`, `Greens`, `Reds`, `Oranges`, `Purples`, `Greys`, `YlOrRd`, `YlGnBu`, `hot`, `jet`, `RdBu`, `Spectral`, `BrBG`, `PiYG`, `twilight`, `twilight_shifted`, `hsv`, `phase` (cyclic: the last four) | `grey`; any name with `_r` reversed; registered names ([Module functions](#module-functions)) |
| `CapStyle` | `flat`, `arrow` | |
| `AxisPosition` | `auto` (bottom/left in 2D; the box silhouette in 3D), `low`, `mid`, `high` | |
| `LegendAnchor` | `inside_tl`, `inside_tr`, `inside_bl`, `inside_br`, `outside_tl`, `outside_tr`, `outside_bl`, `outside_br`, `outside_lt`, `outside_lb`, `outside_rt`, `outside_rb` | |
| `ColorbarAnchor` | `left`, `right`, `top`, `bottom` | |
| `HAlign` | `left`, `center`, `right` | `centre` |
| `VAlign` | `top`, `center`, `baseline`, `bottom` | |
| `Coords` | `data`, `fraction`, `pixels` (from the frame's bottom left) | `axes fraction` |
| `AxLineMode` | `value`, `point_slope`, `two_points` | |
| `NormKind` | `linear`, `log`, `two_slope`, `boundaries` | |
| `ThetaZero` | `e`, `n`, `w`, `s`, `ne`, `nw`, `sw`, `se` | |
| `ArrowHead` | `none`, `open`, `filled`, `bar` | |
| `PanelTheme` | `dark`, `light`, `classic` | |
| `Projection` | `orthographic`, `perspective` | `ortho`, `persp` |
| `PlaneOrientation` | `xy`, `yz`, `zx` | |
| `EventKind` | `close`, `mouse_down`, `mouse_up`, `mouse_move`, `scroll`, `key_down`, `key_up`, `resize`, `pick` | matplotlib's event names ([Events](api.md#events)) |
| `PickKind` | `none`, `line`, `scatter`, `scatter_z`, `bar`, `heatmap`, `bar3d`, `surface`, `surface_tri`, `scatter3d`, `line3d`, `quiver`, `fill`, `poly`, `circle`, `wedge`, `heatmap_tri`, `contour`, `quiver3d`, `pie` | |
| `EventConsumed` | `none`, `select`, `navigate`, `grid_drag` | |

`LegendAnchor`: `inside_*` sits in a corner of the frame (t/b top/bottom, l/r left/right) and reserves no
space. `outside_t*`/`outside_b*` sit above or below the frame, flush left or right, with entries in wrapping
rows; `outside_l*`/`outside_r*` sit left or right of it, flush top or bottom, with entries in a column.

---

## Data classes

Returned by `<kind>_data(i)` and accepted by `set_<kind>_data(i, obj)`. Every attribute is writable; arrays
are numpy `float64` unless noted. Each can also be built: `sextant.LineData(x, y)`.

| Class | Attributes |
|---|---|
| `LineData` | `x`, `y` (from `line(y)`, x is 0, 1, 2, …) |
| `ScatterData` | `x`, `y`, `sizes` (or `None`: every marker takes the call's `size`) |
| `ScatterZData` | `x`, `y`, `z`, `sizes` (or `None`) |
| `BarData` | `x`, `height` (from `hist()`: bin centres and heights), `bottom` (or `None`: the bars stand on 0; a single `bottom=` value reads back repeated) |
| `FillData` | `x` (`y` for a vertical fill), `y1`, `y2` (a number reads back repeated), `exclude` (bool array, or `None`). Each layer of a `stack()` is one |
| `QuiverData` | `x`, `y`, `u`, `v`, `c` (or `None` for a flat quiver) |
| `AxLineData` | `value`; for the sloped forms `mode` (`AxLineMode`), `x`, `y`, `x2`, `y2`, `slope` and their `xcoords`, `ycoords`, `x2coords`, `y2coords` |
| `AxSpanData` | `lo`, `hi` (`lo <= hi` as stored) |
| `HeatmapData` | `data` `(rows, cols)`, rounded to single precision (an `imshow_rgb()` image: `(rows, cols, channels)` in [0, 1]); `xrange`, `yrange` `(lo, hi)`; `x_edges`, `y_edges` (a `heatmap_rec()`'s, else `None`) |
| `HeatmapTriData` | `x`, `y`, `values`, `tri` (`(n, 3)`; the Delaunay one when none was passed) |
| `ContourData` | `x`, `y`, `z` `(len(y), len(x))`, `levels` (as drawn), `filled` |
| `PolyData` | `x`, `y` (in the options' `space`) |
| `CircleData` | `cx`, `cy`, `r` and their `cxcoords`, `cycoords`, `rcoords` |
| `WedgeData` | `CircleData`'s plus `theta1`, `theta2` (degrees), `inner` |
| `PieData` | `values`, `labels`, `explode` |
| `Bar3DData` | `orient` (`PlaneOrientation`), `u`, `v`, `heights` `(len(u), len(v))`, `bottoms` (same shape, or `None`) |
| `SurfaceData` | `orient`, `u`, `v`, `heights` `(len(u), len(v))` |
| `SurfaceTriData` | `x`, `y`, `z`, `tri` (`(n, 3)` `uint32`; the derived triangulation for an `orient=` mesh), `colors` (or `None`) |
| `Scatter3DData` | `x`, `y`, `z`, `colors` (or `None`), `sizes` (or `None`) |
| `Line3DData` | `x`, `y`, `z`, `colors` (or `None`) |
| `Quiver3DData` | `x`, `y`, `z`, `u`, `v`, `w`, `c` (or `None`) |
| `TextData` | `text`; `x`, `y` and their `xcoords`, `ycoords` (`Coords`); `arrow` (bool) and the point `px`, `py`. From `Axes.text_data(i)`, for `text()` and `annotate()` alike |
| `Text3DData` | `text`, `x`, `y`, `z`; `in_frame` (a `text2d()`: `x`, `y` are fractions of the frame); `arrow` and `dx`, `dy` (an `annotate()`: the text's offset in pixels, y up) |

---

## Event

What a callback connected with `fig.connect(kind, callback)` receives. Read-only. Every event has every
attribute; those its kind does not fill keep their defaults (NaN for data coordinates, -1 for indices).

| Attribute | Type | Filled for | Meaning |
|---|---|---|---|
| `kind` | `EventKind` | all | |
| `x`, `y` | float | pointer events | Logical pixels from the top left of the plot area |
| `inaxes` | `Axes`, `Axes3D`, `PolarAxes` or `None` | pointer events | The subplot under the pointer: the same object `add_subplot()` returned. Over a `PolarAxes`, `xdata`/`ydata` are θ (radians) and r |
| `axes` | int | pointer events | That subplot's index as `add_subplot()` takes it (a span: its first cell), or -1 |
| `has_data` | bool | pointer events | Whether `xdata`/`ydata`(/`zdata`) are set: over a 2D axes, or where a 3D pointer ray meets a `Plane2D` |
| `xdata`, `ydata`, `zdata` | float | pointer events | The pointer in data coordinates; NaN when not over data |
| `x2data`, `y2data` | float | pointer events | A 2D axes with twins: the `twiny()`'s x and the `twinx()`'s y under the pointer (NaN without that twin) |
| `button` | int | `mouse_down`, `mouse_up`, `pick` | 0 left, 1 right, 2 middle |
| `double_click` | bool | `mouse_down` | |
| `mods` | int | input events | Modifier bits: 1 ctrl, 2 shift, 4 alt, 8 super |
| `modifiers` | tuple of str | input events | The same by name, in the order `"ctrl"`, `"shift"`, `"alt"`, `"super"` |
| `scroll_x`, `scroll_y` | float | `scroll` | Wheel notches, positive away from the user; queued ones are summed |
| `key` | str | `key_down`, `key_up` | `"a"`, `"A"`, `"ctrl+a"`, `"escape"`, `"f5"`, `"left"`, … |
| `width`, `height` | int | `resize` | The new plot-area size |
| `consumed` | `EventConsumed` | input events | What sextant itself did with the input (informational) |
| `pick_kind` | `PickKind` | `pick` | The kind of object clicked |
| `pick_object` | int | `pick` | The object's id (0 = none), as `last_object()` returned it. `inaxes.find_object(id)` (the plane's, when `pick_plane != 0`) gives its kind and the `i` of `<kind>_data(i)`; `None` if removed since |
| `pick_index` | int | `pick` | The element: a point, bar, marker, vertex or surface sample; for a heatmap or `bar3d`, the flat cell index (`row * cols + col`); for a fill, its segment *i*, between its points *i* and *i* + 1 |
| `pick_row`, `pick_col` | int | `pick` | The cell of a heatmap or `bar3d` grid, else -1 |
| `pick_plane` | int | `pick` | The id of the plane the object is on (`Plane2D.id()`; `Axes3D.plane_index(id)` gives its `plane_at()` index), or 0 for an object of the axes itself |
| `pick_twin` | int | `pick` | In a cell with twins, whose object `pick_object` is: 0 the axes' own, 1 its `twinx()`, 2 its `twiny()` |

---

## Module functions

| Function | |
|---|---|
| `sextant.run()` | Wait until every open figure is closed, delivering their events |
| `sextant.poll_events()` | Process window events once and deliver queued events, without waiting. On macOS, main thread only |
| `sextant.set_message_handler(fn)` | Send sextant's diagnostics to `fn(message: str)`; `None` restores the default (`warnings.warn(message, RuntimeWarning)`). Returns the previous handler |
| `sextant.register_colormap(name, colors, cyclic=False)` | Add a colormap for the rest of the process, usable by name (and with `_r`) wherever `cmap=` is taken: `colors` (two or more) spread evenly from `vmin` to `vmax`, interpolated linearly, alpha included. `cyclic=True` wraps values outside `[vmin, vmax]` (give it equal first and last colours). Returns the name. `ValueError` for fewer than two colours, an empty name, one ending in `_r`, or one already taken; it cannot be removed |
| `sextant.colormaps()` | Every colormap's name, without `_r`: the built-ins, then the registered ones in registration order |
| `sextant.set_repr_formats(*formats)` | What a `Figure` offers Jupyter/IPython to display: `"png"` (the default), `"svg"`, both, or none |
