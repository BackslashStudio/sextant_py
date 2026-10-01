# sextant for Python: reference

Every keyword option, enum, data class and event field of the core API, with defaults. For how they are
used, see the [API guide](api.md) and [3D plots](3d.md).

Sizes are logical pixels (1/96 inch) unless a row says otherwise. *colour* means a colour as described in
[api.md](api.md#colours): a name, `"#rrggbb[aa]"`, or a 3- or 4-tuple of floats in [0, 1]; colour defaults are
written as `(r, g, b)` tuples. Named colours are matplotlib's tab10 shades: `"blue"` is `#1f77b4`,
`"orange"` `#ff7f0e`, and so on. Enum values are strings ([Enums](#enums)). *array* means anything
`numpy.asarray(x, float)` accepts.

- [Figure and output](#figure-and-output): [Figure()](#figure-options) · [margins](#figuremargins) ·
  [suptitle style](#suptitleoptions) · [PNG export](#pngexportoptions) · [SVG export](#svgexportoptions) ·
  [SvgSaveReport](#svgsavereport) · [FrameStats](#framestats)
- [2D plot options](#2d-plot-options): [line](#lineoptions) · [scatter](#scatteroptions) ·
  [scatter_z](#scatterzoptions) · [bar and hist](#baroptions) · [heatmap and imshow](#heatmapoptions) ·
  [ErrorBar](#errorbar) · [errorbar style](#errorbaroptions)
- [Decoration and style](#decoration-and-style): [set_axes_style](#axesstyle) · [grid](#gridoptions) ·
  [legend](#legendoptions) · [set_colorbar_style](#colorbaroptions)
- [3D options](#3d-options): [camera](#camera3d) · [box style](#box3dstyle) · [plane](#plane2doptions) ·
  [bar3d](#bar3doptions) · [surface](#surfaceoptions) · [surface_tri](#surfacetrioptions) ·
  [scatter3d](#scatter3doptions) · [line3d](#line3doptions) · [ErrorBar3D](#errorbar3d) ·
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
| `color` | colour | `"blue"` | |
| `linewidth` | float | `1.5` | |
| `linestyle` | `LineStyle` | `"solid"` | `"none"` draws no stroke and drops the series from the legend |
| `name` | str | `""` | Legend key |
| `show_legend` | bool | `True` | Keyed when this is true and `name` is not empty |
| `alpha` | float | `1` | |
| `loop` | bool | `False` | Add a closing segment from the last point to the first |
| `errorbar` | dict | | [ErrorBarOptions](#errorbaroptions): style of the error bars passed as `err` |
| `hint_labels` | list of str | `[]` | Extra hover text per point |

### ScatterOptions

`scatter(x, y, *, err=None, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour | `"blue"` | |
| `size` | float | `20` | Marker diameter; constant under zoom |
| `marker` | `MarkerStyle` | `"circle"` | |
| `name` | str | `""` | Legend key |
| `show_legend` | bool | `True` | |
| `alpha` | float | `0.8` | |
| `errorbar` | dict | | [ErrorBarOptions](#errorbaroptions) |
| `hint_labels` | list of str | `[]` | |

### ScatterZOptions

`scatter_z(x, y, z, *, err=None, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `cmap` | `Colormap` | `"viridis"` | |
| `size` | float | `20` | |
| `marker` | `MarkerStyle` | `"circle"` | |
| `alpha` | float | `0.8` | |
| `vmin`, `vmax` | float | `0`, `1` | Data range the colormap spans |
| `colorbar` | bool | `False` | Draw a colorbar for this series |
| `name` | str | `""` | Titles the colorbar; also the legend key (marker filled white, black edge) |
| `show_legend` | bool | `True` | |
| `errorbar` | dict | | [ErrorBarOptions](#errorbaroptions); unset colour = black |
| `hint_labels` | list of str | `[]` | Default hover text is `x=.., y=.., z=..` |

### BarOptions

`bar(x, height, *, err=None, **options)`, and the bar style of `hist(data, bins=10, *, density=False,
cumulative=False, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour | `"blue"` | |
| `width` | float | `0.8`; `1.0` for `hist()` | Fraction of the bar spacing (for `hist()`, of the bin width) |
| `alpha` | float | `1` | |
| `name` | str | `""` | Legend key |
| `show_legend` | bool | `True` | |
| `edgecolor` | colour | `"black"` | |
| `linewidth` | float | `0.5` | Edge width |
| `errorbar` | dict | | [ErrorBarOptions](#errorbaroptions); hung off the bar's tip; unset colour = `edgecolor` |
| `hint_labels` | list of str | `[]` | |

`hist()`'s own keywords: `density` (normalize so the bars' area is 1) and `cumulative` (each bin counts itself
and every bin before it), both `False`.

### HeatmapOptions

`heatmap(data, xrange, yrange, **options)` and `imshow(data, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `cmap` | `Colormap` | `"viridis"` | |
| `vmin`, `vmax` | float | `0`, `1` | Data range the colormap spans |
| `colorbar` | bool | `False` | |
| `name` | str | `""` | Titles the colorbar |
| `origin` | str | `"lower"` | `"lower"` draws row 0 at the bottom, `"upper"` at the top |
| `contours` | list of float | `[]` | Contour levels in data units |
| `contour_color` | colour | `"black"` | |
| `contour_linewidth` | float | `1` | |
| `contour_labels` | bool | `False` | Write each level on its line |
| `contour_fontsize` | float | `10` | |
| `hint_labels` | list of str | `[]` | Row-major: `row * cols + col` |

`xrange`, `yrange` are `(lo, hi)`: the outer cell edges. `lo > hi` mirrors; `lo == hi` is a `ValueError`.

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
`set_xticks(positions, labels=[])`, `set_yticks(...)`.

### AxesStyle

`set_axes_style(**options)` on `Axes` and `Axes3D`. Unnamed keywords reset to their defaults.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `spine_color` | colour | `(0.3, 0.3, 0.3)` | Frame and axis lines |
| `spine_linewidth` | float | `1` | |
| `spine_bottom`, `spine_left`, `spine_top`, `spine_right` | bool | `True` | The four 2D frame edges; ignored in 3D |
| `xaxis_y`, `xaxis_z` | `AxisPosition` | `"auto"` | Where the x axis sits along y (and z, in 3D) |
| `yaxis_x`, `yaxis_z` | `AxisPosition` | `"auto"` | Where the y axis sits along x (and z) |
| `zaxis_x`, `zaxis_y` | `AxisPosition` | `"auto"` | Where the z axis sits (3D) |
| `origin_x`, `origin_y`, `origin_z` | float or `None` | `None` | Put the other axes through this data value; overrides the positions |
| `frame_margin` | float | `0` | Space around the frame and its labels before an outside legend or colorbar |
| `tick_color` | colour | `(0.3, 0.3, 0.3)` | |
| `tick_length` | float | `5` | |
| `tick_linewidth` | float | `1` | |
| `label_color` | colour | `(0.2, 0.2, 0.2)` | Tick labels |
| `label_fontsize` | float | `11` | |
| `title_color`, `title_fontsize` | colour, float | `(0.15, 0.15, 0.15)`, `18` | Also set by `set_title()` |
| `xtitle_color`, `xtitle_fontsize` | colour, float | `(0.15, 0.15, 0.15)`, `16.5` | |
| `ytitle_color`, `ytitle_fontsize` | colour, float | `(0.15, 0.15, 0.15)`, `16.5` | |
| `ztitle_color`, `ztitle_fontsize` | colour, float | `(0.15, 0.15, 0.15)`, `16.5` | 3D only |
| `font_path` | str | `""` | Titles and tick labels; absolute path to a .ttf/.ttc/.otf |

### GridOptions

`grid(enable=True, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour | `(0.8, 0.8, 0.8)` | |
| `linestyle` | `LineStyle` | `"solid"` | `"none"` draws no grid lines |
| `linewidth` | float | `0.5` | |

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
| `visible` | bool | `True` | Hides the drawing only; limits, colorbars and legend keys stay |

### Bar3DOptions

`bar3d(orient, u, v, heights, *, bottoms=None, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour | `"blue"` | |
| `alpha` | float | `1` | |
| `width`, `depth` | float | `0.8` | Footprint as a fraction of the grid spacing along u and v |
| `bottom` | float | `0` | Base of every bar, unless `bottoms` is passed |
| `shading` | float | `0.45` | 0 = flat colour, 1 = the dimmest face black |
| `edges` | bool | `False` | Outline the bars |
| `edgecolor` | colour | `"black"` | |
| `edge_alpha` | float | `1` | Independent of `alpha` |
| `edge_linewidth` | float | `1` | At the bar's depth |
| `name` | str | `""` | Legend key (a swatch of `color`) |
| `show_legend` | bool | `True` | |
| `hint_labels` | list of str | `[]` | Per bar, u major like `heights` |

### SurfaceOptions

`surface(orient, u, v, heights, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour | `"blue"` | Flat colour, unless `colormap` |
| `colormap` | bool | `False` | Colour each cell by height |
| `cmap` | `Colormap` | `"viridis"` | |
| `vmin`, `vmax` | float | `0`, `0` | Equal = the surface's own range |
| `colorbar` | bool | `False` | Only with `colormap` |
| `name` | str | `""` | Colorbar title when colour-mapped, else legend key |
| `show_legend` | bool | `True` | |
| `alpha` | float | `1` | |
| `shading` | float | `0.45` | |
| `edges` | bool | `False` | Wireframe along the sampling grid |
| `edgecolor` | colour | `"black"` | |
| `edge_alpha` | float | `1` | |
| `edge_linewidth` | float | `1` | At the box centre |
| `hint_labels` | list of str | `[]` | Per sample, u major like `heights` |

### SurfaceTriOptions

`surface_tri(x, y, z, triangles=None, *, orient=None, colors=None, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour | `"blue"` | Flat colour, unless `colors` is passed |
| `cmap` | `Colormap` | `"viridis"` | For `colors` |
| `vmin`, `vmax` | float | `0`, `0` | Equal = the mesh's own range |
| `colorbar` | bool | `False` | Only with `colors` |
| `name` | str | `""` | Colorbar title when colour-mapped, else legend key |
| `show_legend` | bool | `True` | |
| `alpha` | float | `1` | |
| `shading` | float | `0.45` | Per face, both sides lit |
| `edges` | bool | `False` | All three edges of every triangle |
| `edgecolor` | colour | `"black"` | |
| `edge_alpha` | float | `1` | |
| `edge_linewidth` | float | `1` | At the box centre |
| `hint_labels` | list of str | `[]` | Per vertex |

### Scatter3DOptions

`scatter3d(x, y, z, *, colors=None, err=None, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour | `"blue"` | Flat colour, unless `colors` is passed |
| `size` | float | `20` | Marker diameter, constant at any depth |
| `marker` | `MarkerStyle` | `"circle"` | |
| `alpha` | float | `1` | |
| `depthshade` | float | `0` | Darken with distance; 0 = off |
| `cmap` | `Colormap` | `"viridis"` | For `colors` |
| `vmin`, `vmax` | float | `0`, `0` | Equal = the series' own range |
| `colorbar` | bool | `False` | Only with `colors` |
| `name` | str | `""` | Legend key; with `colors`, also the colorbar title |
| `show_legend` | bool | `True` | |
| `errorbar` | dict | | [ErrorBar3DOptions](#errorbar3doptions) |
| `hint_labels` | list of str | `[]` | |

### Line3DOptions

`line3d(x, y, z, *, colors=None, err=None, **options)`.

| Keyword | Type | Default | Meaning |
|---|---|---|---|
| `color` | colour | `"blue"` | Flat colour, unless `colors` is passed |
| `linewidth` | float | `1.5` | At the box centre; thinner with distance under perspective |
| `alpha` | float | `1` | |
| `loop` | bool | `False` | Close the path |
| `depthshade` | float | `0` | |
| `cmap` | `Colormap` | `"viridis"` | For `colors` |
| `vmin`, `vmax` | float | `0`, `0` | Equal = the path's own range |
| `colorbar` | bool | `False` | Only with `colors` |
| `name` | str | `""` | Legend key |
| `show_legend` | bool | `True` | |
| `errorbar` | dict | | [ErrorBar3DOptions](#errorbar3doptions) |
| `hint_labels` | list of str | `[]` | Per vertex |

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
| `box_alpha` | float | `0.25` | Block faces |
| `edge_alpha` | float | `0.5` | Block edges |

---

## Enums

Pass the value as a string (case, `_`, `-` and spaces are ignored), or use the class member
(`sextant.LineStyle.DASHED`). Members are `str` and compare equal to their value.

| Class | Values | Aliases |
|---|---|---|
| `LineStyle` | `solid`, `dashed`, `dotted`, `dashdot`, `none` | `-` `--` `:` `-.` `""` |
| `MarkerStyle` | `none`, `circle`, `square`, `triangle`, `cross`, `plus`, `diamond` | `o` `s` `^` `x` `+` `D` `d` `""` |
| `Colormap` | `viridis`, `plasma`, `inferno`, `magma`, `cividis`, `turbo`, `coolwarm`, `gray` | `grey` |
| `CapStyle` | `flat`, `arrow` | |
| `AxisPosition` | `auto` (bottom/left in 2D; the box silhouette in 3D), `low`, `mid`, `high` | |
| `LegendAnchor` | `inside_tl`, `inside_tr`, `inside_bl`, `inside_br`, `outside_tl`, `outside_tr`, `outside_bl`, `outside_br`, `outside_lt`, `outside_lb`, `outside_rt`, `outside_rb` | |
| `ColorbarAnchor` | `left`, `right`, `top`, `bottom` | |
| `HAlign` | `left`, `center`, `right` | `centre` |
| `PanelTheme` | `dark`, `light`, `classic` | |
| `Projection` | `orthographic`, `perspective` | `ortho`, `persp` |
| `PlaneOrientation` | `xy`, `yz`, `zx` | |
| `EventKind` | `close`, `mouse_down`, `mouse_up`, `mouse_move`, `scroll`, `key_down`, `key_up`, `resize`, `pick` | matplotlib's event names ([Events](api.md#events)) |
| `PickKind` | `none`, `line`, `scatter`, `scatter_z`, `bar`, `heatmap`, `bar3d`, `surface`, `surface_tri`, `scatter3d`, `line3d` | |
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
| `ScatterData` | `x`, `y` |
| `ScatterZData` | `x`, `y`, `z` |
| `BarData` | `x`, `height` (from `hist()`: bin centres and heights) |
| `HeatmapData` | `data` `(rows, cols)`, rounded to single precision; `xrange`, `yrange` `(lo, hi)` |
| `Bar3DData` | `orient` (`PlaneOrientation`), `u`, `v`, `heights` `(len(u), len(v))`, `bottoms` (same shape, or `None`) |
| `SurfaceData` | `orient`, `u`, `v`, `heights` `(len(u), len(v))` |
| `SurfaceTriData` | `x`, `y`, `z`, `tri` (`(n, 3)` `uint32`; the derived triangulation for an `orient=` mesh), `colors` (or `None`) |
| `Scatter3DData` | `x`, `y`, `z`, `colors` (or `None`) |
| `Line3DData` | `x`, `y`, `z`, `colors` (or `None`) |

---

## Event

What a callback connected with `fig.connect(kind, callback)` receives. Read-only. Every event has every
attribute; those its kind does not fill keep their defaults (NaN for data coordinates, -1 for indices).

| Attribute | Type | Filled for | Meaning |
|---|---|---|---|
| `kind` | `EventKind` | all | |
| `x`, `y` | float | pointer events | Logical pixels from the top left of the plot area |
| `inaxes` | `Axes`, `Axes3D` or `None` | pointer events | The subplot under the pointer: the same object `add_subplot()` returned |
| `axes` | int | pointer events | That subplot's index as `add_subplot()` takes it (a span: its first cell), or -1 |
| `has_data` | bool | pointer events | Whether `xdata`/`ydata`(/`zdata`) are set: over a 2D axes, or where a 3D pointer ray meets a `Plane2D` |
| `xdata`, `ydata`, `zdata` | float | pointer events | The pointer in data coordinates; NaN when not over data |
| `button` | int | `mouse_down`, `mouse_up`, `pick` | 0 left, 1 right, 2 middle |
| `double_click` | bool | `mouse_down` | |
| `mods` | int | input events | Modifier bits: 1 ctrl, 2 shift, 4 alt, 8 super |
| `modifiers` | tuple of str | input events | The same by name, in the order `"ctrl"`, `"shift"`, `"alt"`, `"super"` |
| `scroll_x`, `scroll_y` | float | `scroll` | Wheel notches, positive away from the user; queued ones are summed |
| `key` | str | `key_down`, `key_up` | `"a"`, `"A"`, `"ctrl+a"`, `"escape"`, `"f5"`, `"left"`, … |
| `width`, `height` | int | `resize` | The new plot-area size |
| `consumed` | `EventConsumed` | input events | What sextant itself did with the input (informational) |
| `pick_kind` | `PickKind` | `pick` | The kind of object clicked |
| `pick_object` | int | `pick` | Its index within the kind: the `i` of `<kind>_data(i)` (on the plane, when `pick_plane >= 0`) |
| `pick_index` | int | `pick` | The element: a point, bar, marker, vertex or surface sample; for a heatmap or `bar3d`, the flat cell index (`row * cols + col`) |
| `pick_row`, `pick_col` | int | `pick` | The cell of a heatmap or `bar3d` grid, else -1 |
| `pick_plane` | int | `pick` | `Axes3D.plane_at()` index of the plane the object is on, or -1 |

---

## Module functions

| Function | |
|---|---|
| `sextant.run()` | Wait until every open figure is closed, delivering their events |
| `sextant.poll_events()` | Process window events once and deliver queued events, without waiting. On macOS, main thread only |
| `sextant.set_message_handler(fn)` | Send sextant's diagnostics to `fn(message: str)`; `None` restores the default (`warnings.warn(message, RuntimeWarning)`). Returns the previous handler |
| `sextant.set_repr_formats(*formats)` | What a `Figure` offers Jupyter/IPython to display: `"png"` (the default), `"svg"`, both, or none |
