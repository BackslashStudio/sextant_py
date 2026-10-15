"""Type-checked by test_stubs.py with mypy --strict, never run.

Each `# type: ignore[code]` marks a call the stubs must reject; mypy runs with
--warn-unused-ignores, so an error that stops being reported fails the test.
"""

import numpy as np
import numpy.typing as npt

import sextant
from sextant import LineStyle


def plots() -> None:
    fig = sextant.Figure(width=800, height=600, title="t", theme="light", margins={"left": 0.1})
    ax: sextant.Axes = fig.add_subplot(2, 2, 1)

    # Chained calls keep their type.
    same: sextant.Axes = ax.line([0, 1, 2], [0, 1, 0], color="red", linestyle="--", linewidth=2.0) \
        .scatter(np.arange(3), np.ones(3), marker="o", errorbar={"capsize": 4.0}) \
        .set_title("title")
    ax.line([1, 2], linestyle=LineStyle.DASHED, hint_labels=["a", "b"])
    ax.bar([0, 1], [3, 4], color=(0.1, 0.2, 0.3), edgecolor="#336699")
    ax.heatmap(np.eye(3), (0.0, 1.0), (0.0, 1.0), cmap="viridis", colorbar=True)
    ax.legend(anchor="outside_tr", frameon=False)

    ax3: sextant.Axes3D = fig.add_subplot3d(2, 2, 2)
    ax3.surface("xy", [0, 1], [0, 1], [[0, 1], [1, 0]], cmap="coolwarm", edges=True)
    ax3.set_camera(azimuth=30.0, elevation=20.0, projection="perspective", target=(0.0, 0.0, 0.0))
    plane: sextant.Plane2D = ax3.plane("zx", 0.5, alpha=0.5)
    plane.line([0, 1], [1, 0])
    orient: sextant.PlaneOrientation = plane.orientation()

    ax.text("t", 0.5, 0.9, coords="fraction", ha="center", va=sextant.VAlign.TOP, edgecolor=None)
    ax.annotate(1.0, 2.0, "peak", 0.6, 0.8, xcoords="axes fraction", arrow={"head": "open", "arc": 0.2})
    td: sextant.TextData = ax.text_data(0)
    where_x: sextant.Coords = td.xcoords
    ax3.text2d("caption", 0.02, 0.95).annotate(1.0, 1.0, 1.0, "tip", 40.0, 30.0, arrow={"tail": "bar"})
    t3: sextant.Text3DData = ax3.text_data(0)

    data: sextant.LineData = ax.line_data(0)
    xs: npt.NDArray[np.float64] = data.x
    rgba: npt.NDArray[np.uint8] = fig.render_rgba(width=64, height=48, dpi=1.0)
    svg, report = fig.render_svg(max_splits=10)
    text: str = svg
    saved: sextant.SvgSaveReport | None = fig.savefig("out.svg", max_tests=100)
    fig.savefig("out.png", peel_layers=8)

    def on_event(ev: sextant.Event) -> None:
        where: sextant.Axes | sextant.Axes3D | None = ev.inaxes
        twin_y: float = ev.y2data
        whose: int = ev.pick_twin

    fig.connect("button_press_event", on_event)
    prev = sextant.set_message_handler(print)
    with fig as f:
        g: sextant.Figure = f

    # sextant 1.2: color cycle, colormaps by name, scales, sharing, twins, aspect.
    ax.line([0, 1], color=None).set_color_cycle(["red", (0.0, 0.5, 1.0)])
    ax.scatter_z([0], [0], [0], cmap="RdBu_r").scatter_z([0], [0], [0], cmap=sextant.Colormap.TWILIGHT)
    name: str = sextant.register_colormap("typing_map", ["red", "blue"], cyclic=True)
    names: list[str] = sextant.colormaps()
    ax.set_xscale("log", base=2.0).set_yscale(sextant.Scale.SYMLOG, linthresh=0.5).set_xtick_format("%.1f")
    scale: sextant.Scale = ax.xscale()
    ax.set_axes_style(minor_xticks=True, show_ytick_labels=None).grid(True, minor=True, minor_color="gray")
    other: sextant.Axes = fig.add_subplot(2, 2, 3).sharex(ax).sharey(ax)
    twin: sextant.Axes = ax.twinx()
    ax.set_aspect(1.0, "datalim")
    adj: sextant.Adjustable = ax.adjustable()

    # sextant 1.2 step 36c: markers, steps, per-point sizes, bar bases, reference lines.
    ax.scatter([0, 1], [0, 1], marker="x", sizes=[4.0, 8.0]).line([0, 1], drawstyle="steps-mid")
    ax.bar([0, 1], [1, 2], bottom=0.5, orientation=sextant.Orientation.HORIZONTAL)
    ax.axline(0.5, orientation="vertical", from_=0.2, to=0.8).axspan(0, 1, edgecolor="k", linewidth=1.0)
    level: float = ax.axline_data(0).value
    sizes = ax.scatter_data(0).sizes
    ax.set_bar_data(0, sextant.BarData([0, 1], [1, 2], bottom=None)).set_axspan_data(0, 1.0, 2.0)
    style: sextant.DrawStyle = sextant.DrawStyle.STEPS_PRE

    # sextant 1.2 steps 36d/36e: filled areas and arrow fields.
    ax.fill([0, 1], [1, 2], 0.5, exclude=[False, True], drawstyle="steps-post").fill([0, 1], [1, 2], [0, 1])
    ax.stack([0, 1], [[1, 2], [3, 4]], baseline="weighted_wiggle", colors=["red", (0.0, 0.0, 1.0)])
    ax.quiver([0, 1], [0, 1], [1, 1], [0, 1], [0.5, 1.0], amp=0.5, pivot="mid", cmap="plasma")
    mask = ax.fill_data(0).exclude
    arrows: sextant.QuiverData = ax.quiver_data(0)
    ax.set_fill_data(0, sextant.FillData([0, 1], [1, 2], [0, 0], exclude=None)).set_quiver_data(0, arrows)
    base: sextant.StackBaseline = sextant.StackBaseline.SYM

    # What the stubs must reject.
    ax.axline(0.5, orientation="diagonal")  # type: ignore[arg-type]
    ax.axline(0.5, xmin=0.2)  # type: ignore[call-arg]
    ax.line([0, 1], colour="red")  # type: ignore[call-arg]
    ax.line([0, 1], linestyle="dashy")  # type: ignore[arg-type]
    ax.line([0, 1], linewidth="thick")  # type: ignore[arg-type]
    ax.scatter([0], [0], errorbar={"capsise": 4.0})  # type: ignore[typeddict-unknown-key]
    ax3.bar3d("xz", [0], [0], [[1]])  # type: ignore[arg-type]
    fig.render_png(max_splits=3)  # type: ignore[call-arg]
    bad: sextant.Axes3D = ax.grid(True)  # type: ignore[assignment]
    ax.text("t", 0, 0, coords="pixels")  # type: ignore[arg-type]
    ax.annotate(0, 0, "t", 1, 1, arrow={"style": "->"})  # type: ignore[arg-type]
    ax.set_xscale("logit")  # type: ignore[arg-type]
    ax.set_xscale("log", subs=[2])  # type: ignore[call-arg]
    ax.set_aspect(1.0, "frame")  # type: ignore[arg-type]
    ax.stack([0, 1], [[1, 2]], baseline="silhouette")  # type: ignore[arg-type]
    ax.quiver([0], [0], [1], [0], pivot="head")  # type: ignore[arg-type]
    ax.quiver([0], [0], [1], [0], scale=2.0)  # type: ignore[call-arg]
