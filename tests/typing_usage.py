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

    data: sextant.LineData = ax.line_data(0)
    xs: npt.NDArray[np.float64] = data.x
    rgba: npt.NDArray[np.uint8] = fig.render_rgba(width=64, height=48, dpi=1.0)
    svg, report = fig.render_svg(max_splits=10)
    text: str = svg
    saved: sextant.SvgSaveReport | None = fig.savefig("out.svg", max_tests=100)
    fig.savefig("out.png", peel_layers=8)

    def on_event(ev: sextant.Event) -> None:
        where: sextant.Axes | sextant.Axes3D | None = ev.inaxes

    fig.connect("button_press_event", on_event)
    prev = sextant.set_message_handler(print)
    with fig as f:
        g: sextant.Figure = f

    # What the stubs must reject.
    ax.line([0, 1], colour="red")  # type: ignore[call-arg]
    ax.line([0, 1], linestyle="dashy")  # type: ignore[arg-type]
    ax.line([0, 1], linewidth="thick")  # type: ignore[arg-type]
    ax.scatter([0], [0], errorbar={"capsise": 4.0})  # type: ignore[typeddict-unknown-key]
    ax3.bar3d("xz", [0], [0], [[1]])  # type: ignore[arg-type]
    fig.render_png(max_splits=3)  # type: ignore[call-arg]
    bad: sextant.Axes3D = ax.grid(True)  # type: ignore[assignment]
