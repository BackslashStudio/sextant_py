"""sextant: scientific plotting from a C++20 core."""

import atexit as _atexit

from . import _sextant
from ._enums import (
    AxisPosition,
    CapStyle,
    ColorbarAnchor,
    Colormap,
    EventConsumed,
    EventKind,
    HAlign,
    LegendAnchor,
    LineStyle,
    MarkerStyle,
    PanelTheme,
    PickKind,
    PlaneOrientation,
    Projection,
)
from ._sextant import (
    Axes,
    Axes3D,
    Bar3DData,
    BarData,
    ErrorBar,
    ErrorBar3D,
    Event,
    Figure,
    FrameStats,
    HeatmapData,
    Line3DData,
    LineData,
    Plane2D,
    Scatter3DData,
    ScatterData,
    ScatterZData,
    SurfaceData,
    SurfaceTriData,
    SvgSaveReport,
    poll_events,
    run,
    set_message_handler,
    set_repr_formats,
)

# Close every window while the interpreter is still whole: a figure left to
# finalization would join its window thread against a half torn-down runtime.
_atexit.register(_sextant._shutdown)

# In a Jupyter kernel this registers the "sextant" event loop, for %gui sextant.
from . import _interactive  # noqa: E402

__all__ = [
    "Axes",
    "Axes3D",
    "AxisPosition",
    "Bar3DData",
    "BarData",
    "CapStyle",
    "ColorbarAnchor",
    "Colormap",
    "ErrorBar",
    "ErrorBar3D",
    "Event",
    "EventConsumed",
    "EventKind",
    "Figure",
    "FrameStats",
    "HAlign",
    "HeatmapData",
    "LegendAnchor",
    "Line3DData",
    "LineData",
    "LineStyle",
    "MarkerStyle",
    "PanelTheme",
    "PickKind",
    "Plane2D",
    "PlaneOrientation",
    "Projection",
    "Scatter3DData",
    "ScatterData",
    "ScatterZData",
    "SurfaceData",
    "SurfaceTriData",
    "SvgSaveReport",
    "poll_events",
    "run",
    "set_message_handler",
    "set_repr_formats",
]
