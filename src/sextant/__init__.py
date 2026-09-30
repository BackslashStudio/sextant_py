"""sextant: scientific plotting from a C++20 core."""

import atexit as _atexit

from . import _sextant
from ._enums import (
    AxisPosition,
    CapStyle,
    ColorbarAnchor,
    Colormap,
    HAlign,
    LegendAnchor,
    LineStyle,
    MarkerStyle,
    PanelTheme,
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
    poll_events,
    run,
    set_message_handler,
)

# Close every window while the interpreter is still whole: a figure left to
# finalization would join its window thread against a half torn-down runtime.
_atexit.register(_sextant._shutdown)

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
    "Plane2D",
    "PlaneOrientation",
    "Projection",
    "Scatter3DData",
    "ScatterData",
    "ScatterZData",
    "SurfaceData",
    "SurfaceTriData",
    "poll_events",
    "run",
    "set_message_handler",
]
