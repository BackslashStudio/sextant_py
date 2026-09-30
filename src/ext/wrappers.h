// The C++ types behind the Python classes. Every wrapper holds its figure's
// state, so any of them keeps the figure (and its window) alive, as in
// matplotlib.
#pragma once

#include "casters.h"
#include "state.h"

#include <cstdint>
#include <optional>

namespace sextant_py {
    using namespace nb::literals;

    struct PyFigure {
        std::shared_ptr<FigureState> st;
    };

    // An object of a figure's graph: Axes, Axes3D or Plane2D.
    template <class T>
    struct Wrapped {
        std::shared_ptr<FigureState> st;
        std::shared_ptr<T> obj;
    };
    using PyAxes = Wrapped<sextant::Axes>;
    using PyAxes3D = Wrapped<sextant::Axes3D>;
    using PyPlane = Wrapped<sextant::Plane2D>;

    template <class T>
    using SelfOf = nb::pointer_and_handle<Wrapped<T>>;

    // Run a graph call on the object and return the same Python object, for
    // chaining. Arguments are converted before, with the GIL.
    template <class T, class F>
    nb::object chain(SelfOf<T> self, F&& f) {
        Wrapped<T>& w = *self.p;
        locked(*w.st, [&] { f(*w.obj); });
        return nb::borrow(self.h);
    }

    template <class T, class F>
    auto read(Wrapped<T>& w, F&& f) {
        return locked(*w.st, [&] { return f(*w.obj); });
    }

    // Python indexing: negative counts from the end. Call under the graph
    // lock, so `n` is the count the call will see.
    std::size_t index(std::int64_t i, std::size_t n);

    // `h` as an N-D float64 array, or TypeError naming `what`.
    template <int N>
    Array<N> as_array(nb::handle h, const char* what) {
        try {
            return nb::cast<Array<N>>(h);
        } catch (const nb::cast_error&) {
            throw nb::type_error((std::string(what) + ": expected a " + std::to_string(N) +
                                  "-D array of numbers, got " + nb::cast<std::string>(nb::repr(h))).c_str());
        }
    }

    // sextant::ErrorBar holds spans, so the arrays live here until the plotting
    // call has copied them.
    struct PyErrorBar {
        std::optional<Vec> x_cap_lo, x_cap_hi, x_box_lo, x_box_hi;
        std::optional<Vec> y_cap_lo, y_cap_hi, y_box_lo, y_box_hi;
        sextant::ErrorBar spans() const;
    };

    struct PyErrorBar3D {
        std::optional<Vec> x_cap_lo, x_cap_hi, x_box_lo, x_box_hi;
        std::optional<Vec> y_cap_lo, y_cap_hi, y_box_lo, y_box_hi;
        std::optional<Vec> z_cap_lo, z_cap_hi, z_box_lo, z_box_hi;
        sextant::ErrorBar3D spans() const;
    };

    // Read-back: numpy arrays that own the copies sextant returned (None for an
    // empty optional part). Also what set_*_data() takes back.
    struct PyLineData { nb::object x, y; };
    struct PyScatterData { nb::object x, y; };
    struct PyScatterZData { nb::object x, y, z; };
    struct PyBarData { nb::object x, height; };
    struct PyHeatmapData {
        nb::object data; // (rows, cols)
        sextant::Range xrange, yrange;
    };
    struct PyBar3DData {
        sextant::PlaneOrientation orient;
        nb::object u, v, heights, bottoms; // heights/bottoms (len(u), len(v)); bottoms may be None
    };
    struct PySurfaceData {
        sextant::PlaneOrientation orient;
        nb::object u, v, heights;
    };
    struct PySurfaceTriData { nb::object x, y, z, tri, colors; }; // tri (n, 3) uint32
    struct PyScatter3DData { nb::object x, y, z, colors; };
    struct PyLine3DData { nb::object x, y, z, colors; };

    PyLineData to_python(sextant::LineData&& d);
    PyScatterData to_python(sextant::ScatterData&& d);
    PyScatterZData to_python(sextant::ScatterZData&& d);
    PyBarData to_python(sextant::BarData&& d);
    PyHeatmapData to_python(sextant::HeatmapData&& d);
    PyBar3DData to_python(sextant::Bar3DData&& d);
    PySurfaceData to_python(sextant::SurfaceData&& d);
    PySurfaceTriData to_python(sextant::SurfaceTriData&& d);
    PyScatter3DData to_python(sextant::Scatter3DData&& d);
    PyLine3DData to_python(sextant::Line3DData&& d);

    void bind_data(nb::module_& m);
    void bind_axes3d(nb::module_& m);
    void bind_plane(nb::module_& m);
} // namespace sextant_py
