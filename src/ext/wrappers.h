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

    // The Python callable inside sextant's std::function (events.cpp).
    struct PyCallable;

    struct PyFigure {
        std::shared_ptr<FigureState> st;

        // Callbacks connected through this object, which owns them: the GC sees
        // them through the Figure type's tp_traverse, so a callback referring
        // to its figure is collectable, and they are disconnected when this
        // object goes. GIL-protected.
        struct Connection {
            int id;
            std::shared_ptr<PyCallable> cb;
        };
        std::vector<Connection> callbacks;

        explicit PyFigure(std::shared_ptr<FigureState> s) : st(std::move(s)) {}
        PyFigure(const PyFigure&) = delete;
        PyFigure& operator=(const PyFigure&) = delete;
        ~PyFigure();
    };

    // tp_traverse/tp_clear for the Figure type (events.cpp).
    int figure_traverse(PyObject* self, visitproc visit, void* arg);
    int figure_clear(PyObject* self);

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
    nb::typed<nb::object, Wrapped<T>> chain(SelfOf<T> self, F&& f) {
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

    // set_<kind>_visible(i, visible) and <kind>_visible(i) for one kind, from
    // T's own member functions; `set`/`get` are the Python names.
    template <class T>
    void bind_visible(nb::class_<Wrapped<T>>& cls, const char* set, const char* get,
                      std::size_t (T::*count)() const, T& (T::*setter)(std::size_t, bool),
                      bool (T::*getter)(std::size_t) const) {
        cls.def(set,
                [count, setter](SelfOf<T> self, std::int64_t i, bool visible) {
                    return chain(self, [&](T& t) { (t.*setter)(index(i, (t.*count)()), visible); });
                },
                "i"_a, "visible"_a,
                "Show or hide object i of this kind (its options' visible). A hidden object is\n"
                "not drawn or exported, has no legend key or colorbar, takes no hover or pick,\n"
                "and leaves the automatic limits; its data, color and draw order stay.")
            .def(get,
                 [count, getter](Wrapped<T>& w, std::int64_t i) {
                     return read(w, [&](T& t) { return (t.*getter)(index(i, (t.*count)())); });
                 },
                 "i"_a);
    }

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
    // empty optional part). Also what set_*_data() takes back. Any object is
    // stored; the types only name them for the stubs.
    using F64 = NumpyObject<double>;
    using F64OrNone = NumpyObject<double, true>;
    using U32 = NumpyObject<std::uint32_t>;
    using BoolOrNone = NumpyObject<bool, true>;
    struct PyLineData { F64 x, y; };
    // sizes/bottom: per point, None for none (the uniform size, bars on 0).
    // to_cpp() is what set_*_data(i, data) sends: None clears them, as in C++.
    struct PyScatterData {
        F64 x, y;
        F64OrNone sizes;
        sextant::ScatterData to_cpp() const;
    };
    struct PyScatterZData {
        F64 x, y, z;
        F64OrNone sizes;
        sextant::ScatterZData to_cpp() const;
    };
    struct PyBarData {
        F64 x, height;
        F64OrNone bottom;
        sextant::BarData to_cpp() const;
    };
    // exclude: one bool per point, None for none (FillOptions::exclude).
    struct PyFillData {
        F64 x, y1, y2;
        BoolOrNone exclude;
        sextant::FillData to_cpp() const;
    };
    // c: one value per arrow to colormap, None for a flat quiver.
    struct PyQuiverData {
        F64 x, y, u, v;
        F64OrNone c;
        sextant::QuiverData to_cpp() const;
    };
    struct PyHeatmapData {
        F64 data; // (rows, cols)
        sextant::Range xrange, yrange;
    };
    struct PyBar3DData {
        sextant::PlaneOrientation orient;
        F64 u, v, heights; // heights/bottoms (len(u), len(v))
        F64OrNone bottoms;
    };
    struct PySurfaceData {
        sextant::PlaneOrientation orient;
        F64 u, v, heights;
    };
    struct PySurfaceTriData { F64 x, y, z; U32 tri; F64OrNone colors; }; // tri (n, 3)
    struct PyScatter3DData {
        F64 x, y, z;
        F64OrNone colors, sizes;
        sextant::Scatter3DData to_cpp() const;
    };
    struct PyLine3DData { F64 x, y, z; F64OrNone colors; };
    // sextant::TextData with each Pos split into a value and its Coords.
    struct PyTextData {
        std::string text;
        double x = 0.0, y = 0.0;
        sextant::Coords xcoords = sextant::Coords::Data, ycoords = sextant::Coords::Data;
        bool arrow = false;
        double px = 0.0, py = 0.0;
        sextant::TextData to_cpp() const;
    };
    using PyText3DData = sextant::Text3DData; // plain fields, bound as they are
    // sextant::PolyData, CircleData, WedgeData and AxLineData with each Pos split into a
    // value and its Coords (as PyTextData).
    struct PyPolyData { F64 x, y; };
    struct PyCircleData {
        double cx = 0.0, cy = 0.0, r = 1.0;
        sextant::Coords cxcoords = sextant::Coords::Data, cycoords = sextant::Coords::Data,
                        rcoords = sextant::Coords::Data;
        sextant::CircleData to_cpp() const;
    };
    struct PyWedgeData {
        double cx = 0.0, cy = 0.0, r = 1.0, theta1 = 0.0, theta2 = 90.0, inner = 0.0;
        sextant::Coords cxcoords = sextant::Coords::Data, cycoords = sextant::Coords::Data,
                        rcoords = sextant::Coords::Data;
        sextant::WedgeData to_cpp() const;
    };
    struct PyAxLineData {
        double value = 0.0;
        sextant::AxLineMode mode = sextant::AxLineMode::Value;
        double x = 0.0, y = 0.0, x2 = 0.0, y2 = 0.0, slope = 0.0;
        sextant::Coords xcoords = sextant::Coords::Data, ycoords = sextant::Coords::Data,
                        x2coords = sextant::Coords::Data, y2coords = sextant::Coords::Data;
        sextant::AxLineData to_cpp() const;
    };
    using PyAxSpanData = sextant::AxSpanData;

    PyLineData to_python(sextant::LineData&& d);
    PyScatterData to_python(sextant::ScatterData&& d);
    PyScatterZData to_python(sextant::ScatterZData&& d);
    PyBarData to_python(sextant::BarData&& d);
    PyFillData to_python(sextant::FillData&& d);
    PyQuiverData to_python(sextant::QuiverData&& d);
    PyHeatmapData to_python(sextant::HeatmapData&& d);
    PyBar3DData to_python(sextant::Bar3DData&& d);
    PySurfaceData to_python(sextant::SurfaceData&& d);
    PySurfaceTriData to_python(sextant::SurfaceTriData&& d);
    PyScatter3DData to_python(sextant::Scatter3DData&& d);
    PyLine3DData to_python(sextant::Line3DData&& d);
    PyTextData to_python(sextant::TextData&& d);
    PyPolyData to_python(sextant::PolyData&& d);
    PyCircleData to_python(sextant::CircleData&& d);
    PyWedgeData to_python(sextant::WedgeData&& d);
    PyAxLineData to_python(sextant::AxLineData&& d);

    void bind_data(nb::module_& m);
    void bind_events(nb::module_& m, nb::class_<PyFigure>& fig);
    void bind_axes3d(nb::module_& m);
    void bind_plane(nb::module_& m);
} // namespace sextant_py
