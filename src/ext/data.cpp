// ErrorBar/ErrorBar3D and the read-back data classes.
#include "options.h"
#include "wrappers.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <string>

namespace sextant_py {
    namespace {
        F64 vec(std::vector<double>&& v) {
            const std::size_t n = v.size();
            return to_numpy(std::move(v), {n});
        }

        // None for an empty optional part (bottoms, colors).
        F64OrNone vec_or_none(std::vector<double>&& v) { return v.empty() ? nb::none() : nb::object(vec(std::move(v))); }

        // A mask as a numpy bool array, None when empty.
        BoolOrNone bools_or_none(const std::vector<bool>& v) {
            if (v.empty()) return nb::none();
            const std::size_t n = v.size();
            auto* heap = new bool[n];
            for (std::size_t i = 0; i < n; ++i) heap[i] = v[i];
            nb::capsule owner(heap, [](void* p) noexcept { delete[] static_cast<bool*>(p); });
            return nb::cast(nb::ndarray<nb::numpy, bool>(heap, {n}, owner));
        }

        // Bar3D/surface heights as a (len(u), len(v)) array: the C++ layout, u major.
        F64 grid(std::vector<double>&& v, std::size_t nu, std::size_t nv) {
            return to_numpy(std::move(v), {nu, nv});
        }

        std::span<const double> s(const std::optional<Vec>& v) { return v ? v->span() : std::span<const double>{}; }

        // A read-back field as the C++ vector (empty for None).
        std::vector<double> owned(const nb::object& o, const char* what) {
            if (o.is_none()) return {};
            const auto a = as_array<1>(o, what);
            return {a.span().begin(), a.span().end()};
        }

        std::string repr_fields(const char* cls, std::initializer_list<std::pair<const char*, nb::handle>> fields) {
            std::string out = std::string(cls) + "(";
            bool first = true;
            for (const auto& [name, value] : fields) {
                out += (first ? "" : ", ") + std::string(name) + "=" + nb::cast<std::string>(nb::repr(value));
                first = false;
            }
            return out + ")";
        }
    } // namespace

    std::size_t index(std::int64_t i, std::size_t n) {
        if (i < 0) i += static_cast<std::int64_t>(n);
        if (i < 0 || static_cast<std::size_t>(i) >= n)
            throw nb::index_error(("index out of range (" + std::to_string(n) + " objects)").c_str());
        return static_cast<std::size_t>(i);
    }

    sextant::ErrorBar PyErrorBar::spans() const {
        return {s(x_cap_lo), s(x_cap_hi), s(x_box_lo), s(x_box_hi),
                s(y_cap_lo), s(y_cap_hi), s(y_box_lo), s(y_box_hi)};
    }

    sextant::ErrorBar3D PyErrorBar3D::spans() const {
        return {s(x_cap_lo), s(x_cap_hi), s(x_box_lo), s(x_box_hi),
                s(y_cap_lo), s(y_cap_hi), s(y_box_lo), s(y_box_hi),
                s(z_cap_lo), s(z_cap_hi), s(z_box_lo), s(z_box_hi)};
    }

    PyLineData to_python(sextant::LineData&& d) { return {vec(std::move(d.x)), vec(std::move(d.y))}; }

    PyScatterData to_python(sextant::ScatterData&& d) {
        return {vec(std::move(d.x)), vec(std::move(d.y)), vec_or_none(std::move(d.sizes))};
    }

    PyScatterZData to_python(sextant::ScatterZData&& d) {
        return {vec(std::move(d.x)), vec(std::move(d.y)), vec(std::move(d.z)), vec_or_none(std::move(d.sizes))};
    }

    PyBarData to_python(sextant::BarData&& d) {
        return {vec(std::move(d.x)), vec(std::move(d.height)), vec_or_none(std::move(d.bottom))};
    }

    PyFillData to_python(sextant::FillData&& d) {
        return {vec(std::move(d.x)), vec(std::move(d.y1)), vec(std::move(d.y2)), bools_or_none(d.exclude)};
    }

    PyQuiverData to_python(sextant::QuiverData&& d) {
        return {vec(std::move(d.x)), vec(std::move(d.y)), vec(std::move(d.u)), vec(std::move(d.v)),
                vec_or_none(std::move(d.c))};
    }

    sextant::FillData PyFillData::to_cpp() const {
        sextant::FillData d{owned(x, "FillData.x"), owned(y1, "FillData.y1"), owned(y2, "FillData.y2"), {}};
        if (!bools_from_python(exclude, d.exclude))
            throw nb::type_error(("FillData.exclude: expected a 1-D array of booleans or None, got " +
                                  nb::cast<std::string>(nb::repr(exclude))).c_str());
        return d;
    }

    sextant::QuiverData PyQuiverData::to_cpp() const {
        return {owned(x, "QuiverData.x"), owned(y, "QuiverData.y"), owned(u, "QuiverData.u"),
                owned(v, "QuiverData.v"), owned(c, "QuiverData.c")};
    }

    sextant::ScatterData PyScatterData::to_cpp() const {
        return {owned(x, "ScatterData.x"), owned(y, "ScatterData.y"), owned(sizes, "ScatterData.sizes")};
    }

    sextant::ScatterZData PyScatterZData::to_cpp() const {
        return {owned(x, "ScatterZData.x"), owned(y, "ScatterZData.y"), owned(z, "ScatterZData.z"),
                owned(sizes, "ScatterZData.sizes")};
    }

    sextant::BarData PyBarData::to_cpp() const {
        return {owned(x, "BarData.x"), owned(height, "BarData.height"), owned(bottom, "BarData.bottom")};
    }

    sextant::Scatter3DData PyScatter3DData::to_cpp() const {
        return {owned(x, "Scatter3DData.x"), owned(y, "Scatter3DData.y"), owned(z, "Scatter3DData.z"),
                owned(colors, "Scatter3DData.colors"), owned(sizes, "Scatter3DData.sizes")};
    }

    PyHeatmapData to_python(sextant::HeatmapData&& d) {
        const auto rows = static_cast<std::size_t>(d.rows), cols = static_cast<std::size_t>(d.cols);
        return {to_numpy(std::move(d.data), {rows, cols}), d.xrange, d.yrange};
    }

    PyBar3DData to_python(sextant::Bar3DData&& d) {
        const std::size_t nu = d.u.size(), nv = d.v.size();
        nb::object bottoms = d.bottoms.empty() ? nb::none() : grid(std::move(d.bottoms), nu, nv);
        return {d.orient, vec(std::move(d.u)), vec(std::move(d.v)), grid(std::move(d.heights), nu, nv), bottoms};
    }

    PySurfaceData to_python(sextant::SurfaceData&& d) {
        const std::size_t nu = d.u.size(), nv = d.v.size();
        return {d.orient, vec(std::move(d.u)), vec(std::move(d.v)), grid(std::move(d.heights), nu, nv)};
    }

    PySurfaceTriData to_python(sextant::SurfaceTriData&& d) {
        const std::size_t n = d.tri.size() / 3;
        return {vec(std::move(d.x)), vec(std::move(d.y)), vec(std::move(d.z)),
                to_numpy(std::move(d.tri), {n, 3}), vec_or_none(std::move(d.colors))};
    }

    PyScatter3DData to_python(sextant::Scatter3DData&& d) {
        return {vec(std::move(d.x)), vec(std::move(d.y)), vec(std::move(d.z)), vec_or_none(std::move(d.colors)),
                vec_or_none(std::move(d.sizes))};
    }

    PyLine3DData to_python(sextant::Line3DData&& d) {
        return {vec(std::move(d.x)), vec(std::move(d.y)), vec(std::move(d.z)), vec_or_none(std::move(d.colors))};
    }

    PyTextData to_python(sextant::TextData&& d) {
        return {std::move(d.text), d.x.v, d.y.v, d.x.space, d.y.space, d.arrow, d.px, d.py};
    }

    sextant::TextData PyTextData::to_cpp() const {
        sextant::TextData d;
        d.text = text;
        d.x = {x};
        d.x.space = xcoords;
        d.y = {y};
        d.y.space = ycoords;
        d.arrow = arrow;
        d.px = px;
        d.py = py;
        return d;
    }

    void bind_data(nb::module_& m) {
        using OV = std::optional<Vec>;
        nb::class_<PyErrorBar>(m, "ErrorBar",
                               "Per-point error-bar data for line/scatter/scatter_z/bar (err=...).\n"
                               "Offsets from the point, magnitude-valued; one end given means symmetric.\n"
                               "cap_* draw the capped whisker, box_* a box; zero draws nothing on that side.\n"
                               "Each given array holds one entry per point.")
            .def(nb::init<OV, OV, OV, OV, OV, OV, OV, OV>(), nb::kw_only(),
                 "x_cap_lo"_a = nb::none(), "x_cap_hi"_a = nb::none(), "x_box_lo"_a = nb::none(),
                 "x_box_hi"_a = nb::none(), "y_cap_lo"_a = nb::none(), "y_cap_hi"_a = nb::none(),
                 "y_box_lo"_a = nb::none(), "y_box_hi"_a = nb::none());

        nb::class_<PyErrorBar3D>(m, "ErrorBar3D",
                                 "ErrorBar for scatter3d/line3d, with a z axis. Same rules as ErrorBar.")
            .def(nb::init<OV, OV, OV, OV, OV, OV, OV, OV, OV, OV, OV, OV>(), nb::kw_only(),
                 "x_cap_lo"_a = nb::none(), "x_cap_hi"_a = nb::none(), "x_box_lo"_a = nb::none(),
                 "x_box_hi"_a = nb::none(), "y_cap_lo"_a = nb::none(), "y_cap_hi"_a = nb::none(),
                 "y_box_lo"_a = nb::none(), "y_box_hi"_a = nb::none(), "z_cap_lo"_a = nb::none(),
                 "z_cap_hi"_a = nb::none(), "z_box_lo"_a = nb::none(), "z_box_hi"_a = nb::none());

        nb::class_<PyLineData>(m, "LineData")
            .def(nb::init<F64, F64>(), "x"_a, "y"_a)
            .def_rw("x", &PyLineData::x)
            .def_rw("y", &PyLineData::y)
            .def("__repr__", [](const PyLineData& d) { return repr_fields("LineData", {{"x", d.x}, {"y", d.y}}); });

        nb::class_<PyScatterData>(m, "ScatterData",
                                  "sizes is None when every marker takes the call's `size`, else one\n"
                                  "diameter per point. set_scatter_data(i, data) takes it as given (None\n"
                                  "clears it); the (x, y) form keeps it while the count holds.")
            .def(nb::init<F64, F64, F64OrNone>(), "x"_a, "y"_a, "sizes"_a.none() = nb::none())
            .def_rw("x", &PyScatterData::x)
            .def_rw("y", &PyScatterData::y)
            .def_rw("sizes", &PyScatterData::sizes, nb::arg().none())
            .def("__repr__", [](const PyScatterData& d) {
                return repr_fields("ScatterData", {{"x", d.x}, {"y", d.y}, {"sizes", d.sizes}});
            });

        nb::class_<PyScatterZData>(m, "ScatterZData", "sizes as ScatterData's.")
            .def(nb::init<F64, F64, F64, F64OrNone>(), "x"_a, "y"_a, "z"_a, "sizes"_a.none() = nb::none())
            .def_rw("x", &PyScatterZData::x)
            .def_rw("y", &PyScatterZData::y)
            .def_rw("z", &PyScatterZData::z)
            .def_rw("sizes", &PyScatterZData::sizes, nb::arg().none())
            .def("__repr__", [](const PyScatterZData& d) {
                return repr_fields("ScatterZData", {{"x", d.x}, {"y", d.y}, {"z", d.z}, {"sizes", d.sizes}});
            });

        nb::class_<PyBarData>(m, "BarData",
                              "bottom is None when the bars stand on 0, else one base per bar (a single\n"
                              "bottom= value comes back repeated). set_bar_data(i, data) takes it as\n"
                              "given (None clears it); the (x, height) form keeps it while the count holds.")
            .def(nb::init<F64, F64, F64OrNone>(), "x"_a, "height"_a, "bottom"_a.none() = nb::none())
            .def_rw("x", &PyBarData::x)
            .def_rw("height", &PyBarData::height)
            .def_rw("bottom", &PyBarData::bottom, nb::arg().none())
            .def("__repr__", [](const PyBarData& d) {
                return repr_fields("BarData", {{"x", d.x}, {"height", d.height}, {"bottom", d.bottom}});
            });

        nb::class_<PyFillData>(m, "FillData",
                               "fill(): the run coordinate x (y for a vertical fill) and the two curves,\n"
                               "one value each per point (a number y2 comes back repeated). exclude is\n"
                               "None when no point is left out, else one bool per point.\n"
                               "set_fill_data(i, data) takes it as given (None clears it); the\n"
                               "(x, y1, y2) form keeps it while the count holds.")
            .def(nb::init<F64, F64, F64, BoolOrNone>(), "x"_a, "y1"_a, "y2"_a, "exclude"_a.none() = nb::none())
            .def_rw("x", &PyFillData::x)
            .def_rw("y1", &PyFillData::y1)
            .def_rw("y2", &PyFillData::y2)
            .def_rw("exclude", &PyFillData::exclude, nb::arg().none())
            .def("__repr__", [](const PyFillData& d) {
                return repr_fields("FillData", {{"x", d.x}, {"y1", d.y1}, {"y2", d.y2}, {"exclude", d.exclude}});
            });

        nb::class_<PyQuiverData>(m, "QuiverData",
                                 "quiver(): each arrow's anchor (x, y) and vector (u, v); c is None for a\n"
                                 "flat quiver, else one value per arrow, colormapped.\n"
                                 "set_quiver_data(i, data) takes c as given (None makes it flat); the\n"
                                 "(x, y, u, v) form keeps it while the count holds. The amp stays.")
            .def(nb::init<F64, F64, F64, F64, F64OrNone>(), "x"_a, "y"_a, "u"_a, "v"_a, "c"_a.none() = nb::none())
            .def_rw("x", &PyQuiverData::x)
            .def_rw("y", &PyQuiverData::y)
            .def_rw("u", &PyQuiverData::u)
            .def_rw("v", &PyQuiverData::v)
            .def_rw("c", &PyQuiverData::c, nb::arg().none())
            .def("__repr__", [](const PyQuiverData& d) {
                return repr_fields("QuiverData", {{"x", d.x}, {"y", d.y}, {"u", d.u}, {"v", d.v}, {"c", d.c}});
            });

        nb::class_<PyAxLineData>(m, "AxLineData",
                                 "Axes.axline(): where the line is, on the axis its orientation names.")
            .def("__init__", [](PyAxLineData* p, double value) { new (p) PyAxLineData{value}; }, "value"_a)
            .def_rw("value", &PyAxLineData::value)
            .def("__repr__", [](const PyAxLineData& d) {
                return repr_fields("AxLineData", {{"value", nb::cast(d.value)}});
            });

        nb::class_<PyAxSpanData>(m, "AxSpanData", "Axes.axspan(): the band's ends, lo <= hi as stored.")
            .def("__init__", [](PyAxSpanData* p, double lo, double hi) { new (p) PyAxSpanData{lo, hi}; },
                 "lo"_a, "hi"_a)
            .def_rw("lo", &PyAxSpanData::lo)
            .def_rw("hi", &PyAxSpanData::hi)
            .def("__repr__", [](const PyAxSpanData& d) {
                return repr_fields("AxSpanData", {{"lo", nb::cast(d.lo)}, {"hi", nb::cast(d.hi)}});
            });

        nb::class_<PyHeatmapData>(m, "HeatmapData",
                                  "data is (rows, cols), row-major as passed; values come back rounded\n"
                                  "to float32, which is how sextant stores them.")
            .def(nb::init<F64, sextant::Range, sextant::Range>(), "data"_a, "xrange"_a, "yrange"_a)
            .def_rw("data", &PyHeatmapData::data)
            .def_rw("xrange", &PyHeatmapData::xrange)
            .def_rw("yrange", &PyHeatmapData::yrange)
            .def("__repr__", [](const PyHeatmapData& d) {
                return repr_fields("HeatmapData", {{"data", d.data},
                                                   {"xrange", nb::cast(d.xrange)},
                                                   {"yrange", nb::cast(d.yrange)}});
            });

        nb::class_<PyBar3DData>(m, "Bar3DData",
                                "heights and bottoms are (len(u), len(v)); bottoms is None when every bar\n"
                                "stands on the bar3d() call's `bottom`.")
            .def(nb::init<sextant::PlaneOrientation, F64, F64, F64, F64OrNone>(),
                 "orient"_a, "u"_a, "v"_a, "heights"_a, "bottoms"_a.none() = nb::none())
            .def_rw("orient", &PyBar3DData::orient)
            .def_rw("u", &PyBar3DData::u)
            .def_rw("v", &PyBar3DData::v)
            .def_rw("heights", &PyBar3DData::heights)
            .def_rw("bottoms", &PyBar3DData::bottoms, nb::arg().none())
            .def("__repr__", [](const PyBar3DData& d) {
                return repr_fields("Bar3DData", {{"orient", nb::cast(d.orient)}, {"u", d.u}, {"v", d.v},
                                                 {"heights", d.heights}, {"bottoms", d.bottoms}});
            });

        nb::class_<PySurfaceData>(m, "SurfaceData", "heights is (len(u), len(v)).")
            .def(nb::init<sextant::PlaneOrientation, F64, F64, F64>(),
                 "orient"_a, "u"_a, "v"_a, "heights"_a)
            .def_rw("orient", &PySurfaceData::orient)
            .def_rw("u", &PySurfaceData::u)
            .def_rw("v", &PySurfaceData::v)
            .def_rw("heights", &PySurfaceData::heights)
            .def("__repr__", [](const PySurfaceData& d) {
                return repr_fields("SurfaceData", {{"orient", nb::cast(d.orient)}, {"u", d.u}, {"v", d.v},
                                                   {"heights", d.heights}});
            });

        nb::class_<PySurfaceTriData>(m, "SurfaceTriData",
                                     "tri is (n, 3) uint32: as passed, or the Delaunay triangulation made from\n"
                                     "an orientation. colors is None for a flat mesh.")
            .def(nb::init<F64, F64, F64, U32, F64OrNone>(),
                 "x"_a, "y"_a, "z"_a, "tri"_a, "colors"_a.none() = nb::none())
            .def_rw("x", &PySurfaceTriData::x)
            .def_rw("y", &PySurfaceTriData::y)
            .def_rw("z", &PySurfaceTriData::z)
            .def_rw("tri", &PySurfaceTriData::tri)
            .def_rw("colors", &PySurfaceTriData::colors, nb::arg().none())
            .def("__repr__", [](const PySurfaceTriData& d) {
                return repr_fields("SurfaceTriData", {{"x", d.x}, {"y", d.y}, {"z", d.z}, {"tri", d.tri},
                                                      {"colors", d.colors}});
            });

        nb::class_<PyScatter3DData>(m, "Scatter3DData",
                                    "colors is None for a flat series; sizes as ScatterData's.")
            .def(nb::init<F64, F64, F64, F64OrNone, F64OrNone>(),
                 "x"_a, "y"_a, "z"_a, "colors"_a.none() = nb::none(), "sizes"_a.none() = nb::none())
            .def_rw("x", &PyScatter3DData::x)
            .def_rw("y", &PyScatter3DData::y)
            .def_rw("z", &PyScatter3DData::z)
            .def_rw("colors", &PyScatter3DData::colors, nb::arg().none())
            .def_rw("sizes", &PyScatter3DData::sizes, nb::arg().none())
            .def("__repr__", [](const PyScatter3DData& d) {
                return repr_fields("Scatter3DData", {{"x", d.x}, {"y", d.y}, {"z", d.z}, {"colors", d.colors},
                                                     {"sizes", d.sizes}});
            });

        nb::class_<PyLine3DData>(m, "Line3DData", "colors is None for a flat path.")
            .def(nb::init<F64, F64, F64, F64OrNone>(),
                 "x"_a, "y"_a, "z"_a, "colors"_a.none() = nb::none())
            .def_rw("x", &PyLine3DData::x)
            .def_rw("y", &PyLine3DData::y)
            .def_rw("z", &PyLine3DData::z)
            .def_rw("colors", &PyLine3DData::colors, nb::arg().none())
            .def("__repr__", [](const PyLine3DData& d) {
                return repr_fields("Line3DData", {{"x", d.x}, {"y", d.y}, {"z", d.z}, {"colors", d.colors}});
            });

        using sextant::Coords;
        nb::class_<PyTextData>(m, "TextData",
                               "Axes.text() and annotate(): the string, its position (each coordinate a\n"
                               "value and what it is in, 'data' or 'fraction' of the frame), and for an\n"
                               "annotation (arrow=True) the data point (px, py) the arrow points at.")
            .def(nb::init<std::string, double, double, Coords, Coords, bool, double, double>(),
                 "text"_a, "x"_a, "y"_a, "xcoords"_a = Coords::Data, "ycoords"_a = Coords::Data,
                 "arrow"_a = false, "px"_a = 0.0, "py"_a = 0.0)
            .def_rw("text", &PyTextData::text)
            .def_rw("x", &PyTextData::x)
            .def_rw("y", &PyTextData::y)
            .def_rw("xcoords", &PyTextData::xcoords)
            .def_rw("ycoords", &PyTextData::ycoords)
            .def_rw("arrow", &PyTextData::arrow)
            .def_rw("px", &PyTextData::px)
            .def_rw("py", &PyTextData::py)
            .def("__repr__", [](const PyTextData& d) {
                return repr_fields("TextData", {{"text", nb::cast(d.text)}, {"x", nb::cast(d.x)}, {"y", nb::cast(d.y)},
                                                {"xcoords", nb::cast(d.xcoords)}, {"ycoords", nb::cast(d.ycoords)},
                                                {"arrow", nb::cast(d.arrow)}, {"px", nb::cast(d.px)},
                                                {"py", nb::cast(d.py)}});
            });

        nb::class_<PyText3DData>(m, "Text3DData",
                                 "Axes3D.text(), text2d() and annotate(): the string and where it is.\n"
                                 "  text():     (x, y, z) the data point it is drawn at.\n"
                                 "  text2d():   in_frame=True; (x, y) fractions of the frame, z unused.\n"
                                 "  annotate(): arrow=True; (x, y, z) the point, the text (dx, dy) pixels\n"
                                 "              from it, y up.")
            .def("__init__",
                 [](PyText3DData* p, std::string text, double x, double y, double z, bool in_frame, bool arrow,
                    double dx, double dy) {
                     new (p) PyText3DData{std::move(text), x, y, z, in_frame, arrow, dx, dy};
                 },
                 "text"_a, "x"_a, "y"_a, "z"_a = 0.0, "in_frame"_a = false, "arrow"_a = false, "dx"_a = 0.0,
                 "dy"_a = 0.0)
            .def_rw("text", &PyText3DData::text)
            .def_rw("x", &PyText3DData::x)
            .def_rw("y", &PyText3DData::y)
            .def_rw("z", &PyText3DData::z)
            .def_rw("in_frame", &PyText3DData::in_frame)
            .def_rw("arrow", &PyText3DData::arrow)
            .def_rw("dx", &PyText3DData::dx)
            .def_rw("dy", &PyText3DData::dy)
            .def("__repr__", [](const PyText3DData& d) {
                return repr_fields("Text3DData", {{"text", nb::cast(d.text)}, {"x", nb::cast(d.x)},
                                                  {"y", nb::cast(d.y)}, {"z", nb::cast(d.z)},
                                                  {"in_frame", nb::cast(d.in_frame)}, {"arrow", nb::cast(d.arrow)},
                                                  {"dx", nb::cast(d.dx)}, {"dy", nb::cast(d.dy)}});
            });
    }
} // namespace sextant_py
