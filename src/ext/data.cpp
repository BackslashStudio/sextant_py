// ErrorBar/ErrorBar3D and the read-back data classes.
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

        // Bar3D/surface heights as a (len(u), len(v)) array: the C++ layout, u major.
        F64 grid(std::vector<double>&& v, std::size_t nu, std::size_t nv) {
            return to_numpy(std::move(v), {nu, nv});
        }

        std::span<const double> s(const std::optional<Vec>& v) { return v ? v->span() : std::span<const double>{}; }

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

    PyScatterData to_python(sextant::ScatterData&& d) { return {vec(std::move(d.x)), vec(std::move(d.y))}; }

    PyScatterZData to_python(sextant::ScatterZData&& d) {
        return {vec(std::move(d.x)), vec(std::move(d.y)), vec(std::move(d.z))};
    }

    PyBarData to_python(sextant::BarData&& d) { return {vec(std::move(d.x)), vec(std::move(d.height))}; }

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
        return {vec(std::move(d.x)), vec(std::move(d.y)), vec(std::move(d.z)), vec_or_none(std::move(d.colors))};
    }

    PyLine3DData to_python(sextant::Line3DData&& d) {
        return {vec(std::move(d.x)), vec(std::move(d.y)), vec(std::move(d.z)), vec_or_none(std::move(d.colors))};
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

        nb::class_<PyScatterData>(m, "ScatterData")
            .def(nb::init<F64, F64>(), "x"_a, "y"_a)
            .def_rw("x", &PyScatterData::x)
            .def_rw("y", &PyScatterData::y)
            .def("__repr__",
                 [](const PyScatterData& d) { return repr_fields("ScatterData", {{"x", d.x}, {"y", d.y}}); });

        nb::class_<PyScatterZData>(m, "ScatterZData")
            .def(nb::init<F64, F64, F64>(), "x"_a, "y"_a, "z"_a)
            .def_rw("x", &PyScatterZData::x)
            .def_rw("y", &PyScatterZData::y)
            .def_rw("z", &PyScatterZData::z)
            .def("__repr__", [](const PyScatterZData& d) {
                return repr_fields("ScatterZData", {{"x", d.x}, {"y", d.y}, {"z", d.z}});
            });

        nb::class_<PyBarData>(m, "BarData")
            .def(nb::init<F64, F64>(), "x"_a, "height"_a)
            .def_rw("x", &PyBarData::x)
            .def_rw("height", &PyBarData::height)
            .def("__repr__",
                 [](const PyBarData& d) { return repr_fields("BarData", {{"x", d.x}, {"height", d.height}}); });

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

        nb::class_<PyScatter3DData>(m, "Scatter3DData", "colors is None for a flat series.")
            .def(nb::init<F64, F64, F64, F64OrNone>(),
                 "x"_a, "y"_a, "z"_a, "colors"_a.none() = nb::none())
            .def_rw("x", &PyScatter3DData::x)
            .def_rw("y", &PyScatter3DData::y)
            .def_rw("z", &PyScatter3DData::z)
            .def_rw("colors", &PyScatter3DData::colors, nb::arg().none())
            .def("__repr__", [](const PyScatter3DData& d) {
                return repr_fields("Scatter3DData", {{"x", d.x}, {"y", d.y}, {"z", d.z}, {"colors", d.colors}});
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
    }
} // namespace sextant_py
