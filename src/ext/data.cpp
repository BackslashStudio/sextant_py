// ErrorBar/ErrorBar3D and the read-back data classes.
#include "options.h"
#include "wrappers.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

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

    PyPieData to_python(sextant::PieData&& d) {
        return {vec(std::move(d.values)), std::move(d.labels), vec_or_none(std::move(d.explode))};
    }

    sextant::PieData PyPieData::to_cpp() const {
        return {owned(values, "PieData.values"), labels, owned(explode, "PieData.explode")};
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
        const auto ch = static_cast<std::size_t>(d.channels);
        F64 data = ch > 1 ? to_numpy(std::move(d.data), {rows, cols, ch}) : to_numpy(std::move(d.data), {rows, cols});
        return {std::move(data), d.xrange, d.yrange, vec_or_none(std::move(d.x_edges)),
                vec_or_none(std::move(d.y_edges))};
    }

    sextant::HeatmapData PyHeatmapData::to_cpp() const {
        const int nd = nb::cast<int>(data.attr("ndim"));
        if (nd != 2 && nd != 3)
            throw nb::type_error("HeatmapData.data: expected a (rows, cols) or (rows, cols, channels) array");
        const auto g = nb::cast<Grid>(data);
        sextant::HeatmapData h;
        h.data.assign(g.span().begin(), g.span().end());
        h.rows = static_cast<int>(g.shape(0));
        h.cols = static_cast<int>(g.shape(1));
        h.channels = nd == 3 ? static_cast<int>(g.shape(2)) : 1;
        h.xrange = xrange;
        h.yrange = yrange;
        h.x_edges = owned(x_edges, "HeatmapData.x_edges");
        h.y_edges = owned(y_edges, "HeatmapData.y_edges");
        return h;
    }

    sextant::ContourData PyContourData::to_cpp() const {
        auto xs = owned(x, "ContourData.x"), ys = owned(y, "ContourData.y");
        const auto zs = as_array<2>(z, "ContourData.z");
        if (zs.shape(0) != ys.size() || zs.shape(1) != xs.size())
            throw nb::value_error("ContourData.z: must be (len(y), len(x))");
        return {std::move(xs), std::move(ys), {zs.span().begin(), zs.span().end()},
                owned(levels, "ContourData.levels"), filled};
    }

    Triangles triangles(nb::handle obj, const std::string& ctx) {
        nb::module_ np = nb::module_::import_("numpy");
        nb::object arr = np.attr("asarray")(obj);
        const nb::object dtype = arr.attr("dtype");
        const std::string kind = nb::cast<std::string>(dtype.attr("kind"));
        if (kind != "i" && kind != "u")
            throw nb::type_error((ctx + ": triangles must be integers, got dtype " +
                                  nb::cast<std::string>(nb::str(dtype))).c_str());
        if (nb::cast<std::size_t>(arr.attr("size")) > 0) {
            const double lo = nb::cast<double>(nb::int_(arr.attr("min")()));
            const double hi = nb::cast<double>(nb::int_(arr.attr("max")()));
            if (lo < 0 || hi > std::numeric_limits<std::uint32_t>::max())
                throw nb::value_error((ctx + ": a triangle index is negative or does not fit uint32").c_str());
        }
        arr = np.attr("ascontiguousarray")(arr.attr("reshape")(-1), "dtype"_a = np.attr("uint32"));
        auto nd = nb::cast<nb::ndarray<const std::uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>>(arr);
        return {arr, {nd.data(), nd.size()}};
    }

    PyHeatmapTriData to_python(sextant::HeatmapTriData&& d) {
        const std::size_t n = d.tri.size() / 3;
        return {vec(std::move(d.x)), vec(std::move(d.y)), vec(std::move(d.values)),
                to_numpy(std::move(d.tri), {n, 3})};
    }

    PyContourData to_python(sextant::ContourData&& d) {
        const std::size_t rows = d.y.size(), cols = d.x.size();
        return {vec(std::move(d.x)), vec(std::move(d.y)), to_numpy(std::move(d.z), {rows, cols}),
                vec(std::move(d.levels)), d.filled};
    }

    PyQuiver3DData to_python(sextant::Quiver3DData&& d) {
        return {vec(std::move(d.x)), vec(std::move(d.y)), vec(std::move(d.z)), vec(std::move(d.u)),
                vec(std::move(d.v)), vec(std::move(d.w)), vec_or_none(std::move(d.c))};
    }

    sextant::Quiver3DData PyQuiver3DData::to_cpp() const {
        return {owned(x, "Quiver3DData.x"), owned(y, "Quiver3DData.y"), owned(z, "Quiver3DData.z"),
                owned(u, "Quiver3DData.u"), owned(v, "Quiver3DData.v"), owned(w, "Quiver3DData.w"),
                owned(c, "Quiver3DData.c")};
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


    namespace {
        sextant::Pos at(double v, sextant::Coords c) {
            sextant::Pos p(v);
            p.space = c;
            return p;
        }
    } // namespace

    PyPolyData to_python(sextant::PolyData&& d) { return {vec(std::move(d.x)), vec(std::move(d.y))}; }

    PyCircleData to_python(sextant::CircleData&& d) {
        return {d.cx.v, d.cy.v, d.r.v, d.cx.space, d.cy.space, d.r.space};
    }

    PyWedgeData to_python(sextant::WedgeData&& d) {
        return {d.cx.v, d.cy.v, d.r.v, d.theta1, d.theta2, d.inner, d.cx.space, d.cy.space, d.r.space};
    }

    PyAxLineData to_python(sextant::AxLineData&& d) {
        return {d.value, d.mode, d.x.v, d.y.v, d.x2.v, d.y2.v, d.slope, d.x.space, d.y.space, d.x2.space, d.y2.space};
    }

    sextant::CircleData PyCircleData::to_cpp() const {
        return {at(cx, cxcoords), at(cy, cycoords), at(r, rcoords)};
    }

    sextant::WedgeData PyWedgeData::to_cpp() const {
        return {at(cx, cxcoords), at(cy, cycoords), at(r, rcoords), theta1, theta2, inner};
    }

    sextant::AxLineData PyAxLineData::to_cpp() const {
        sextant::AxLineData d;
        d.value = value;
        d.mode = mode;
        d.x = at(x, xcoords);
        d.y = at(y, ycoords);
        d.x2 = at(x2, x2coords);
        d.y2 = at(y2, y2coords);
        d.slope = slope;
        return d;
    }

    void bind_data(nb::module_& m) {
        using sextant::Coords;
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
                                 "Axes.axline(): mode 'value' is the line at `value` (on the axis its\n"
                                 "orientation names); 'point_slope' goes through (x, y) with `slope`;\n"
                                 "'two_points' through (x, y) and (x2, y2). Each coordinate has its coords\n"
                                 "('data', 'fraction' or 'pixels'); the unused fields are 0.")
            .def("__init__",
                 [](PyAxLineData* p, double value, sextant::AxLineMode mode, double x, double y, double x2, double y2,
                    double slope, Coords xcoords, Coords ycoords, Coords x2coords, Coords y2coords) {
                     new (p) PyAxLineData{value, mode, x, y, x2, y2, slope, xcoords, ycoords, x2coords, y2coords};
                 },
                 "value"_a = 0.0, nb::kw_only(), "mode"_a = sextant::AxLineMode::Value, "x"_a = 0.0, "y"_a = 0.0,
                 "x2"_a = 0.0, "y2"_a = 0.0, "slope"_a = 0.0, "xcoords"_a = Coords::Data, "ycoords"_a = Coords::Data,
                 "x2coords"_a = Coords::Data, "y2coords"_a = Coords::Data)
            .def_rw("value", &PyAxLineData::value)
            .def_rw("mode", &PyAxLineData::mode)
            .def_rw("x", &PyAxLineData::x)
            .def_rw("y", &PyAxLineData::y)
            .def_rw("x2", &PyAxLineData::x2)
            .def_rw("y2", &PyAxLineData::y2)
            .def_rw("slope", &PyAxLineData::slope)
            .def_rw("xcoords", &PyAxLineData::xcoords)
            .def_rw("ycoords", &PyAxLineData::ycoords)
            .def_rw("x2coords", &PyAxLineData::x2coords)
            .def_rw("y2coords", &PyAxLineData::y2coords)
            .def("__repr__", [](const PyAxLineData& d) {
                return repr_fields("AxLineData", {{"value", nb::cast(d.value)}, {"mode", nb::cast(d.mode)},
                                                  {"x", nb::cast(d.x)}, {"y", nb::cast(d.y)},
                                                  {"x2", nb::cast(d.x2)}, {"y2", nb::cast(d.y2)},
                                                  {"slope", nb::cast(d.slope)}, {"xcoords", nb::cast(d.xcoords)},
                                                  {"ycoords", nb::cast(d.ycoords)}, {"x2coords", nb::cast(d.x2coords)},
                                                  {"y2coords", nb::cast(d.y2coords)}});
            });

        nb::class_<PyPolyData>(m, "PolyData", "shape_poly(): the vertices, in the options' `space`.")
            .def(nb::init<F64, F64>(), "x"_a, "y"_a)
            .def_rw("x", &PyPolyData::x)
            .def_rw("y", &PyPolyData::y)
            .def("__repr__", [](const PyPolyData& d) { return repr_fields("PolyData", {{"x", d.x}, {"y", d.y}}); });

        nb::class_<PyCircleData>(m, "CircleData",
                                 "shape_circle(): the centre and radius, each a value and what it is in\n"
                                 "('data', 'fraction' of the frame, or 'pixels'). A data radius needs a data\n"
                                 "centre.")
            .def(nb::init<double, double, double, Coords, Coords, Coords>(), "cx"_a, "cy"_a, "r"_a,
                 "cxcoords"_a = Coords::Data, "cycoords"_a = Coords::Data, "rcoords"_a = Coords::Data)
            .def_rw("cx", &PyCircleData::cx)
            .def_rw("cy", &PyCircleData::cy)
            .def_rw("r", &PyCircleData::r)
            .def_rw("cxcoords", &PyCircleData::cxcoords)
            .def_rw("cycoords", &PyCircleData::cycoords)
            .def_rw("rcoords", &PyCircleData::rcoords)
            .def("__repr__", [](const PyCircleData& d) {
                return repr_fields("CircleData", {{"cx", nb::cast(d.cx)}, {"cy", nb::cast(d.cy)}, {"r", nb::cast(d.r)},
                                                  {"cxcoords", nb::cast(d.cxcoords)}, {"cycoords", nb::cast(d.cycoords)},
                                                  {"rcoords", nb::cast(d.rcoords)}});
            });

        nb::class_<PyWedgeData>(m, "WedgeData",
                                "shape_wedge(): a CircleData's numbers, the sweep theta1 to theta2 in degrees\n"
                                "counter-clockwise from +x, and `inner`, the radius of the hole (0 = none) in\n"
                                "the space of `r`.")
            .def(nb::init<double, double, double, double, double, double, Coords, Coords, Coords>(), "cx"_a, "cy"_a,
                 "r"_a, "theta1"_a, "theta2"_a, "inner"_a = 0.0, "cxcoords"_a = Coords::Data,
                 "cycoords"_a = Coords::Data, "rcoords"_a = Coords::Data)
            .def_rw("cx", &PyWedgeData::cx)
            .def_rw("cy", &PyWedgeData::cy)
            .def_rw("r", &PyWedgeData::r)
            .def_rw("theta1", &PyWedgeData::theta1)
            .def_rw("theta2", &PyWedgeData::theta2)
            .def_rw("inner", &PyWedgeData::inner)
            .def_rw("cxcoords", &PyWedgeData::cxcoords)
            .def_rw("cycoords", &PyWedgeData::cycoords)
            .def_rw("rcoords", &PyWedgeData::rcoords)
            .def("__repr__", [](const PyWedgeData& d) {
                return repr_fields("WedgeData", {{"cx", nb::cast(d.cx)}, {"cy", nb::cast(d.cy)}, {"r", nb::cast(d.r)},
                                                 {"theta1", nb::cast(d.theta1)}, {"theta2", nb::cast(d.theta2)},
                                                 {"inner", nb::cast(d.inner)}, {"cxcoords", nb::cast(d.cxcoords)},
                                                 {"cycoords", nb::cast(d.cycoords)}, {"rcoords", nb::cast(d.rcoords)}});
            });

        nb::class_<PyPieData>(m, "PieData",
                              "Axes.pie(): the values as given (normalized at draw time), one label per\n"
                              "slice or none, and explode, None when no slice is offset, else one fraction\n"
                              "of the radius per slice.")
            .def(nb::init<F64, std::vector<std::string>, F64OrNone>(), "values"_a,
                 "labels"_a = std::vector<std::string>{}, "explode"_a.none() = nb::none())
            .def_rw("values", &PyPieData::values)
            .def_rw("labels", &PyPieData::labels)
            .def_rw("explode", &PyPieData::explode, nb::arg().none())
            .def("__repr__", [](const PyPieData& d) {
                return repr_fields("PieData", {{"values", d.values}, {"labels", nb::cast(d.labels)},
                                               {"explode", d.explode}});
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
                                  "to float32, which is how sextant stores them. An imshow_rgb() image has\n"
                                  "(rows, cols, channels) with each channel in [0, 1] (bytes come back / 255).\n"
                                  "x_edges (cols + 1) and y_edges (rows + 1) are the cell edges of a\n"
                                  "heatmap_rec(), None for the other kinds; xrange / yrange are then their ends.")
            .def(nb::init<F64, sextant::Range, sextant::Range, F64OrNone, F64OrNone>(), "data"_a, "xrange"_a,
                 "yrange"_a, "x_edges"_a.none() = nb::none(), "y_edges"_a.none() = nb::none())
            .def_rw("data", &PyHeatmapData::data)
            .def_rw("xrange", &PyHeatmapData::xrange)
            .def_rw("yrange", &PyHeatmapData::yrange)
            .def_rw("x_edges", &PyHeatmapData::x_edges, nb::arg().none())
            .def_rw("y_edges", &PyHeatmapData::y_edges, nb::arg().none())
            .def("__repr__", [](const PyHeatmapData& d) {
                return repr_fields("HeatmapData", {{"data", d.data},
                                                   {"xrange", nb::cast(d.xrange)},
                                                   {"yrange", nb::cast(d.yrange)},
                                                   {"x_edges", d.x_edges},
                                                   {"y_edges", d.y_edges}});
            });

        nb::class_<PyHeatmapTriData>(m, "HeatmapTriData",
                                     "heatmap_tri(): the vertices, one value each (NaN kept), and tri (n, 3)\n"
                                     "uint32 vertex indices: as passed, or the Delaunay triangulation made.")
            .def(nb::init<F64, F64, F64, U32>(), "x"_a, "y"_a, "values"_a, "tri"_a)
            .def_rw("x", &PyHeatmapTriData::x)
            .def_rw("y", &PyHeatmapTriData::y)
            .def_rw("values", &PyHeatmapTriData::values)
            .def_rw("tri", &PyHeatmapTriData::tri)
            .def("__repr__", [](const PyHeatmapTriData& d) {
                return repr_fields("HeatmapTriData", {{"x", d.x}, {"y", d.y}, {"values", d.values}, {"tri", d.tri}});
            });

        nb::class_<PyContourData>(m, "ContourData",
                                  "contour() / contourf(): x (one per column), y (one per row), z (len(y),\n"
                                  "len(x)), the levels drawn (from contour_data(): resolved; to\n"
                                  "set_contour_data(): empty re-resolves from the options) and filled.")
            .def("__init__",
                 [](PyContourData* p, F64 x, F64 y, F64 z, F64OrNone levels, bool filled) {
                     F64 lv = levels.is_none() ? to_numpy(std::vector<double>{}, {0}) : F64(levels);
                     new (p) PyContourData{std::move(x), std::move(y), std::move(z), std::move(lv), filled};
                 },
                 "x"_a, "y"_a, "z"_a, "levels"_a.none() = nb::none(), "filled"_a = false)
            .def_rw("x", &PyContourData::x)
            .def_rw("y", &PyContourData::y)
            .def_rw("z", &PyContourData::z)
            .def_rw("levels", &PyContourData::levels)
            .def_rw("filled", &PyContourData::filled)
            .def("__repr__", [](const PyContourData& d) {
                return repr_fields("ContourData", {{"x", d.x}, {"y", d.y}, {"z", d.z}, {"levels", d.levels},
                                                   {"filled", nb::cast(d.filled)}});
            });

        nb::class_<PyQuiver3DData>(m, "Quiver3DData",
                                   "quiver3d(): each arrow's anchor (x, y, z) and vector (u, v, w); c is None\n"
                                   "for a flat field, else one value per arrow, colormapped. As QuiverData.")
            .def(nb::init<F64, F64, F64, F64, F64, F64, F64OrNone>(), "x"_a, "y"_a, "z"_a, "u"_a, "v"_a, "w"_a,
                 "c"_a.none() = nb::none())
            .def_rw("x", &PyQuiver3DData::x)
            .def_rw("y", &PyQuiver3DData::y)
            .def_rw("z", &PyQuiver3DData::z)
            .def_rw("u", &PyQuiver3DData::u)
            .def_rw("v", &PyQuiver3DData::v)
            .def_rw("w", &PyQuiver3DData::w)
            .def_rw("c", &PyQuiver3DData::c, nb::arg().none())
            .def("__repr__", [](const PyQuiver3DData& d) {
                return repr_fields("Quiver3DData", {{"x", d.x}, {"y", d.y}, {"z", d.z}, {"u", d.u}, {"v", d.v},
                                                    {"w", d.w}, {"c", d.c}});
            });

        using sextant::ColorNorm;
        using sextant_py::PyColorNorm;
        nb::class_<PyColorNorm>(
            m, "ColorNorm",
            "How a value maps onto the colormap: linear() between vmin and vmax, log(), two_slope(center)\n"
            "(center is the colormap's middle, each side spread separately), or boundaries(bounds) (steps\n"
            "at the sorted bounds). Anywhere a norm is taken, 'linear' and 'log' are taken too.")
            .def_static("linear", [] { return PyColorNorm{ColorNorm::linear()}; })
            .def_static("log", [] { return PyColorNorm{ColorNorm::log()}; })
            .def_static("two_slope",
                        [](double center) { return PyColorNorm{ColorNorm::two_slope(static_cast<float>(center))}; },
                        "center"_a)
            .def_static("boundaries",
                        [](const Vec& bounds) {
                            return PyColorNorm{ColorNorm::boundaries({bounds.span().begin(), bounds.span().end()})};
                        },
                        "bounds"_a)
            .def_prop_ro("kind", [](const PyColorNorm& p) { return p.n.kind; })
            .def_prop_ro("center", [](const PyColorNorm& p) { return static_cast<double>(p.n.center); })
            .def_prop_ro("bounds", [](const PyColorNorm& p) { return p.n.bounds; })
            .def("__eq__",
                 [](const PyColorNorm& a, nb::object b) {
                     PyColorNorm* o = nullptr;
                     if (!nb::try_cast(b, o)) return false;
                     return a.n == o->n;
                 },
                 nb::is_operator(), "other"_a)
            .def("__repr__", [](const PyColorNorm& p) {
                const ColorNorm& n = p.n;
                using K = ColorNorm::Kind;
                switch (n.kind) {
                case K::Log: return std::string("ColorNorm.log()");
                case K::TwoSlope:
                    return "ColorNorm.two_slope(" + nb::cast<std::string>(nb::repr(nb::float_(n.center))) + ")";
                case K::Boundaries:
                    return "ColorNorm.boundaries(" + nb::cast<std::string>(nb::repr(nb::cast(n.bounds))) + ")";
                default: return std::string("ColorNorm.linear()");
                }
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
