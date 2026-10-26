// PolarAxes.
#include "options.h"
#include "wrappers.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <optional>
#include <string>

namespace sextant_py {
    namespace {
        using sextant::Coords;
        using sextant::PolarAxes;
        using Self = SelfOf<PolarAxes>;

        // fill()'s r2: a number (one value for every point) or one per point.
        using NumberOrArray = nb::typed<nb::object, Named<"float | ArrayLike">>;
        using ArrowArg = nb::typed<nb::object, Named<"ArrowOptions">>;

        bool is_number(nb::handle h) {
            if (PyFloat_Check(h.ptr()) || PyLong_Check(h.ptr())) return !PyBool_Check(h.ptr());
            // A numpy scalar or 0-d array.
            return nb::hasattr(h, "ndim") && nb::cast<int>(h.attr("ndim")) == 0;
        }

        sextant::Pos pos(double v, Coords c) {
            sextant::Pos p(v);
            p.space = c;
            return p;
        }

        template <class T>
        std::string opts_doc(const char* signature, const char* text) {
            return std::string(signature) + "\n\n" + text + "\n\nKeyword options (" + Fields<T>::name +
                   "): " + field_names<T>() + ".";
        }

        std::size_t count(PyPolar& w, std::size_t (PolarAxes::*f)() const) {
            return read(w, [&](PolarAxes& p) { return (p.*f)(); });
        }
    } // namespace

    void bind_polar(nb::module_& m) {
        using sextant::LineOptions;

        static const std::string plot_doc = opts_doc<LineOptions>(
            "plot(theta, r=None, **opts)",
            "A line through (theta[i], r[i]), theta in radians, straight in (theta, r): drawn as\n"
            "an arc where theta changes; NaN breaks it. plot(r) plots r against theta = 0, 1, 2,\n"
            "... radians. drawstyle steps in theta; loop closes back to the first point.");
        static const std::string scatter_doc = opts_doc<sextant::ScatterOptions>(
            "scatter(theta, r, **opts)",
            "Markers at (theta[i], r[i]); a point outside the r limits or the theta span is not drawn.");
        static const std::string scatter_z_doc = opts_doc<sextant::ScatterZOptions>(
            "scatter_z(theta, r, z, **opts)",
            "Markers colored by z through cmap/vmin/vmax (its colorbar beside the frame).");
        static const std::string bar_doc = opts_doc<sextant::BarOptions>(
            "bar(theta, height, **opts)",
            "Ring sectors: bar i spans theta[i] +- width/2 (width a fraction of the theta spacing,\n"
            "as Axes.bar()) and r from bottom[i] to bottom[i] + height[i], cut to the r limits and\n"
            "the theta span. ValueError for horizontal bars.");
        static const std::string fill_doc = opts_doc<sextant::FillOptions>(
            "fill(theta, r1, r2=0, **opts)",
            "The area between r1(theta) and r2(theta), r2 a number or one value per point; the\n"
            "default 0 fills inside a curve (a radar chart's). Its curves are straight in\n"
            "(theta, r), cut to the r limits and the theta span. With orientation='vertical',\n"
            "between theta1(r) and theta2(r): the arrays then read r, theta1, theta2.");
        static const std::string poly_doc = opts_doc<sextant::PolyOptions>(
            "shape_poly(theta, r, **opts)",
            "A filled polygon through (theta[i], r[i]), closed implicitly, its edges straight in\n"
            "(theta, r). Only space='data': another space raises ValueError.");
        static const std::string text_doc = opts_doc<sextant::TextOptions>(
            "text(s, theta, r, *, coords='data', **opts)",
            "Text at (theta, r): data values (radians, r), or with coords='fraction' / 'pixels' a\n"
            "point of the frame's box (pixels from its bottom left). Hidden while a data point is\n"
            "outside the r limits or the theta span.");
        static const std::string annotate_doc = opts_doc<sextant::TextOptions>(
            "annotate(theta, r, s, ttheta, tr, *, coords='data', arrow=None, **opts)",
            "Text at (ttheta, tr), placed as text() places it, with an arrow to the data point\n"
            "(theta, r). arrow is a dict of ArrowOptions fields. Hidden while the point is not\n"
            "shown.");
        static const std::string grid_doc = opts_doc<sextant::GridOptions>(
            "grid(enable=True, **opts)", "Rings at the r ticks, spokes at the theta grid lines; on by default.");
        static const std::string style_doc = opts_doc<sextant::AxesStyle>(
            "set_axes_style(**opts)",
            "background fills the frame, spine_color/spine_linewidth stroke its outline,\n"
            "tick_color and the label sizes style the labels; show_xticks hides the theta labels\n"
            "and show_yticks the r labels. Axis positions, origins, tick lengths and minor ticks\n"
            "do not apply. Unnamed fields reset to their defaults.");
        static const std::string legend_doc =
            opts_doc<sextant::LegendOptions>("legend(**opts)", "Show the legend (series with a name).");
        static const std::string colorbar_doc = opts_doc<sextant::ColorbarOptions>(
            "set_colorbar_style(**opts)",
            "Style shared by this axes' colorbars; a colorbar itself is asked for with\n"
            "colorbar=True on scatter_z.");

        nb::class_<PyPolar> cls(m, "PolarAxes", nb::is_weak_referenceable(),
                                "A polar subplot (Figure.add_subplot_polar()). theta is in radians in data\n"
                                "(plot(), line_data(), events), in degrees in every setter that places the frame\n"
                                "or its grid, as matplotlib's. r runs outward from rmin at the centre to rmax at\n"
                                "the frame.");

        // --- plot methods ------------------------------------------------------
        cls.def("plot",
                [](Self self, const Vec& theta, const std::optional<Vec>& r, const nb::kwargs& kw) {
                    auto o = options<LineOptions>("plot()", kw);
                    return chain(self, [&](PolarAxes& p) {
                        if (r) p.plot(theta.span(), r->span(), std::move(o));
                        else p.plot(theta.span(), std::move(o));
                    });
                },
                "theta"_a, "r"_a = nb::none(), "opts"_a, plot_doc.c_str())
            .def("scatter",
                 [](Self self, const Vec& theta, const Vec& r, const nb::kwargs& kw) {
                     auto o = options<sextant::ScatterOptions>("scatter()", kw);
                     return chain(self, [&](PolarAxes& p) { p.scatter(theta.span(), r.span(), std::move(o)); });
                 },
                 "theta"_a, "r"_a, "opts"_a, scatter_doc.c_str())
            .def("scatter_z",
                 [](Self self, const Vec& theta, const Vec& r, const Vec& z, const nb::kwargs& kw) {
                     auto o = options<sextant::ScatterZOptions>("scatter_z()", kw);
                     return chain(self,
                                  [&](PolarAxes& p) { p.scatter_z(theta.span(), r.span(), z.span(), std::move(o)); });
                 },
                 "theta"_a, "r"_a, "z"_a, "opts"_a, scatter_z_doc.c_str())
            .def("bar",
                 [](Self self, const Vec& theta, const Vec& height, const nb::kwargs& kw) {
                     auto o = options<sextant::BarOptions>("bar()", kw);
                     return chain(self, [&](PolarAxes& p) { p.bar(theta.span(), height.span(), std::move(o)); });
                 },
                 "theta"_a, "height"_a, "opts"_a, bar_doc.c_str())
            .def("fill",
                 [](Self self, const Vec& theta, const Vec& r1, NumberOrArray r2, const nb::kwargs& kw) {
                     auto o = options<sextant::FillOptions>("fill()", kw);
                     if (is_number(r2)) {
                         const double v = nb::cast<double>(r2);
                         return chain(self, [&](PolarAxes& p) { p.fill(theta.span(), r1.span(), v, std::move(o)); });
                     }
                     const auto a = as_array<1>(r2, "fill() r2");
                     return chain(self,
                                  [&](PolarAxes& p) { p.fill(theta.span(), r1.span(), a.span(), std::move(o)); });
                 },
                 "theta"_a, "r1"_a, "r2"_a = 0.0, "opts"_a, fill_doc.c_str())
            .def("shape_poly",
                 [](Self self, const Vec& theta, const Vec& r, const nb::kwargs& kw) {
                     auto o = options<sextant::PolyOptions>("shape_poly()", kw);
                     return chain(self, [&](PolarAxes& p) { p.shape_poly(theta.span(), r.span(), std::move(o)); });
                 },
                 "theta"_a, "r"_a, "opts"_a, poly_doc.c_str())
            .def("text",
                 [](Self self, std::string s, double theta, double r, Coords coords, const nb::kwargs& kw) {
                     auto o = options<sextant::TextOptions>("text()", kw);
                     return chain(self,
                                  [&](PolarAxes& p) { p.text(s, pos(theta, coords), pos(r, coords), std::move(o)); });
                 },
                 "s"_a, "theta"_a, "r"_a, nb::kw_only(), "coords"_a = Coords::Data, "opts"_a, text_doc.c_str())
            .def("annotate",
                 [](Self self, double theta, double r, std::string s, double ttheta, double tr, Coords coords,
                    ArrowArg arrow, const nb::kwargs& kw) {
                     auto o = options<sextant::TextOptions>("annotate()", kw);
                     auto a = options_arg<sextant::ArrowOptions>(arrow, "annotate()", "arrow");
                     return chain(self, [&](PolarAxes& p) {
                         p.annotate(theta, r, s, pos(ttheta, coords), pos(tr, coords), std::move(o), std::move(a));
                     });
                 },
                 "theta"_a, "r"_a, "s"_a, "ttheta"_a, "tr"_a, nb::kw_only(), "coords"_a = Coords::Data,
                 "arrow"_a.none() = nb::none(), "opts"_a, annotate_doc.c_str())

            // --- identity and removal -----------------------------------------
            .def("last_object", [](PyPolar& w) { return read(w, [](PolarAxes& p) { return p.last_object(); }); },
                 "The id (an int, 0 = none) of the object the latest plotting call made.")
            .def("remove", [](PyPolar& w, std::uint64_t id) { return read(w, [&](PolarAxes& p) { return p.remove(id); }); },
                 "id"_a,
                 "Take the object with this id out for good: True. An id that is not here changes\n"
                 "nothing, returns False and sends a message to the message handler. Later objects\n"
                 "of its kind move down one index; ids stay.")
            .def("find_object",
                 [](PyPolar& w, std::uint64_t id) {
                     return read(w, [&](PolarAxes& p) -> std::optional<std::pair<sextant::PickKind, std::size_t>> {
                         const auto loc = p.find_object(id);
                         if (!loc) return std::nullopt;
                         return std::pair{loc->kind, loc->index};
                     });
                 },
                 "id"_a, "(PickKind, index) of the object with this id; None if it is not here, was\n"
                         "removed, or is a text.")
            .def("set_zorder",
                 [](PyPolar& w, std::uint64_t id, double zorder) {
                     return read(w, [&](PolarAxes& p) { return p.set_zorder(id, zorder); });
                 },
                 "id"_a, "zorder"_a,
                 "Set the paint order of the object with this id, as Axes.set_zorder(): True; False\n"
                 "and a message for an id that is not here or a text.")
            .def("zorder", [](PyPolar& w, std::uint64_t id) { return read(w, [&](PolarAxes& p) { return p.zorder(id); }); },
                 "id"_a, "The zorder of the object with this id; None (and a message) for an id that is\n"
                         "not here or a text.")

            // --- read-back ---------------------------------------------------
            .def("line_count", [](PyPolar& w) { return count(w, &PolarAxes::line_count); })
            .def("scatter_count", [](PyPolar& w) { return count(w, &PolarAxes::scatter_count); })
            .def("scatter_z_count", [](PyPolar& w) { return count(w, &PolarAxes::scatter_z_count); })
            .def("bar_count", [](PyPolar& w) { return count(w, &PolarAxes::bar_count); })
            .def("fill_count", [](PyPolar& w) { return count(w, &PolarAxes::fill_count); })
            .def("shape_poly_count", [](PyPolar& w) { return count(w, &PolarAxes::shape_poly_count); })
            .def("text_count", [](PyPolar& w) { return count(w, &PolarAxes::text_count); })
            .def("line_data",
                 [](PyPolar& w, std::int64_t i) {
                     return to_python(read(w, [&](PolarAxes& p) { return p.line_data(index(i, p.line_count())); }));
                 },
                 "i"_a, "x is theta (radians), y is r.")
            .def("scatter_data",
                 [](PyPolar& w, std::int64_t i) {
                     return to_python(
                         read(w, [&](PolarAxes& p) { return p.scatter_data(index(i, p.scatter_count())); }));
                 },
                 "i"_a, "x is theta (radians), y is r.")
            .def("scatter_z_data",
                 [](PyPolar& w, std::int64_t i) {
                     return to_python(
                         read(w, [&](PolarAxes& p) { return p.scatter_z_data(index(i, p.scatter_z_count())); }));
                 },
                 "i"_a, "x is theta (radians), y is r.")
            .def("bar_data",
                 [](PyPolar& w, std::int64_t i) {
                     return to_python(read(w, [&](PolarAxes& p) { return p.bar_data(index(i, p.bar_count())); }));
                 },
                 "i"_a, "Centres are theta (radians), heights and bottoms r.")
            .def("fill_data",
                 [](PyPolar& w, std::int64_t i) {
                     return to_python(read(w, [&](PolarAxes& p) { return p.fill_data(index(i, p.fill_count())); }));
                 },
                 "i"_a, "x is theta (radians), y1/y2 are r; vertical: x is r, y1/y2 theta.")
            .def("shape_poly_data",
                 [](PyPolar& w, std::int64_t i) {
                     return to_python(
                         read(w, [&](PolarAxes& p) { return p.shape_poly_data(index(i, p.shape_poly_count())); }));
                 },
                 "i"_a, "x is theta (radians), y is r.")
            .def("text_data",
                 [](PyPolar& w, std::int64_t i) {
                     return to_python(read(w, [&](PolarAxes& p) { return p.text_data(index(i, p.text_count())); }));
                 },
                 "i"_a,
                 "text() and annotate() alike: x is theta, y is r (or both frame fractions or\n"
                 "pixels); px/py the arrow's point (theta, r).")

            // --- updating plotted data ----------------------------------------
            .def("set_line_data",
                 [](Self self, std::int64_t i, const PyLineData& d) {
                     const auto x = as_array<1>(d.x, "LineData.x"), y = as_array<1>(d.y, "LineData.y");
                     return chain(self, [&](PolarAxes& p) { p.set_line_data(index(i, p.line_count()), x.span(), y.span()); });
                 },
                 "i"_a, "data"_a)
            .def("set_line_data",
                 [](Self self, std::int64_t i, const Vec& theta, const Vec& r) {
                     return chain(self,
                                  [&](PolarAxes& p) { p.set_line_data(index(i, p.line_count()), theta.span(), r.span()); });
                 },
                 "i"_a, "theta"_a, "r"_a)
            .def("set_scatter_data",
                 [](Self self, std::int64_t i, const PyScatterData& d) {
                     const sextant::ScatterData cd = d.to_cpp();
                     return chain(self, [&](PolarAxes& p) { p.set_scatter_data(index(i, p.scatter_count()), cd); });
                 },
                 "i"_a, "data"_a)
            .def("set_scatter_data",
                 [](Self self, std::int64_t i, const Vec& theta, const Vec& r) {
                     return chain(self, [&](PolarAxes& p) {
                         p.set_scatter_data(index(i, p.scatter_count()), theta.span(), r.span());
                     });
                 },
                 "i"_a, "theta"_a, "r"_a)
            .def("set_scatter_z_data",
                 [](Self self, std::int64_t i, const PyScatterZData& d) {
                     const sextant::ScatterZData cd = d.to_cpp();
                     return chain(self, [&](PolarAxes& p) { p.set_scatter_z_data(index(i, p.scatter_z_count()), cd); });
                 },
                 "i"_a, "data"_a)
            .def("set_scatter_z_data",
                 [](Self self, std::int64_t i, const Vec& theta, const Vec& r, const Vec& z) {
                     return chain(self, [&](PolarAxes& p) {
                         p.set_scatter_z_data(index(i, p.scatter_z_count()), theta.span(), r.span(), z.span());
                     });
                 },
                 "i"_a, "theta"_a, "r"_a, "z"_a)
            .def("set_bar_data",
                 [](Self self, std::int64_t i, const PyBarData& d) {
                     const sextant::BarData cd = d.to_cpp();
                     return chain(self, [&](PolarAxes& p) { p.set_bar_data(index(i, p.bar_count()), cd); });
                 },
                 "i"_a, "data"_a)
            .def("set_bar_data",
                 [](Self self, std::int64_t i, const Vec& theta, const Vec& height) {
                     return chain(self, [&](PolarAxes& p) {
                         p.set_bar_data(index(i, p.bar_count()), theta.span(), height.span());
                     });
                 },
                 "i"_a, "theta"_a, "height"_a, "New centres re-derive the width, as Axes.set_bar_data().")
            .def("set_fill_data",
                 [](Self self, std::int64_t i, const PyFillData& d) {
                     const sextant::FillData cd = d.to_cpp();
                     return chain(self, [&](PolarAxes& p) { p.set_fill_data(index(i, p.fill_count()), cd); });
                 },
                 "i"_a, "data"_a)
            .def("set_fill_data",
                 [](Self self, std::int64_t i, const Vec& theta, const Vec& r1, const Vec& r2) {
                     return chain(self, [&](PolarAxes& p) {
                         p.set_fill_data(index(i, p.fill_count()), theta.span(), r1.span(), r2.span());
                     });
                 },
                 "i"_a, "theta"_a, "r1"_a, "r2"_a)
            .def("set_shape_poly_data",
                 [](Self self, std::int64_t i, const PyPolyData& d) {
                     const auto x = as_array<1>(d.x, "PolyData.x"), y = as_array<1>(d.y, "PolyData.y");
                     return chain(self, [&](PolarAxes& p) {
                         p.set_shape_poly_data(index(i, p.shape_poly_count()), x.span(), y.span());
                     });
                 },
                 "i"_a, "data"_a)
            .def("set_shape_poly_data",
                 [](Self self, std::int64_t i, const Vec& theta, const Vec& r) {
                     return chain(self, [&](PolarAxes& p) {
                         p.set_shape_poly_data(index(i, p.shape_poly_count()), theta.span(), r.span());
                     });
                 },
                 "i"_a, "theta"_a, "r"_a)
            .def("set_text_data",
                 [](Self self, std::int64_t i, const PyTextData& d) {
                     const sextant::TextData cd = d.to_cpp();
                     return chain(self, [&](PolarAxes& p) { p.set_text_data(index(i, p.text_count()), cd); });
                 },
                 "i"_a, "data"_a, "Replace text i's string, position and arrow (both coords alike).")
            .def("set_text_data",
                 [](Self self, std::int64_t i, std::string s, double theta, double r, std::optional<Coords> coords) {
                     return chain(self, [&](PolarAxes& p) {
                         const std::size_t k = index(i, p.text_count());
                         const Coords c = coords ? *coords : p.text_data(k).x.space;
                         p.set_text_data(k, s, pos(theta, c), pos(r, c));
                     });
                 },
                 "i"_a, "s"_a, "theta"_a, "r"_a, nb::kw_only(), "coords"_a = nb::none(),
                 "The string and position only; an annotation keeps its arrow and point. The\n"
                 "coords stay unless given.")

            // --- decoration --------------------------------------------------
            .def("set_title",
                 [](Self self, std::string text, float fontsize) {
                     return chain(self, [&](PolarAxes& p) { p.set_title(text, fontsize); });
                 },
                 "text"_a, "fontsize"_a = 18.0f)
            .def("title", [](PyPolar& w) { return read(w, [](PolarAxes& p) { return p.title(); }); })
            .def("legend",
                 [](Self self, const nb::kwargs& kw) {
                     auto o = options<sextant::LegendOptions>("legend()", kw);
                     return chain(self, [&](PolarAxes& p) { p.legend(std::move(o)); });
                 },
                 "opts"_a, legend_doc.c_str())
            .def("set_color_cycle",
                 [](Self self, std::vector<sextant::Color> colors) {
                     return chain(self, [&](PolarAxes& p) { p.set_color_cycle(std::move(colors)); });
                 },
                 "colors"_a, "As Axes.set_color_cycle(); [] restores the default.")
            .def("grid",
                 [](Self self, bool enable, const nb::kwargs& kw) {
                     auto o = options<sextant::GridOptions>("grid()", kw);
                     return chain(self, [&](PolarAxes& p) { p.grid(enable, o); });
                 },
                 "enable"_a = true, "opts"_a, grid_doc.c_str())
            .def("set_axes_style",
                 [](Self self, const nb::kwargs& kw) {
                     auto o = options<sextant::AxesStyle>("set_axes_style()", kw);
                     return chain(self, [&](PolarAxes& p) { p.set_axes_style(std::move(o)); });
                 },
                 "opts"_a, style_doc.c_str())
            .def("set_colorbar_style",
                 [](Self self, const nb::kwargs& kw) {
                     auto o = options<sextant::ColorbarOptions>("set_colorbar_style()", kw);
                     return chain(self, [&](PolarAxes& p) { p.set_colorbar_style(std::move(o)); });
                 },
                 "opts"_a, colorbar_doc.c_str())
            .def("cla", [](Self self) { return chain(self, [](PolarAxes& p) { p.cla(); }); },
                 "Everything back to a new polar axes' defaults, the plot objects too.")

            // --- the r axis --------------------------------------------------
            .def("set_rlim",
                 [](Self self, double lo, double hi) { return chain(self, [&](PolarAxes& p) { p.set_rlim(lo, hi); }); },
                 "lo"_a, "hi"_a, "ValueError unless both are finite and lo != hi (lo > hi turns r inward).")
            .def("set_rmin", [](Self self, double lo) { return chain(self, [&](PolarAxes& p) { p.set_rmin(lo); }); },
                 "lo"_a, "One end, the other kept at its current value (rlim()).")
            .def("set_rmax", [](Self self, double hi) { return chain(self, [&](PolarAxes& p) { p.set_rmax(hi); }); },
                 "hi"_a)
            .def("rlim", [](PyPolar& w) { return read(w, [](PolarAxes& p) { return p.rlim(); }); },
                 "The r limits as drawn: set ones, or automatic (rmin 0 when no shown r is negative,\n"
                 "rmax the largest rounded up to a tick; (0, 1) with no data).")
            .def("set_rticks",
                 [](Self self, const Vec& positions, std::vector<std::string> labels) {
                     return chain(self, [&](PolarAxes& p) { p.set_rticks(positions.span(), std::move(labels)); });
                 },
                 "positions"_a, "labels"_a = std::vector<std::string>{},
                 "Rings at `positions` labelled `labels` ([] = formatted); [] positions goes back\n"
                 "to automatic ones.")
            .def("set_rtick_format",
                 [](Self self, std::string fmt) { return chain(self, [&](PolarAxes& p) { p.set_rtick_format(fmt); }); },
                 "fmt"_a, "As Axes.set_ytick_format().")
            .def("set_rlabel_position",
                 [](Self self, double degrees) {
                     return chain(self, [&](PolarAxes& p) { p.set_rlabel_position(degrees); });
                 },
                 "degrees"_a, "The angle (degrees, in theta) the r labels sit along.")
            .def("rlabel_position", [](PyPolar& w) { return read(w, [](PolarAxes& p) { return p.rlabel_position(); }); })

            // --- the theta axis (degrees) -------------------------------------
            .def("set_thetagrids",
                 [](Self self, const Vec& degrees, std::vector<std::string> labels) {
                     return chain(self, [&](PolarAxes& p) { p.set_thetagrids(degrees.span(), std::move(labels)); });
                 },
                 "degrees"_a, "labels"_a = std::vector<std::string>{},
                 "Spokes at `degrees` labelled `labels` ([] = '45°' ...); [] degrees goes back to\n"
                 "every 45°.")
            .def("set_theta_zero_location",
                 [](Self self, sextant::ThetaZero loc, double offset) {
                     return chain(self, [&](PolarAxes& p) { p.set_theta_zero_location(loc, offset); });
                 },
                 "loc"_a, "offset"_a = 0.0,
                 "theta = 0 at `loc` ('e', 'n', 'w', 's', 'ne', 'nw', 'sw', 'se'), turned a further\n"
                 "`offset` degrees counterclockwise.")
            .def("set_theta_offset",
                 [](Self self, double degrees) { return chain(self, [&](PolarAxes& p) { p.set_theta_offset(degrees); }); },
                 "degrees"_a, "The screen angle of theta = 0, degrees counterclockwise from east.")
            .def("theta_offset", [](PyPolar& w) { return read(w, [](PolarAxes& p) { return p.theta_offset(); }); })
            .def("set_theta_direction",
                 [](Self self, int direction) {
                     return chain(self, [&](PolarAxes& p) { p.set_theta_direction(direction); });
                 },
                 "direction"_a, "1 counterclockwise (the default), -1 clockwise; anything else raises ValueError.")
            .def("theta_direction", [](PyPolar& w) { return read(w, [](PolarAxes& p) { return p.theta_direction(); }); })
            .def("set_thetamin",
                 [](Self self, double degrees) { return chain(self, [&](PolarAxes& p) { p.set_thetamin(degrees); }); },
                 "degrees"_a)
            .def("set_thetamax",
                 [](Self self, double degrees) { return chain(self, [&](PolarAxes& p) { p.set_thetamax(degrees); }); },
                 "degrees"_a)
            .def("set_thetalim",
                 [](Self self, double lo, double hi) {
                     return chain(self, [&](PolarAxes& p) { p.set_thetalim(lo, hi); });
                 },
                 "min_degrees"_a, "max_degrees"_a,
                 "The theta span drawn: a sector between the two (degrees, finite, min < max, at\n"
                 "most 360 apart; else ValueError).")
            .def("thetalim", [](PyPolar& w) { return read(w, [](PolarAxes& p) { return p.thetalim(); }); },
                 "The theta span in degrees.");

        // --- hiding plotted objects -------------------------------------------
        bind_visible(cls, "set_line_visible", "line_visible", &PolarAxes::line_count, &PolarAxes::set_line_visible,
                     &PolarAxes::line_visible);
        bind_visible(cls, "set_scatter_visible", "scatter_visible", &PolarAxes::scatter_count,
                     &PolarAxes::set_scatter_visible, &PolarAxes::scatter_visible);
        bind_visible(cls, "set_scatter_z_visible", "scatter_z_visible", &PolarAxes::scatter_z_count,
                     &PolarAxes::set_scatter_z_visible, &PolarAxes::scatter_z_visible);
        bind_visible(cls, "set_bar_visible", "bar_visible", &PolarAxes::bar_count, &PolarAxes::set_bar_visible,
                     &PolarAxes::bar_visible);
        bind_visible(cls, "set_fill_visible", "fill_visible", &PolarAxes::fill_count, &PolarAxes::set_fill_visible,
                     &PolarAxes::fill_visible);
        bind_visible(cls, "set_shape_poly_visible", "shape_poly_visible", &PolarAxes::shape_poly_count,
                     &PolarAxes::set_shape_poly_visible, &PolarAxes::shape_poly_visible);
        bind_visible(cls, "set_text_visible", "text_visible", &PolarAxes::text_count, &PolarAxes::set_text_visible,
                     &PolarAxes::text_visible);
    }
} // namespace sextant_py
