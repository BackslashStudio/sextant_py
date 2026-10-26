// Axes and Plane2D: the 2D plot kinds, bound once for both, plus what each has
// of its own.
#include "options.h"
#include "wrappers.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <string>

namespace sextant_py {
    namespace {
        using sextant::BarOptions;
        using sextant::HeatmapOptions;
        using sextant::LineOptions;
        using sextant::Range;
        using sextant::ScatterOptions;
        using sextant::ScatterZOptions;

        // fill()'s y2: a number (one value for every point) or one per point.
        using NumberOrArray = nb::typed<nb::object, Named<"float | ArrayLike">>;

        bool is_number(nb::handle h) {
            if (PyFloat_Check(h.ptr()) || PyLong_Check(h.ptr())) return !PyBool_Check(h.ptr());
            // A numpy scalar or 0-d array.
            return nb::hasattr(h, "ndim") && nb::cast<int>(h.attr("ndim")) == 0;
        }

        // stack()'s layers: a 2-D array-like, one layer per row (a 1-D one is one
        // layer), as the C++ vectors.
        std::vector<std::vector<double>> layers(nb::handle ys) {
            nb::object a;
            try {
                nb::object np = nb::module_::import_("numpy");
                a = np.attr("asarray")(ys, nb::arg("dtype") = np.attr("float64"));
            } catch (const nb::python_error&) {
                throw nb::type_error(("stack(): ys must be a 2-D array of numbers (one row per layer) or a "
                                      "sequence of equally long 1-D arrays, got " +
                                      nb::cast<std::string>(nb::repr(ys))).c_str());
            }
            const int nd = nb::cast<int>(a.attr("ndim"));
            if (nd == 1) a = a.attr("reshape")(1, -1);
            else if (nd != 2)
                throw nb::type_error("stack(): ys must be 2-D (one row per layer)");
            const Mat m = as_array<2>(a, "stack() ys");
            std::vector<std::vector<double>> out(m.shape(0));
            const auto s = m.span();
            const std::size_t n = m.shape(1);
            for (std::size_t k = 0; k < out.size(); ++k) out[k].assign(s.begin() + k * n, s.begin() + (k + 1) * n);
            return out;
        }

        // z is (len(y), len(x)): the C++ check only sees the product.
        void check_grid(const Vec& x, const Vec& y, const Mat& z, const char* what) {
            if (z.shape(0) != y.shape(0) || z.shape(1) != x.shape(0))
                throw nb::value_error((std::string(what) + ": z must be (len(y), len(x)) = (" +
                                       std::to_string(y.shape(0)) + ", " + std::to_string(x.shape(0)) + "), got (" +
                                       std::to_string(z.shape(0)) + ", " + std::to_string(z.shape(1)) + ")").c_str());
        }

        sextant::ErrorBar spans(const PyErrorBar* err) { return err ? err->spans() : sextant::ErrorBar{}; }

        using sextant::Coords;
        using ArrowArg = nb::typed<nb::object, Named<"ArrowOptions">>;

        sextant::Pos pos(double v, Coords c) {
            sextant::Pos p(v);
            p.space = c;
            return p;
        }

        // sharex()/sharey() take only an axes of the same figure: another
        // figure's graph is under another lock, so it is refused before the call.
        void same_figure(const PyAxes& self, const PyAxes& other, const char* what) {
            if (self.st != other.st)
                throw nb::value_error((std::string(what) + ": the axes belong to different figures").c_str());
        }

        template <class T>
        std::string opts_doc(const char* signature, const char* text) {
            return std::string(signature) + "\n\n" + text + "\n\nKeyword options (" + Fields<T>::name +
                   "): " + field_names<T>() + ".";
        }

        // The plot kinds, read-back and updates Axes and Plane2D share. `T` is
        // sextant::Axes or sextant::Plane2D, whose methods have the same names.
        template <class T>
        void bind_plot2d(nb::class_<Wrapped<T>>& cls) {
            using W = Wrapped<T>;
            using Self = SelfOf<T>;

            static const std::string line_doc = opts_doc<LineOptions>(
                "line(x, y=None, *, err=None, **opts)", "A line through (x, y); line(y) plots y against 0, 1, 2, ...");
            static const std::string scatter_doc =
                opts_doc<ScatterOptions>("scatter(x, y, *, err=None, **opts)", "Markers at (x, y).");
            static const std::string scatter_z_doc = opts_doc<ScatterZOptions>(
                "scatter_z(x, y, z, *, err=None, **opts)", "Markers at (x, y), coloured by z through cmap/vmin/vmax.");
            static const std::string bar_doc =
                opts_doc<BarOptions>("bar(x, height, *, err=None, **opts)", "Bars centred on x.");
            static const std::string heatmap_doc = opts_doc<HeatmapOptions>(
                "heatmap(data, xrange, yrange, **opts)",
                "A 2-D array (rows, cols) drawn over xrange x yrange, (lo, hi) tuples of outer cell edges.");
            static const std::string imshow_doc = opts_doc<HeatmapOptions>(
                "imshow(data, **opts)", "heatmap() over x in [0, cols], y in [0, rows].");
            static const std::string fill_doc = opts_doc<sextant::FillOptions>(
                "fill(x, y1, y2=0, **opts)",
                "The area between y1(x) and y2(x) (matplotlib's fill_between); y2 is a number or one\n"
                "value per point. With orientation='vertical', between x1(y) and x2(y), the arrays\n"
                "then reading y, x1, x2 (fill_betweenx). exclude= leaves points out (the inverse of\n"
                "matplotlib's where); a NaN splits the fill. ValueError for lengths that differ.");
            static const std::string stack_doc = opts_doc<sextant::StackOptions>(
                "stack(x, ys, **opts)",
                "Stacked areas (matplotlib's stackplot): ys is a 2-D array, one row per layer (or a\n"
                "sequence of 1-D arrays), each one value per x. Each layer is added as an ordinary\n"
                "fill(), read back and edited as one. colors/names: empty, or one per layer.");
            static const std::string imshow_rgb_doc = opts_doc<sextant::ImageOptions>(
                "imshow_rgb(data, **opts)",
                "A true-color image: data is (rows, cols, 3 or 4) of RGB(A). uint8 arrays span 0..255, anything\n"
                "else 0..1; a pixel with a NaN channel is transparent. Over x in [0, cols], y in [0, rows]\n"
                "unless xrange / yrange are given. A heatmap for read-back and picking.");
            static const std::string heatmap_rec_doc = opts_doc<sextant::HeatmapRecOptions>(
                "heatmap_rec(data, x_edges, y_edges, **opts)",
                "A heatmap over uneven cells: data is (rows, cols), x_edges has cols + 1 increasing values,\n"
                "y_edges rows + 1 (matplotlib's pcolormesh with edges). A heatmap for read-back and picking.");
            static const std::string heatmap_tri_doc = opts_doc<sextant::HeatmapTriOptions>(
                "heatmap_tri(x, y, values, tri=None, **opts)",
                "A heatmap over triangles (matplotlib's tripcolor, gouraud): one value per vertex. tri is\n"
                "(n, 3) or flat vertex indices; left out, the Delaunay triangulation of (x, y) is used.");
            static const std::string contour_doc = opts_doc<sextant::ContourOptions>(
                "contour(x, y, z, **opts)",
                "Contour lines over a grid: x one per column, y one per row, z (len(y), len(x)).");
            static const std::string contourf_doc = opts_doc<sextant::ContourOptions>(
                "contourf(x, y, z, **opts)", "Filled contours: contour() with filled set.");
            static const std::string quiver_doc = opts_doc<sextant::QuiverOptions>(
                "quiver(x, y, u, v, c=None, **opts)",
                "An arrow per point from (x, y) along amp * (u, v) in data units (matplotlib's quiver\n"
                "with angles='xy', scale_units='xy'; amp is its 1/scale); unset amp is chosen from the\n"
                "data and kept. With c (one value per arrow), each arrow is colored through\n"
                "cmap/vmin/vmax. width, head_width and head_length are pixels. ValueError for\n"
                "lengths that differ or an amp that is not finite and > 0.");
            static const std::string axspan_doc = opts_doc<sextant::AxSpanOptions>(
                "axspan(lo, hi, **opts)",
                "A band across the frame over y in [lo, hi] (matplotlib's axhspan), or over x with\n"
                "orientation='vertical' (axvspan); lo > hi is stored swapped. Limits take both ends\n"
                "as axline() takes its value. Takes no color from the color cycle (unset: blue).\n"
                "On a Plane2D the band spans the plane's v (its u when vertical).");
            static const std::string poly_doc = opts_doc<sextant::PolyOptions>(
                "shape_poly(x, y, **opts)",
                "A filled polygon through the vertices (x[i], y[i]), closed implicitly, in `space`\n"
                "('data' by default, 'fraction' of the frame, or 'pixels' from its bottom left\n"
                "corner). A simple polygon only. ValueError for fewer than 3 vertices, lengths that\n"
                "differ or a non-finite number; nothing is added then.");
            static const std::string circle_doc = opts_doc<sextant::CircleOptions>(
                "shape_circle(cx, cy, r, *, coords='data', xcoords=None, ycoords=None, rcoords=None, **opts)",
                "A filled circle. Each of cx, cy and r is in xcoords/ycoords/rcoords (default:\n"
                "coords): 'data', 'fraction' of the frame, or 'pixels'. A data radius is measured\n"
                "along each axis, so it is an ellipse on unequal axes and bends on a log axis; a\n"
                "'fraction' radius is a fraction of the frame's shorter side, and a pixel one a\n"
                "length; both are round. ValueError for a non-finite number, r <= 0, or a data\n"
                "radius with a centre that is not in data units.");
            static const std::string wedge_doc = opts_doc<sextant::WedgeOptions>(
                "shape_wedge(cx, cy, r, theta1, theta2, *, coords='data', xcoords=None, ycoords=None, "
                "rcoords=None, **opts)",
                "A sector of the circle, from theta1 to theta2 degrees counter-clockwise from +x\n"
                "(theta2 < theta1 wraps; 360 or more apart is the whole circle), with a hole of\n"
                "radius `inner` in r's space (a ring segment). Checks as shape_circle(), plus finite\n"
                "angles and 0 <= inner < r.");

            cls.def("line",
                    [](Self self, const Vec& x, const std::optional<Vec>& y, const PyErrorBar* err,
                       const nb::kwargs& kw) {
                        auto o = options<LineOptions>("line()", kw);
                        const auto e = spans(err);
                        return chain(self, [&](T& t) {
                            if (y && err) t.line(x.span(), y->span(), e, std::move(o));
                            else if (y) t.line(x.span(), y->span(), std::move(o));
                            else if (err) t.line(x.span(), e, std::move(o));
                            else t.line(x.span(), std::move(o));
                        });
                    },
                    "x"_a, "y"_a = nb::none(), nb::kw_only(), "err"_a.none() = nb::none(), "opts"_a,
                    line_doc.c_str())
                .def("scatter",
                     [](Self self, const Vec& x, const Vec& y, const PyErrorBar* err, const nb::kwargs& kw) {
                         auto o = options<ScatterOptions>("scatter()", kw);
                         const auto e = spans(err);
                         return chain(self, [&](T& t) {
                             if (err) t.scatter(x.span(), y.span(), e, std::move(o));
                             else t.scatter(x.span(), y.span(), std::move(o));
                         });
                     },
                     "x"_a, "y"_a, nb::kw_only(), "err"_a.none() = nb::none(), "opts"_a, scatter_doc.c_str())
                .def("scatter_z",
                     [](Self self, const Vec& x, const Vec& y, const Vec& z, const PyErrorBar* err,
                        const nb::kwargs& kw) {
                         auto o = options<ScatterZOptions>("scatter_z()", kw);
                         const auto e = spans(err);
                         return chain(self, [&](T& t) {
                             if (err) t.scatter_z(x.span(), y.span(), z.span(), e, std::move(o));
                             else t.scatter_z(x.span(), y.span(), z.span(), std::move(o));
                         });
                     },
                     "x"_a, "y"_a, "z"_a, nb::kw_only(), "err"_a.none() = nb::none(), "opts"_a,
                     scatter_z_doc.c_str())
                .def("bar",
                     [](Self self, const Vec& x, const Vec& height, const PyErrorBar* err, const nb::kwargs& kw) {
                         auto o = options<BarOptions>("bar()", kw);
                         const auto e = spans(err);
                         return chain(self, [&](T& t) {
                             if (err) t.bar(x.span(), height.span(), e, std::move(o));
                             else t.bar(x.span(), height.span(), std::move(o));
                         });
                     },
                     "x"_a, "height"_a, nb::kw_only(), "err"_a.none() = nb::none(), "opts"_a, bar_doc.c_str())
                .def("heatmap",
                     [](Self self, const Mat& data, Range xrange, Range yrange, const nb::kwargs& kw) {
                         auto o = options<HeatmapOptions>("heatmap()", kw);
                         return chain(self, [&](T& t) {
                             t.heatmap(data.span(), static_cast<int>(data.shape(0)), static_cast<int>(data.shape(1)),
                                       xrange, yrange, std::move(o));
                         });
                     },
                     "data"_a, "xrange"_a, "yrange"_a, "opts"_a, heatmap_doc.c_str())
                .def("imshow",
                     [](Self self, const Mat& data, const nb::kwargs& kw) {
                         auto o = options<HeatmapOptions>("imshow()", kw);
                         return chain(self, [&](T& t) {
                             t.imshow(data.span(), static_cast<int>(data.shape(0)), static_cast<int>(data.shape(1)),
                                      std::move(o));
                         });
                     },
                     "data"_a, "opts"_a, imshow_doc.c_str())
                .def("imshow_rgb",
                     [](Self self, nb::object data, const nb::kwargs& kw) {
                         auto o = options<sextant::ImageOptions>("imshow_rgb()", kw);
                         nb::module_ np = nb::module_::import_("numpy");
                         nb::object a = np.attr("asarray")(data);
                         if (nb::cast<int>(a.attr("ndim")) != 3)
                             throw nb::value_error("imshow_rgb(): data must be (rows, cols, channels)");
                         const auto shape = nb::cast<nb::tuple>(a.attr("shape"));
                         const int rows = nb::cast<int>(shape[0]), cols = nb::cast<int>(shape[1]),
                                   ch = nb::cast<int>(shape[2]);
                         if (nb::cast<std::string>(a.attr("dtype").attr("name")) == "uint8") {
                             a = np.attr("ascontiguousarray")(a, "dtype"_a = np.attr("uint8"));
                             const auto nd = nb::cast<
                                 nb::ndarray<const std::uint8_t, nb::c_contig, nb::device::cpu>>(a);
                             return chain(self, [&](T& t) {
                                 t.imshow_rgb(std::span<const std::uint8_t>(nd.data(), nd.size()), rows, cols, ch,
                                              std::move(o));
                             });
                         }
                         const auto g = nb::cast<Grid>(a);
                         return chain(self, [&](T& t) { t.imshow_rgb(g.span(), rows, cols, ch, std::move(o)); });
                     },
                     "data"_a, "opts"_a, imshow_rgb_doc.c_str())
                .def("heatmap_rec",
                     [](Self self, const Mat& data, const Vec& x_edges, const Vec& y_edges, const nb::kwargs& kw) {
                         auto o = options<sextant::HeatmapRecOptions>("heatmap_rec()", kw);
                         return chain(self, [&](T& t) {
                             t.heatmap_rec(data.span(), static_cast<int>(data.shape(0)),
                                           static_cast<int>(data.shape(1)), x_edges.span(), y_edges.span(),
                                           std::move(o));
                         });
                     },
                     "data"_a, "x_edges"_a, "y_edges"_a, "opts"_a, heatmap_rec_doc.c_str())
                .def("heatmap_tri",
                     [](Self self, const Vec& x, const Vec& y, const Vec& values, nb::object tri,
                        const nb::kwargs& kw) {
                         auto o = options<sextant::HeatmapTriOptions>("heatmap_tri()", kw);
                         std::optional<Triangles> tr;
                         if (!tri.is_none()) tr = triangles(tri, "heatmap_tri()");
                         return chain(self, [&](T& t) {
                             if (tr) t.heatmap_tri(x.span(), y.span(), values.span(), tr->s, std::move(o));
                             else t.heatmap_tri(x.span(), y.span(), values.span(), std::move(o));
                         });
                     },
                     "x"_a, "y"_a, "values"_a, "tri"_a.none() = nb::none(), "opts"_a, heatmap_tri_doc.c_str())
                .def("contour",
                     [](Self self, const Vec& x, const Vec& y, const Mat& z, const nb::kwargs& kw) {
                         auto o = options<sextant::ContourOptions>("contour()", kw);
                         check_grid(x, y, z, "contour()");
                         return chain(self, [&](T& t) { t.contour(x.span(), y.span(), z.span(), std::move(o)); });
                     },
                     "x"_a, "y"_a, "z"_a, "opts"_a, contour_doc.c_str())
                .def("contourf",
                     [](Self self, const Vec& x, const Vec& y, const Mat& z, const nb::kwargs& kw) {
                         auto o = options<sextant::ContourOptions>("contourf()", kw);
                         check_grid(x, y, z, "contourf()");
                         return chain(self, [&](T& t) { t.contourf(x.span(), y.span(), z.span(), std::move(o)); });
                     },
                     "x"_a, "y"_a, "z"_a, "opts"_a, contourf_doc.c_str())
                .def("fill",
                     [](Self self, const Vec& x, const Vec& y1, NumberOrArray y2, const nb::kwargs& kw) {
                         auto o = options<sextant::FillOptions>("fill()", kw);
                         if (is_number(y2)) {
                             const double v = nb::cast<double>(y2);
                             return chain(self, [&](T& t) { t.fill(x.span(), y1.span(), v, std::move(o)); });
                         }
                         const auto a = as_array<1>(y2, "fill() y2");
                         return chain(self, [&](T& t) { t.fill(x.span(), y1.span(), a.span(), std::move(o)); });
                     },
                     "x"_a, "y1"_a, "y2"_a = 0.0, "opts"_a, fill_doc.c_str())
                .def("stack",
                     [](Self self, const Vec& x, nb::typed<nb::object, Named<"ArrayLike">> ys, const nb::kwargs& kw) {
                         auto o = options<sextant::StackOptions>("stack()", kw);
                         const auto l = layers(ys);
                         return chain(self, [&](T& t) { t.stack(x.span(), l, std::move(o)); });
                     },
                     "x"_a, "ys"_a, "opts"_a, stack_doc.c_str())
                .def("quiver",
                     [](Self self, const Vec& x, const Vec& y, const Grid& u, const Grid& v, const std::optional<Grid>& c,
                        const nb::kwargs& kw) {
                         auto o = options<sextant::QuiverOptions>("quiver()", kw);
                         return chain(self, [&](T& t) {
                             if (c) t.quiver(x.span(), y.span(), u.span(), v.span(), c->span(), std::move(o));
                             else t.quiver(x.span(), y.span(), u.span(), v.span(), std::move(o));
                         });
                     },
                     "x"_a, "y"_a, "u"_a, "v"_a, "c"_a.none() = nb::none(), "opts"_a, quiver_doc.c_str())
                .def("axspan",
                     [](Self self, double lo, double hi, const nb::kwargs& kw) {
                         auto o = options<sextant::AxSpanOptions>("axspan()", kw);
                         return chain(self, [&](T& t) { t.axspan(lo, hi, std::move(o)); });
                     },
                     "lo"_a, "hi"_a, "opts"_a, axspan_doc.c_str())
                .def("shape_poly",
                     [](Self self, const Vec& x, const Vec& y, const nb::kwargs& kw) {
                         auto o = options<sextant::PolyOptions>("shape_poly()", kw);
                         return chain(self, [&](T& t) { t.shape_poly(x.span(), y.span(), std::move(o)); });
                     },
                     "x"_a, "y"_a, "opts"_a, poly_doc.c_str())
                .def("shape_circle",
                     [](Self self, double cx, double cy, double r, Coords coords, std::optional<Coords> xcoords,
                        std::optional<Coords> ycoords, std::optional<Coords> rcoords, const nb::kwargs& kw) {
                         auto o = options<sextant::CircleOptions>("shape_circle()", kw);
                         return chain(self, [&](T& t) {
                             t.shape_circle(pos(cx, xcoords.value_or(coords)), pos(cy, ycoords.value_or(coords)),
                                            pos(r, rcoords.value_or(coords)), std::move(o));
                         });
                     },
                     "cx"_a, "cy"_a, "r"_a, nb::kw_only(), "coords"_a = Coords::Data, "xcoords"_a = nb::none(),
                     "ycoords"_a = nb::none(), "rcoords"_a = nb::none(), "opts"_a, circle_doc.c_str())
                .def("shape_wedge",
                     [](Self self, double cx, double cy, double r, double theta1, double theta2, Coords coords,
                        std::optional<Coords> xcoords, std::optional<Coords> ycoords, std::optional<Coords> rcoords,
                        const nb::kwargs& kw) {
                         auto o = options<sextant::WedgeOptions>("shape_wedge()", kw);
                         return chain(self, [&](T& t) {
                             t.shape_wedge(pos(cx, xcoords.value_or(coords)), pos(cy, ycoords.value_or(coords)),
                                           pos(r, rcoords.value_or(coords)), theta1, theta2, std::move(o));
                         });
                     },
                     "cx"_a, "cy"_a, "r"_a, "theta1"_a, "theta2"_a, nb::kw_only(), "coords"_a = Coords::Data,
                     "xcoords"_a = nb::none(), "ycoords"_a = nb::none(), "rcoords"_a = nb::none(), "opts"_a,
                     wedge_doc.c_str())
                .def("cla", [](Self self) { return chain(self, [](T& t) { t.cla(); }); },
                     "Clear all plot objects (on Axes, also reset limits, the color cycle and the\n"
                     "aspect; a shared axis stays shared).")

                // --- identity and removal -----------------------------------
                .def("last_object", [](W& w) { return read(w, [](T& t) { return t.last_object(); }); },
                     "The id (an int, 0 = none) of the object the latest plotting call made; the last\n"
                     "one when a call makes several (stack()). Events name picked objects by it.")
                .def("remove",
                     [](W& w, std::uint64_t id) { return read(w, [&](T& t) { return t.remove(id); }); },
                     "id"_a,
                     "Take the object with this id out for good: True. An id that is not here (removed\n"
                     "already, unknown, another axes') changes nothing, returns False and sends a\n"
                     "message to the message handler. Later objects of its kind move down one index\n"
                     "(line_data(i) and the rest follow); ids stay.")
                .def("find_object",
                     [](W& w, std::uint64_t id) {
                         return read(w, [&](T& t) -> std::optional<std::pair<sextant::PickKind, std::size_t>> {
                             const auto loc = t.find_object(id);
                             if (!loc) return std::nullopt;
                             return std::pair{loc->kind, loc->index};
                         });
                     },
                     "id"_a,
                     "(PickKind, index) of the object with this id, the i of line_data(i) and the\n"
                     "other read-back calls; None if it is not here, was removed, or is a text,\n"
                     "axline or axspan.")
                .def("set_zorder",
                     [](W& w, std::uint64_t id, double zorder) {
                         return read(w, [&](T& t) { return t.set_zorder(id, zorder); });
                     },
                     "id"_a, "zorder"_a,
                     "Set the paint order of the object with this id: objects draw in ascending zorder,\n"
                     "equal ones in the order of the calls (the legend and colorbars stay in call\n"
                     "order). True; False and a message for an id that is not here or a text. A zorder\n"
                     "that is not finite raises ValueError.")
                .def("zorder",
                     [](W& w, std::uint64_t id) {
                         return read(w, [&](T& t) { return t.zorder(id); });
                     },
                     "id"_a,
                     "The zorder of the object with this id; None (and a message) for an id that is\n"
                     "not here or a text.")

                // --- read-back ---------------------------------------------
                .def("line_count", [](W& w) { return read(w, [](T& t) { return t.line_count(); }); })
                .def("scatter_count", [](W& w) { return read(w, [](T& t) { return t.scatter_count(); }); })
                .def("scatter_z_count", [](W& w) { return read(w, [](T& t) { return t.scatter_z_count(); }); })
                .def("bar_count", [](W& w) { return read(w, [](T& t) { return t.bar_count(); }); })
                .def("heatmap_count", [](W& w) { return read(w, [](T& t) { return t.heatmap_count(); }); })
                .def("fill_count", [](W& w) { return read(w, [](T& t) { return t.fill_count(); }); },
                     "fill() and each stack() layer alike.")
                .def("quiver_count", [](W& w) { return read(w, [](T& t) { return t.quiver_count(); }); })
                .def("line_data",
                     [](W& w, std::int64_t i) {
                         return to_python(read(w, [&](T& t) { return t.line_data(index(i, t.line_count())); }));
                     },
                     "i"_a)
                .def("scatter_data",
                     [](W& w, std::int64_t i) {
                         return to_python(read(w, [&](T& t) { return t.scatter_data(index(i, t.scatter_count())); }));
                     },
                     "i"_a)
                .def("scatter_z_data",
                     [](W& w, std::int64_t i) {
                         return to_python(
                             read(w, [&](T& t) { return t.scatter_z_data(index(i, t.scatter_z_count())); }));
                     },
                     "i"_a)
                .def("bar_data",
                     [](W& w, std::int64_t i) {
                         return to_python(read(w, [&](T& t) { return t.bar_data(index(i, t.bar_count())); }));
                     },
                     "i"_a, "bar() and hist() alike.")
                .def("heatmap_data",
                     [](W& w, std::int64_t i) {
                         return to_python(read(w, [&](T& t) { return t.heatmap_data(index(i, t.heatmap_count())); }));
                     },
                     "i"_a, "heatmap() and imshow() alike.")
                .def("heatmap_tri_count", [](W& w) { return read(w, [](T& t) { return t.heatmap_tri_count(); }); })
                .def("heatmap_tri_data",
                     [](W& w, std::int64_t i) {
                         return to_python(
                             read(w, [&](T& t) { return t.heatmap_tri_data(index(i, t.heatmap_tri_count())); }));
                     },
                     "i"_a)
                .def("contour_count", [](W& w) { return read(w, [](T& t) { return t.contour_count(); }); })
                .def("contour_data",
                     [](W& w, std::int64_t i) {
                         return to_python(read(w, [&](T& t) { return t.contour_data(index(i, t.contour_count())); }));
                     },
                     "i"_a, "contour() and contourf() alike.")
                .def("fill_data",
                     [](W& w, std::int64_t i) {
                         return to_python(read(w, [&](T& t) { return t.fill_data(index(i, t.fill_count())); }));
                     },
                     "i"_a)
                .def("quiver_data",
                     [](W& w, std::int64_t i) {
                         return to_python(read(w, [&](T& t) { return t.quiver_data(index(i, t.quiver_count())); }));
                     },
                     "i"_a)
                .def("axspan_count", [](W& w) { return read(w, [](T& t) { return t.axspan_count(); }); })
                .def("axspan_data",
                     [](W& w, std::int64_t i) {
                         return read(w, [&](T& t) { return t.axspan_data(index(i, t.axspan_count())); });
                     },
                     "i"_a)
                .def("shape_poly_count", [](W& w) { return read(w, [](T& t) { return t.shape_poly_count(); }); })
                .def("shape_circle_count", [](W& w) { return read(w, [](T& t) { return t.shape_circle_count(); }); })
                .def("shape_wedge_count", [](W& w) { return read(w, [](T& t) { return t.shape_wedge_count(); }); })
                .def("shape_poly_data",
                     [](W& w, std::int64_t i) {
                         return to_python(
                             read(w, [&](T& t) { return t.shape_poly_data(index(i, t.shape_poly_count())); }));
                     },
                     "i"_a)
                .def("shape_circle_data",
                     [](W& w, std::int64_t i) {
                         return to_python(
                             read(w, [&](T& t) { return t.shape_circle_data(index(i, t.shape_circle_count())); }));
                     },
                     "i"_a)
                .def("shape_wedge_data",
                     [](W& w, std::int64_t i) {
                         return to_python(
                             read(w, [&](T& t) { return t.shape_wedge_data(index(i, t.shape_wedge_count())); }));
                     },
                     "i"_a)

                // --- updating plotted data ------------------------------------
                .def("set_line_data",
                     [](Self self, std::int64_t i, const PyLineData& d) {
                         const auto x = as_array<1>(d.x, "LineData.x"), y = as_array<1>(d.y, "LineData.y");
                         return chain(self, [&](T& t) { t.set_line_data(index(i, t.line_count()), x.span(), y.span()); });
                     },
                     "i"_a, "data"_a)
                .def("set_line_data",
                     [](Self self, std::int64_t i, const Vec& x, const Vec& y) {
                         return chain(self, [&](T& t) { t.set_line_data(index(i, t.line_count()), x.span(), y.span()); });
                     },
                     "i"_a, "x"_a, "y"_a)
                .def("set_scatter_data",
                     [](Self self, std::int64_t i, const PyScatterData& d) {
                         const sextant::ScatterData cd = d.to_cpp();
                         return chain(self, [&](T& t) { t.set_scatter_data(index(i, t.scatter_count()), cd); });
                     },
                     "i"_a, "data"_a)
                .def("set_scatter_data",
                     [](Self self, std::int64_t i, const Vec& x, const Vec& y) {
                         return chain(self, [&](T& t) {
                             t.set_scatter_data(index(i, t.scatter_count()), x.span(), y.span());
                         });
                     },
                     "i"_a, "x"_a, "y"_a)
                .def("set_scatter_z_data",
                     [](Self self, std::int64_t i, const PyScatterZData& d) {
                         const sextant::ScatterZData cd = d.to_cpp();
                         return chain(self, [&](T& t) { t.set_scatter_z_data(index(i, t.scatter_z_count()), cd); });
                     },
                     "i"_a, "data"_a)
                .def("set_scatter_z_data",
                     [](Self self, std::int64_t i, const Vec& x, const Vec& y, const Vec& z) {
                         return chain(self, [&](T& t) {
                             t.set_scatter_z_data(index(i, t.scatter_z_count()), x.span(), y.span(), z.span());
                         });
                     },
                     "i"_a, "x"_a, "y"_a, "z"_a)
                .def("set_bar_data",
                     [](Self self, std::int64_t i, const PyBarData& d) {
                         const sextant::BarData cd = d.to_cpp();
                         return chain(self, [&](T& t) { t.set_bar_data(index(i, t.bar_count()), cd); });
                     },
                     "i"_a, "data"_a)
                .def("set_bar_data",
                     [](Self self, std::int64_t i, const Vec& x, const Vec& height) {
                         return chain(self, [&](T& t) {
                             t.set_bar_data(index(i, t.bar_count()), x.span(), height.span());
                         });
                     },
                     "i"_a, "x"_a, "height"_a)
                .def("set_heatmap_data",
                     [](Self self, std::int64_t i, const PyHeatmapData& d) {
                         const sextant::HeatmapData hd = d.to_cpp();
                         return chain(self, [&](T& t) { t.set_heatmap_data(index(i, t.heatmap_count()), hd); });
                     },
                     "i"_a, "data"_a)
                .def("set_heatmap_data",
                     [](Self self, std::int64_t i, const Mat& data, Range xrange, Range yrange) {
                         return chain(self, [&](T& t) {
                             t.set_heatmap_data(index(i, t.heatmap_count()), data.span(),
                                                static_cast<int>(data.shape(0)), static_cast<int>(data.shape(1)),
                                                xrange, yrange);
                         });
                     },
                     "i"_a, "data"_a, "xrange"_a, "yrange"_a)
                .def("set_heatmap_tri_data",
                     [](Self self, std::int64_t i, const PyHeatmapTriData& d) {
                         const auto x = as_array<1>(d.x, "HeatmapTriData.x"), y = as_array<1>(d.y, "HeatmapTriData.y"),
                                    v = as_array<1>(d.values, "HeatmapTriData.values");
                         const auto tr = triangles(d.tri, "HeatmapTriData.tri");
                         return chain(self, [&](T& t) {
                             t.set_heatmap_tri_data(index(i, t.heatmap_tri_count()), x.span(), y.span(), v.span(),
                                                    tr.s);
                         });
                     },
                     "i"_a, "data"_a)
                .def("set_heatmap_tri_data",
                     [](Self self, std::int64_t i, const Vec& x, const Vec& y, const Vec& values) {
                         return chain(self, [&](T& t) {
                             t.set_heatmap_tri_data(index(i, t.heatmap_tri_count()), x.span(), y.span(),
                                                    values.span());
                         });
                     },
                     "i"_a, "x"_a, "y"_a, "values"_a,
                     "Replace the vertices and values, keeping the triangles (the count must hold).")
                .def("set_contour_data",
                     [](Self self, std::int64_t i, const PyContourData& d) {
                         const sextant::ContourData cd = d.to_cpp();
                         return chain(self, [&](T& t) { t.set_contour_data(index(i, t.contour_count()), cd); });
                     },
                     "i"_a, "data"_a)
                .def("set_contour_data",
                     [](Self self, std::int64_t i, const Vec& x, const Vec& y, const Mat& z) {
                         check_grid(x, y, z, "set_contour_data()");
                         return chain(self, [&](T& t) {
                             t.set_contour_data(index(i, t.contour_count()), x.span(), y.span(), z.span());
                         });
                     },
                     "i"_a, "x"_a, "y"_a, "z"_a, "Replace the grid, keeping the levels and filled.")
                .def("set_fill_data",
                     [](Self self, std::int64_t i, const PyFillData& d) {
                         const sextant::FillData cd = d.to_cpp();
                         return chain(self, [&](T& t) { t.set_fill_data(index(i, t.fill_count()), cd); });
                     },
                     "i"_a, "data"_a)
                .def("set_fill_data",
                     [](Self self, std::int64_t i, const Vec& x, const Vec& y1, const Vec& y2) {
                         return chain(self, [&](T& t) {
                             t.set_fill_data(index(i, t.fill_count()), x.span(), y1.span(), y2.span());
                         });
                     },
                     "i"_a, "x"_a, "y1"_a, "y2"_a)
                .def("set_quiver_data",
                     [](Self self, std::int64_t i, const PyQuiverData& d) {
                         const sextant::QuiverData cd = d.to_cpp();
                         return chain(self, [&](T& t) { t.set_quiver_data(index(i, t.quiver_count()), cd); });
                     },
                     "i"_a, "data"_a)
                .def("set_quiver_data",
                     [](Self self, std::int64_t i, const Vec& x, const Vec& y, const Vec& u, const Vec& v) {
                         return chain(self, [&](T& t) {
                             t.set_quiver_data(index(i, t.quiver_count()), x.span(), y.span(), u.span(), v.span());
                         });
                     },
                     "i"_a, "x"_a, "y"_a, "u"_a, "v"_a)
                .def("set_axspan_data",
                     [](Self self, std::int64_t i, const PyAxSpanData& d) {
                         return chain(self, [&](T& t) { t.set_axspan_data(index(i, t.axspan_count()), d); });
                     },
                     "i"_a, "data"_a)
                .def("set_axspan_data",
                     [](Self self, std::int64_t i, double lo, double hi) {
                         return chain(self, [&](T& t) { t.set_axspan_data(index(i, t.axspan_count()), {lo, hi}); });
                     },
                     "i"_a, "lo"_a, "hi"_a, "Move span i (validated as axspan(); lo > hi stored swapped).")
                .def("set_shape_poly_data",
                     [](Self self, std::int64_t i, const PyPolyData& d) {
                         const auto x = as_array<1>(d.x, "PolyData.x"), y = as_array<1>(d.y, "PolyData.y");
                         return chain(self, [&](T& t) {
                             t.set_shape_poly_data(index(i, t.shape_poly_count()), x.span(), y.span());
                         });
                     },
                     "i"_a, "data"_a)
                .def("set_shape_poly_data",
                     [](Self self, std::int64_t i, const Vec& x, const Vec& y) {
                         return chain(self, [&](T& t) {
                             t.set_shape_poly_data(index(i, t.shape_poly_count()), x.span(), y.span());
                         });
                     },
                     "i"_a, "x"_a, "y"_a)
                .def("set_shape_circle_data",
                     [](Self self, std::int64_t i, const PyCircleData& d) {
                         const sextant::CircleData cd = d.to_cpp();
                         return chain(self, [&](T& t) { t.set_shape_circle_data(index(i, t.shape_circle_count()), cd); });
                     },
                     "i"_a, "data"_a)
                .def("set_shape_circle_data",
                     [](Self self, std::int64_t i, double cx, double cy, double r, std::optional<Coords> coords,
                        std::optional<Coords> xcoords, std::optional<Coords> ycoords, std::optional<Coords> rcoords) {
                         return chain(self, [&](T& t) {
                             const std::size_t k = index(i, t.shape_circle_count());
                             const sextant::CircleData cur = t.shape_circle_data(k);
                             t.set_shape_circle_data(
                                 k, {pos(cx, xcoords ? *xcoords : coords ? *coords : cur.cx.space),
                                     pos(cy, ycoords ? *ycoords : coords ? *coords : cur.cy.space),
                                     pos(r, rcoords ? *rcoords : coords ? *coords : cur.r.space)});
                         });
                     },
                     "i"_a, "cx"_a, "cy"_a, "r"_a, nb::kw_only(), "coords"_a = nb::none(), "xcoords"_a = nb::none(),
                     "ycoords"_a = nb::none(), "rcoords"_a = nb::none(),
                     "Replace circle i's centre and radius (validated as shape_circle()). Each keeps its\n"
                     "current coords unless coords/xcoords/ycoords/rcoords says otherwise.")
                .def("set_shape_wedge_data",
                     [](Self self, std::int64_t i, const PyWedgeData& d) {
                         const sextant::WedgeData cd = d.to_cpp();
                         return chain(self, [&](T& t) { t.set_shape_wedge_data(index(i, t.shape_wedge_count()), cd); });
                     },
                     "i"_a, "data"_a)
                .def("set_shape_wedge_data",
                     [](Self self, std::int64_t i, double cx, double cy, double r, double theta1, double theta2,
                        std::optional<double> inner, std::optional<Coords> coords, std::optional<Coords> xcoords,
                        std::optional<Coords> ycoords, std::optional<Coords> rcoords) {
                         return chain(self, [&](T& t) {
                             const std::size_t k = index(i, t.shape_wedge_count());
                             const sextant::WedgeData cur = t.shape_wedge_data(k);
                             t.set_shape_wedge_data(
                                 k, {pos(cx, xcoords ? *xcoords : coords ? *coords : cur.cx.space),
                                     pos(cy, ycoords ? *ycoords : coords ? *coords : cur.cy.space),
                                     pos(r, rcoords ? *rcoords : coords ? *coords : cur.r.space), theta1, theta2,
                                     inner.value_or(cur.inner)});
                         });
                     },
                     "i"_a, "cx"_a, "cy"_a, "r"_a, "theta1"_a, "theta2"_a, nb::kw_only(), "inner"_a = nb::none(),
                     "coords"_a = nb::none(), "xcoords"_a = nb::none(), "ycoords"_a = nb::none(),
                     "rcoords"_a = nb::none(),
                     "Replace wedge i's numbers (validated as shape_wedge()). inner keeps its value when\n"
                     "left out; each coords is kept as for set_shape_circle_data().");

            // --- hiding plotted objects ---------------------------------------
            bind_visible(cls, "set_line_visible", "line_visible", &T::line_count, &T::set_line_visible,
                         &T::line_visible);
            bind_visible(cls, "set_scatter_visible", "scatter_visible", &T::scatter_count,
                         &T::set_scatter_visible, &T::scatter_visible);
            bind_visible(cls, "set_scatter_z_visible", "scatter_z_visible", &T::scatter_z_count,
                         &T::set_scatter_z_visible, &T::scatter_z_visible);
            bind_visible(cls, "set_bar_visible", "bar_visible", &T::bar_count, &T::set_bar_visible,
                         &T::bar_visible);
            bind_visible(cls, "set_heatmap_visible", "heatmap_visible", &T::heatmap_count,
                         &T::set_heatmap_visible, &T::heatmap_visible);
            bind_visible(cls, "set_heatmap_tri_visible", "heatmap_tri_visible", &T::heatmap_tri_count,
                         &T::set_heatmap_tri_visible, &T::heatmap_tri_visible);
            bind_visible(cls, "set_contour_visible", "contour_visible", &T::contour_count,
                         &T::set_contour_visible, &T::contour_visible);
            bind_visible(cls, "set_fill_visible", "fill_visible", &T::fill_count, &T::set_fill_visible,
                         &T::fill_visible);
            bind_visible(cls, "set_quiver_visible", "quiver_visible", &T::quiver_count,
                         &T::set_quiver_visible, &T::quiver_visible);
            bind_visible(cls, "set_axspan_visible", "axspan_visible", &T::axspan_count, &T::set_axspan_visible,
                         &T::axspan_visible);
            bind_visible(cls, "set_shape_poly_visible", "shape_poly_visible", &T::shape_poly_count,
                         &T::set_shape_poly_visible, &T::shape_poly_visible);
            bind_visible(cls, "set_shape_circle_visible", "shape_circle_visible", &T::shape_circle_count,
                         &T::set_shape_circle_visible, &T::shape_circle_visible);
            bind_visible(cls, "set_shape_wedge_visible", "shape_wedge_visible", &T::shape_wedge_count,
                         &T::set_shape_wedge_visible, &T::shape_wedge_visible);
        }
    } // namespace

    void bind_axes(nb::module_& m) {
        using sextant::Axes;
        using Self = SelfOf<Axes>;

        static const std::string hist_doc = opts_doc<BarOptions>(
            "hist(data, bins=10, *, density=False, cumulative=False, **opts)",
            "Histogram as bars. width defaults to 1.0 here (bins touch), not bar()'s 0.8.");
        static const std::string hist2d_doc = opts_doc<sextant::Hist2dOptions>(
            "hist2d(x, y, **opts)",
            "A 2D histogram of the points (x[i], y[i]) drawn as a heatmap (a heatmap_rec when x_edges or\n"
            "y_edges are given); it counts in heatmap_count().");
        static const std::string axline_doc = opts_doc<sextant::AxLineOptions>(
            "axline(value, **opts)",
            "A line across the frame at y = value (matplotlib's axhline), or at x = value with\n"
            "orientation='vertical' (axvline). It runs from from_ to to, fractions of the frame,\n"
            "whatever the limits. Automatic limits take value as a point on its axis. Not\n"
            "hovered or picked. ValueError for a non-finite value, from_ or to.\n"
            "Also axline(x, y, *, slope=...) and axline(x1, y1, x2, y2): see the next overloads.");
        static const std::string axline_slope_doc = opts_doc<sextant::AxLineOptions>(
            "axline(x, y, *, slope, coords='data', xcoords=None, ycoords=None, **opts)",
            "The infinite line through (x, y) with slope dy/dx (matplotlib's axline). With the\n"
            "point in data units it is the straight line of the data: it bends on a log or symlog\n"
            "axis. With a 'fraction' or 'pixels' coordinate it is a straight screen line, its slope\n"
            "the screen's (y up). orientation, from_ and to do not apply. Automatic limits take the\n"
            "point. ValueError for a non-finite number.");
        static const std::string axline_two_doc = opts_doc<sextant::AxLineOptions>(
            "axline(x1, y1, x2, y2, *, coords='data', xcoords=None, ycoords=None, **opts)",
            "The infinite line through two points, as above; equal x gives a vertical line.\n"
            "ValueError for a non-finite number or two equal points.");
        static const std::string grid_doc = opts_doc<sextant::GridOptions>("grid(enable=True, **opts)", "Grid lines.");
        static const std::string style_doc = opts_doc<sextant::AxesStyle>(
            "set_axes_style(**opts)", "Spines, ticks, labels and titles. Unnamed fields reset to their defaults.");
        static const std::string legend_doc =
            opts_doc<sextant::LegendOptions>("legend(**opts)", "Show the legend (series with a name).");
        static const std::string text_doc = opts_doc<sextant::TextOptions>(
            "text(s, x, y, *, coords='data', xcoords=None, ycoords=None, **opts)",
            "Text at (x, y), drawn over the data at a fixed pixel size. Each coordinate is in\n"
            "xcoords/ycoords (default: coords): 'data', 'fraction' of the plot frame or 'pixels', from its\n"
            "bottom-left corner. Never widens the auto limits; hidden while a data coordinate is\n"
            "out of view (unless clip_to_frame=True, which cuts it at the frame instead).");
        static const std::string annotate_doc = opts_doc<sextant::TextOptions>(
            "annotate(px, py, s, tx, ty, *, coords='data', xcoords=None, ycoords=None, arrow=None, **opts)",
            "Text at (tx, ty), placed as text() places it, with an arrow to the data point\n"
            "(px, py). arrow is a dict of ArrowOptions fields: head, tail, head_length,\n"
            "head_width, linewidth, color, linestyle, gap_text, gap_point, arc. Hidden while\n"
            "the point is out of view.");
        static const std::string colorbar_doc = opts_doc<sextant::ColorbarOptions>(
            "set_colorbar_style(**opts)", "Style shared by this axes' colorbars; a colorbar itself is asked for "
                                          "with colorbar=True on heatmap/scatter_z.");
        static const std::string pie_doc = opts_doc<sextant::PieOptions>(
            "pie(values, **opts)",
            "A pie chart (matplotlib's pie()): slice k is values[k]'s share of the circle (of the\n"
            "sum, or of 1 with normalize=False), from startangle degrees, centred at (cx, cy) in\n"
            "data units. labels and explode: one per slice or empty; explode is a fraction of the\n"
            "radius. autopct is a printf conversion of the percent ('%1.1f%%'). One legend entry\n"
            "per labelled slice; a press on slice k picks the pie with pick_index k. Turns the\n"
            "axis off and sets aspect 1, as matplotlib's does. ValueError for a negative or\n"
            "non-finite value, a sum of 0 (over 1 without normalize), labels or explode of another\n"
            "length, a negative explode, radius <= 0, inner outside [0, radius), or a bad autopct.");

        static const std::string xscale_doc = opts_doc<sextant::ScaleOptions>(
            "set_xscale(scale, **opts)",
            "The x axis' scale: 'linear', 'log' or 'symlog' (linear within +-linthresh, so it\n"
            "takes 0 and negatives). Limits stay in data units. ValueError for options the\n"
            "scale cannot use, and for log/symlog on an axes holding a heatmap.");
        static const std::string yscale_doc =
            opts_doc<sextant::ScaleOptions>("set_yscale(scale, **opts)", "set_xscale() for the y axis.");

        nb::class_<PyAxes> cls(m, "Axes", nb::is_weak_referenceable());
        bind_plot2d<Axes>(cls);
        bind_visible(cls, "set_axline_visible", "axline_visible", &Axes::axline_count, &Axes::set_axline_visible,
                     &Axes::axline_visible);
        bind_visible(cls, "set_text_visible", "text_visible", &Axes::text_count, &Axes::set_text_visible,
                     &Axes::text_visible);
        bind_visible(cls, "set_pie_visible", "pie_visible", &Axes::pie_count, &Axes::set_pie_visible,
                     &Axes::pie_visible);
        cls.def("pie",
                [](Self self, const Vec& values, const nb::kwargs& kw) {
                    auto o = options<sextant::PieOptions>("pie()", kw);
                    return chain(self, [&](Axes& ax) { ax.pie(values.span(), std::move(o)); });
                },
                "values"_a, "opts"_a, pie_doc.c_str())
            .def("pie_count", [](PyAxes& a) { return read(a, [](Axes& ax) { return ax.pie_count(); }); })
            .def("pie_data",
                 [](PyAxes& a, std::int64_t i) {
                     return to_python(read(a, [&](Axes& ax) { return ax.pie_data(index(i, ax.pie_count())); }));
                 },
                 "i"_a)
            .def("set_pie_data",
                 [](Self self, std::int64_t i, const PyPieData& d) {
                     const sextant::PieData cd = d.to_cpp();
                     return chain(self, [&](Axes& ax) { ax.set_pie_data(index(i, ax.pie_count()), cd); });
                 },
                 "i"_a, "data"_a,
                 "Replace pie i's values, labels and explode (checked as pie()); the slices' angles\n"
                 "follow, the options stay.")
            .def("set_pie_data",
                 [](Self self, std::int64_t i, const Vec& values, std::optional<std::vector<std::string>> labels,
                    const std::optional<Vec>& explode) {
                     return chain(self, [&](Axes& ax) {
                         const std::size_t k = index(i, ax.pie_count());
                         sextant::PieData cur = ax.pie_data(k);
                         cur.values.assign(values.span().begin(), values.span().end());
                         if (labels) cur.labels = std::move(*labels);
                         if (explode) cur.explode.assign(explode->span().begin(), explode->span().end());
                         ax.set_pie_data(k, cur);
                     });
                 },
                 "i"_a, "values"_a, "labels"_a = nb::none(), "explode"_a = nb::none(),
                 "New values; labels and explode keep theirs unless given (the count must then hold).");
        cls.def("hist2d",
                [](Self self, const Vec& x, const Vec& y, const nb::kwargs& kw) {
                    auto o = options<sextant::Hist2dOptions>("hist2d()", kw);
                    return chain(self, [&](Axes& ax) { ax.hist2d(x.span(), y.span(), std::move(o)); });
                },
                "x"_a, "y"_a, "opts"_a, hist2d_doc.c_str());
        cls.def("hist",
                [](Self self, const Vec& data, int bins, bool density, bool cumulative, const nb::kwargs& kw) {
                    BarOptions init;
                    init.width = 1.0f; // the C++ default argument's; see hist_doc
                    auto o = options<BarOptions>("hist()", kw, init);
                    return chain(self, [&](Axes& ax) {
                        ax.hist(data.span(), bins, std::move(o), {.density = density, .cumulative = cumulative});
                    });
                },
                "data"_a, "bins"_a = 10, nb::kw_only(), "density"_a = false, "cumulative"_a = false, "opts"_a,
                hist_doc.c_str())

            // --- reference lines and spans -----------------------------------
            .def("axline",
                 [](Self self, double value, const nb::kwargs& kw) {
                     auto o = options<sextant::AxLineOptions>("axline()", kw);
                     return chain(self, [&](Axes& ax) { ax.axline(value, std::move(o)); });
                 },
                 "value"_a, "opts"_a, axline_doc.c_str())
            .def("axline",
                 [](Self self, double x, double y, double slope, Coords coords, std::optional<Coords> xcoords,
                    std::optional<Coords> ycoords, const nb::kwargs& kw) {
                     auto o = options<sextant::AxLineOptions>("axline()", kw);
                     return chain(self, [&](Axes& ax) {
                         ax.axline(pos(x, xcoords.value_or(coords)), pos(y, ycoords.value_or(coords)), slope,
                                   std::move(o));
                     });
                 },
                 "x"_a, "y"_a, nb::kw_only(), "slope"_a, "coords"_a = Coords::Data, "xcoords"_a = nb::none(),
                 "ycoords"_a = nb::none(), "opts"_a, axline_slope_doc.c_str())
            .def("axline",
                 [](Self self, double x1, double y1, double x2, double y2, Coords coords,
                    std::optional<Coords> xcoords, std::optional<Coords> ycoords, const nb::kwargs& kw) {
                     auto o = options<sextant::AxLineOptions>("axline()", kw);
                     return chain(self, [&](Axes& ax) {
                         ax.axline(pos(x1, xcoords.value_or(coords)), pos(y1, ycoords.value_or(coords)),
                                   pos(x2, xcoords.value_or(coords)), pos(y2, ycoords.value_or(coords)), std::move(o));
                     });
                 },
                 "x1"_a, "y1"_a, "x2"_a, "y2"_a, nb::kw_only(), "coords"_a = Coords::Data, "xcoords"_a = nb::none(),
                 "ycoords"_a = nb::none(), "opts"_a, axline_two_doc.c_str())
            .def("axline_count", [](PyAxes& a) { return read(a, [](Axes& ax) { return ax.axline_count(); }); })
            .def("axline_data",
                 [](PyAxes& a, std::int64_t i) {
                     return to_python(read(a, [&](Axes& ax) { return ax.axline_data(index(i, ax.axline_count())); }));
                 },
                 "i"_a)
            .def("set_axline_data",
                 [](Self self, std::int64_t i, const PyAxLineData& d) {
                     const sextant::AxLineData cd = d.to_cpp();
                     return chain(self, [&](Axes& ax) { ax.set_axline_data(index(i, ax.axline_count()), cd); });
                 },
                 "i"_a, "data"_a)
            .def("set_axline_data",
                 [](Self self, std::int64_t i, double value) {
                     return chain(self, [&](Axes& ax) { ax.set_axline_data(index(i, ax.axline_count()), {value}); });
                 },
                 "i"_a, "value"_a, "Move reference line i to this value (mode 'value'; validated as axline()).")

            // --- text ----------------------------------------------------------
            .def("text",
                 [](Self self, std::string s, double x, double y, Coords coords, std::optional<Coords> xcoords,
                    std::optional<Coords> ycoords, const nb::kwargs& kw) {
                     auto o = options<sextant::TextOptions>("text()", kw);
                     return chain(self, [&](Axes& ax) {
                         ax.text(s, pos(x, xcoords.value_or(coords)), pos(y, ycoords.value_or(coords)), std::move(o));
                     });
                 },
                 "s"_a, "x"_a, "y"_a, nb::kw_only(), "coords"_a = Coords::Data, "xcoords"_a = nb::none(),
                 "ycoords"_a = nb::none(), "opts"_a, text_doc.c_str())
            .def("annotate",
                 [](Self self, double px, double py, std::string s, double tx, double ty, Coords coords,
                    std::optional<Coords> xcoords, std::optional<Coords> ycoords, ArrowArg arrow,
                    const nb::kwargs& kw) {
                     auto o = options<sextant::TextOptions>("annotate()", kw);
                     auto a = options_arg<sextant::ArrowOptions>(arrow, "annotate()", "arrow");
                     return chain(self, [&](Axes& ax) {
                         ax.annotate(px, py, s, pos(tx, xcoords.value_or(coords)), pos(ty, ycoords.value_or(coords)),
                                     std::move(o), std::move(a));
                     });
                 },
                 "px"_a, "py"_a, "s"_a, "tx"_a, "ty"_a, nb::kw_only(), "coords"_a = Coords::Data,
                 "xcoords"_a = nb::none(), "ycoords"_a = nb::none(), "arrow"_a.none() = nb::none(), "opts"_a,
                 annotate_doc.c_str())
            .def("text_count", [](PyAxes& a) { return read(a, [](Axes& ax) { return ax.text_count(); }); })
            .def("text_data",
                 [](PyAxes& a, std::int64_t i) {
                     return to_python(read(a, [&](Axes& ax) { return ax.text_data(index(i, ax.text_count())); }));
                 },
                 "i"_a, "text() and annotate() alike.")
            .def("set_text_data",
                 [](Self self, std::int64_t i, const PyTextData& d) {
                     const sextant::TextData cd = d.to_cpp();
                     return chain(self, [&](Axes& ax) { ax.set_text_data(index(i, ax.text_count()), cd); });
                 },
                 "i"_a, "data"_a,
                 "Replace text i's string, position and arrow (arrow=True on a text() gives it\n"
                 "default arrow options).")
            .def("set_text_data",
                 [](Self self, std::int64_t i, std::string s, double x, double y, std::optional<Coords> coords,
                    std::optional<Coords> xcoords, std::optional<Coords> ycoords) {
                     return chain(self, [&](Axes& ax) {
                         const std::size_t k = index(i, ax.text_count());
                         const sextant::TextData cur = ax.text_data(k);
                         const Coords cx = xcoords ? *xcoords : coords ? *coords : cur.x.space;
                         const Coords cy = ycoords ? *ycoords : coords ? *coords : cur.y.space;
                         ax.set_text_data(k, s, pos(x, cx), pos(y, cy));
                     });
                 },
                 "i"_a, "s"_a, "x"_a, "y"_a, nb::kw_only(), "coords"_a = nb::none(), "xcoords"_a = nb::none(),
                 "ycoords"_a = nb::none(),
                 "The string and position only; an annotation keeps its arrow and point. A\n"
                 "coordinate keeps its current coords unless coords/xcoords/ycoords says otherwise.")

            // --- decoration ------------------------------------------------
            .def("set_title",
                 [](Self self, std::string text, float fontsize) {
                     return chain(self, [&](Axes& ax) { ax.set_title(text, fontsize); });
                 },
                 "text"_a, "fontsize"_a = 18.0f)
            .def("set_xtitle",
                 [](Self self, std::string text, float fontsize) {
                     return chain(self, [&](Axes& ax) { ax.set_xtitle(text, fontsize); });
                 },
                 "text"_a, "fontsize"_a = 16.5f)
            .def("set_ytitle",
                 [](Self self, std::string text, float fontsize) {
                     return chain(self, [&](Axes& ax) { ax.set_ytitle(text, fontsize); });
                 },
                 "text"_a, "fontsize"_a = 16.5f)
            .def("set_xlim",
                 [](Self self, double lo, double hi) { return chain(self, [&](Axes& ax) { ax.set_xlim(lo, hi); }); },
                 "lo"_a, "hi"_a)
            .def("set_ylim",
                 [](Self self, double lo, double hi) { return chain(self, [&](Axes& ax) { ax.set_ylim(lo, hi); }); },
                 "lo"_a, "hi"_a)
            .def("grid",
                 [](Self self, bool enable, const nb::kwargs& kw) {
                     auto o = options<sextant::GridOptions>("grid()", kw);
                     return chain(self, [&](Axes& ax) { ax.grid(enable, o); });
                 },
                 "enable"_a = true, "opts"_a, grid_doc.c_str())
            .def("set_axes_style",
                 [](Self self, const nb::kwargs& kw) {
                     auto o = options<sextant::AxesStyle>("set_axes_style()", kw);
                     return chain(self, [&](Axes& ax) { ax.set_axes_style(std::move(o)); });
                 },
                 "opts"_a, style_doc.c_str())
            .def("legend",
                 [](Self self, const nb::kwargs& kw) {
                     auto o = options<sextant::LegendOptions>("legend()", kw);
                     return chain(self, [&](Axes& ax) { ax.legend(std::move(o)); });
                 },
                 "opts"_a, legend_doc.c_str())
            .def("set_colorbar_style",
                 [](Self self, const nb::kwargs& kw) {
                     auto o = options<sextant::ColorbarOptions>("set_colorbar_style()", kw);
                     return chain(self, [&](Axes& ax) { ax.set_colorbar_style(std::move(o)); });
                 },
                 "opts"_a, colorbar_doc.c_str())
            .def("set_xticks",
                 [](Self self, const Vec& positions, std::vector<std::string> labels) {
                     return chain(self, [&](Axes& ax) { ax.set_xticks(positions.span(), std::move(labels)); });
                 },
                 "positions"_a, "labels"_a = std::vector<std::string>{})
            .def("set_yticks",
                 [](Self self, const Vec& positions, std::vector<std::string> labels) {
                     return chain(self, [&](Axes& ax) { ax.set_yticks(positions.span(), std::move(labels)); });
                 },
                 "positions"_a, "labels"_a = std::vector<std::string>{})
            .def("set_xscale",
                 [](Self self, sextant::Scale scale, const nb::kwargs& kw) {
                     auto o = options<sextant::ScaleOptions>("set_xscale()", kw);
                     return chain(self, [&](Axes& ax) { ax.set_xscale(scale, o); });
                 },
                 "scale"_a, "opts"_a, xscale_doc.c_str())
            .def("set_yscale",
                 [](Self self, sextant::Scale scale, const nb::kwargs& kw) {
                     auto o = options<sextant::ScaleOptions>("set_yscale()", kw);
                     return chain(self, [&](Axes& ax) { ax.set_yscale(scale, o); });
                 },
                 "scale"_a, "opts"_a, yscale_doc.c_str())
            .def("set_xtick_format",
                 [](Self self, std::string fmt) { return chain(self, [&](Axes& ax) { ax.set_xtick_format(fmt); }); },
                 "fmt"_a,
                 "printf's format for one double, used for every x tick label: exactly one %f, %F,\n"
                 "%e, %E, %g or %G (flags, width, precision allowed) plus any %% and other text:\n"
                 "'%.1f', '%.0f%%', '$%.2g$ s'. ValueError otherwise. '' restores the default.")
            .def("set_ytick_format",
                 [](Self self, std::string fmt) { return chain(self, [&](Axes& ax) { ax.set_ytick_format(fmt); }); },
                 "fmt"_a, "set_xtick_format() for the y axis.")
            .def("set_color_cycle",
                 [](Self self, std::vector<sextant::Color> colors) {
                     return chain(self, [&](Axes& ax) { ax.set_color_cycle(std::move(colors)); });
                 },
                 "colors"_a,
                 "The colors an unset `color` takes in turn: line(), scatter(), bar(), hist(),\n"
                 "fill() (stack() one per layer), quiver() without c, axline() (not axspan()).\n"
                 "Restarts from the first; [] restores the default, matplotlib's tab10. Objects\n"
                 "already plotted keep theirs. A twin shares its host's cycle.")
            .def("sharex",
                 [](Self self, PyAxes& other) {
                     same_figure(*self.p, other, "sharex()");
                     return chain(self, [&](Axes& ax) { ax.sharex(*other.obj); });
                 },
                 "other"_a,
                 "Share the x axis with `other` (and every axes it shares it with): one set of\n"
                 "limits and one scale, moved together by setters, pan and zoom; this axes takes\n"
                 "other's. Inner tick labels hide by themselves (AxesStyle.show_xtick_labels).\n"
                 "ValueError for another figure's axes, or a heatmap's axes taking a log/symlog\n"
                 "scale. There is no unsharing.")
            .def("sharey",
                 [](Self self, PyAxes& other) {
                     same_figure(*self.p, other, "sharey()");
                     return chain(self, [&](Axes& ax) { ax.sharey(*other.obj); });
                 },
                 "other"_a, "sharex() for the y axis.")
            .def("twinx",
                 [](PyAxes& a) {
                     auto twin = read(a, [](Axes& ax) { return ax.twinx(); });
                     return wrap<PyAxes>(a.st, std::move(twin));
                 },
                 "A second axes over this one's frame, sharing its x axis, its own y axis on the\n"
                 "right. The same object on every call. The cell's title, legend and colorbars\n"
                 "stay this axes' (set_title()/legend() on the twin set them; the legend lists\n"
                 "both axes' series); the twin continues this axes' color cycle. ValueError on a\n"
                 "twin.")
            .def("twiny",
                 [](PyAxes& a) {
                     auto twin = read(a, [](Axes& ax) { return ax.twiny(); });
                     return wrap<PyAxes>(a.st, std::move(twin));
                 },
                 "twinx() with the axes swapped: shares the y axis, its own x axis at the top.")
            .def("set_aspect",
                 [](Self self, double ratio, sextant::Adjustable adjustable) {
                     return chain(self, [&](Axes& ax) { ax.set_aspect(ratio, adjustable); });
                 },
                 "ratio"_a, "adjustable"_a = sextant::Adjustable::Box,
                 "Draw one y unit `ratio` times as long as one x unit (1: equal; 0: auto, fill the\n"
                 "frame), in the scales' units. adjustable='box' shrinks the frame, 'datalim'\n"
                 "widens one axis' limits as drawn. ValueError for a negative or non-finite ratio,\n"
                 "on a twin, and for 'datalim' when both axes are shared. cla() resets it.")
            .def("set_axis_visible",
                 [](Self self, bool on) { return chain(self, [&](Axes& ax) { ax.set_axis_visible(on); }); },
                 "on"_a = true,
                 "Draw the axis or not, as matplotlib's axis('off'): off hides the spines, tick marks\n"
                 "and labels, the x/y titles, the grid and the background fill, and the layout gives\n"
                 "their room to the frame. The data, title, legend, colorbars and texts stay, as do\n"
                 "pan, zoom and hover; the style setters keep their values. On a twin, its own layer\n"
                 "only. pie() turns it off; cla() turns it back on.")
            .def("axis_visible", [](PyAxes& a) { return read(a, [](Axes& ax) { return ax.axis_visible(); }); })

            // --- read-back -------------------------------------------------
            .def("title", [](PyAxes& a) { return read(a, [](Axes& ax) { return ax.title(); }); })
            .def("xtitle", [](PyAxes& a) { return read(a, [](Axes& ax) { return ax.xtitle(); }); })
            .def("ytitle", [](PyAxes& a) { return read(a, [](Axes& ax) { return ax.ytitle(); }); })
            .def("xlim", [](PyAxes& a) { return read(a, [](Axes& ax) { return ax.xlim(); }); },
                 "The x limits as drawn: set_xlim()'s, or the data's auto scale.")
            .def("ylim", [](PyAxes& a) { return read(a, [](Axes& ax) { return ax.ylim(); }); })
            .def("xscale", [](PyAxes& a) { return read(a, [](Axes& ax) { return ax.xscale(); }); })
            .def("yscale", [](PyAxes& a) { return read(a, [](Axes& ax) { return ax.yscale(); }); })
            .def("aspect", [](PyAxes& a) { return read(a, [](Axes& ax) { return ax.aspect(); }); },
                 "set_aspect()'s ratio; 0 is auto.")
            .def("adjustable", [](PyAxes& a) { return read(a, [](Axes& ax) { return ax.adjustable(); }); });
    }

    void bind_plane(nb::module_& m) {
        using sextant::Plane2D;
        using Self = SelfOf<Plane2D>;

        nb::class_<PyPlane> cls(m, "Plane2D", nb::is_weak_referenceable(),
                                "A 2D plane in a 3D scene (Axes3D.plane()), carrying the 2D plot kinds in the\n"
                                "parent's data coordinates along its two axes: xy -> (x, y), yz -> (y, z),\n"
                                "zx -> (z, x). Limits, ticks and titles belong to the parent.");
        bind_plot2d<Plane2D>(cls);
        cls.def("set_offset",
                [](Self self, double offset) { return chain(self, [&](Plane2D& p) { p.set_offset(offset); }); },
                "offset"_a, "Position along the plane's normal axis, in data units.")
            .def("set_alpha",
                 [](Self self, float alpha) { return chain(self, [&](Plane2D& p) { p.set_alpha(alpha); }); },
                 "alpha"_a, "Whole-plane opacity; multiplies the objects' own alpha.")
            .def("id", [](PyPlane& p) { return read(p, [](Plane2D& pl) { return pl.id(); }); },
                 "The plane's id, as events' pick_plane names it and Axes3D.remove_plane() takes it.")
            .def("orientation", [](PyPlane& p) { return read(p, [](Plane2D& pl) { return pl.orientation(); }); })
            .def("offset", [](PyPlane& p) { return read(p, [](Plane2D& pl) { return pl.offset(); }); });
    }
} // namespace sextant_py
