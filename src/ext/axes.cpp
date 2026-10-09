// Axes and Plane2D: the 2D plot kinds, bound once for both, plus what each has
// of its own.
#include "options.h"
#include "wrappers.h"

#include <nanobind/stl/optional.h>
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

        sextant::ErrorBar spans(const PyErrorBar* err) { return err ? err->spans() : sextant::ErrorBar{}; }

        using sextant::Coords;
        using ArrowArg = nb::typed<nb::object, Named<"ArrowOptions">>;

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
                .def("cla", [](Self self) { return chain(self, [](T& t) { t.cla(); }); },
                     "Clear all plot objects (on Axes, also reset limits).")

                // --- read-back ---------------------------------------------
                .def("line_count", [](W& w) { return read(w, [](T& t) { return t.line_count(); }); })
                .def("scatter_count", [](W& w) { return read(w, [](T& t) { return t.scatter_count(); }); })
                .def("scatter_z_count", [](W& w) { return read(w, [](T& t) { return t.scatter_z_count(); }); })
                .def("bar_count", [](W& w) { return read(w, [](T& t) { return t.bar_count(); }); })
                .def("heatmap_count", [](W& w) { return read(w, [](T& t) { return t.heatmap_count(); }); })
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
                         const auto x = as_array<1>(d.x, "ScatterData.x"), y = as_array<1>(d.y, "ScatterData.y");
                         return chain(self, [&](T& t) {
                             t.set_scatter_data(index(i, t.scatter_count()), x.span(), y.span());
                         });
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
                         const auto x = as_array<1>(d.x, "ScatterZData.x"), y = as_array<1>(d.y, "ScatterZData.y"),
                                    z = as_array<1>(d.z, "ScatterZData.z");
                         return chain(self, [&](T& t) {
                             t.set_scatter_z_data(index(i, t.scatter_z_count()), x.span(), y.span(), z.span());
                         });
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
                         const auto x = as_array<1>(d.x, "BarData.x"), h = as_array<1>(d.height, "BarData.height");
                         return chain(self, [&](T& t) { t.set_bar_data(index(i, t.bar_count()), x.span(), h.span()); });
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
                         const auto data = as_array<2>(d.data, "HeatmapData.data");
                         return chain(self, [&](T& t) {
                             t.set_heatmap_data(index(i, t.heatmap_count()), data.span(),
                                                static_cast<int>(data.shape(0)), static_cast<int>(data.shape(1)),
                                                d.xrange, d.yrange);
                         });
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
                     "i"_a, "data"_a, "xrange"_a, "yrange"_a);
        }
    } // namespace

    void bind_axes(nb::module_& m) {
        using sextant::Axes;
        using Self = SelfOf<Axes>;

        static const std::string hist_doc = opts_doc<BarOptions>(
            "hist(data, bins=10, *, density=False, cumulative=False, **opts)",
            "Histogram as bars. width defaults to 1.0 here (bins touch), not bar()'s 0.8.");
        static const std::string grid_doc = opts_doc<sextant::GridOptions>("grid(enable=True, **opts)", "Grid lines.");
        static const std::string style_doc = opts_doc<sextant::AxesStyle>(
            "set_axes_style(**opts)", "Spines, ticks, labels and titles. Unnamed fields reset to their defaults.");
        static const std::string legend_doc =
            opts_doc<sextant::LegendOptions>("legend(**opts)", "Show the legend (series with a name).");
        static const std::string text_doc = opts_doc<sextant::TextOptions>(
            "text(s, x, y, *, coords='data', xcoords=None, ycoords=None, **opts)",
            "Text at (x, y), drawn over the data at a fixed pixel size. Each coordinate is in\n"
            "xcoords/ycoords (default: coords): 'data', or 'fraction' of the plot frame from its\n"
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

        nb::class_<PyAxes> cls(m, "Axes", nb::is_weak_referenceable());
        bind_plot2d<Axes>(cls);
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

            // --- read-back -------------------------------------------------
            .def("title", [](PyAxes& a) { return read(a, [](Axes& ax) { return ax.title(); }); })
            .def("xtitle", [](PyAxes& a) { return read(a, [](Axes& ax) { return ax.xtitle(); }); })
            .def("ytitle", [](PyAxes& a) { return read(a, [](Axes& ax) { return ax.ytitle(); }); })
            .def("xlim", [](PyAxes& a) { return read(a, [](Axes& ax) { return ax.xlim(); }); },
                 "The x limits as drawn: set_xlim()'s, or the data's auto scale.")
            .def("ylim", [](PyAxes& a) { return read(a, [](Axes& ax) { return ax.ylim(); }); });
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
            .def("orientation", [](PyPlane& p) { return read(p, [](Plane2D& pl) { return pl.orientation(); }); })
            .def("offset", [](PyPlane& p) { return read(p, [](Plane2D& pl) { return pl.offset(); }); });
    }
} // namespace sextant_py
