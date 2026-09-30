// Axes3D.
#include "options.h"
#include "wrappers.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace sextant_py {
    namespace {
        using sextant::Axes3D;
        using sextant::PlaneOrientation;
        using Self = SelfOf<Axes3D>;

        template <class T>
        std::string opts_doc(const char* signature, const char* text) {
            return std::string(signature) + "\n\n" + text + "\n\nKeyword options (" + Fields<T>::name +
                   "): " + field_names<T>() + ".";
        }

        sextant::ErrorBar3D spans(const PyErrorBar3D* err) { return err ? err->spans() : sextant::ErrorBar3D{}; }

        // A bar3d/surface `heights` (or `bottoms`): flat, or 2-D with shape
        // (len(u), len(v)) -- u major, as sextant lays it out. A transposed
        // array has the right length and would be misread, so it is refused.
        std::span<const double> grid_values(const Grid& g, std::size_t nu, std::size_t nv, const std::string& what) {
            if (g.ndim() == 2 && (g.shape(0) != nu || g.shape(1) != nv))
                throw nb::value_error((what + " has shape (" + std::to_string(g.shape(0)) + ", " +
                                       std::to_string(g.shape(1)) + "); expected (len(u), len(v)) = (" +
                                       std::to_string(nu) + ", " + std::to_string(nv) + ")").c_str());
            if (g.ndim() > 2)
                throw nb::value_error((what + " must be 1-D or (len(u), len(v))").c_str());
            return g.span();
        }

        Grid as_grid(nb::handle h, const char* what) {
            try {
                return nb::cast<Grid>(h);
            } catch (const nb::cast_error&) {
                throw nb::type_error((std::string(what) + ": expected an array of numbers, got " +
                                      nb::cast<std::string>(nb::repr(h))).c_str());
            }
        }

        // Triangle indices of any integer dtype, (n, 3) or flat, as uint32.
        struct Triangles {
            nb::object keep; // owns the uint32 array the span points into
            std::span<const std::uint32_t> s;
        };

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

        nb::dict camera_dict(const sextant::Camera3D& c) {
            nb::dict d;
            d["azimuth"] = c.azimuth;
            d["elevation"] = c.elevation;
            d["target"] = nb::cast(c.target);
            d["zoom"] = c.zoom;
            d["projection"] = nb::cast(c.projection);
            d["fov"] = c.fov;
            return d;
        }
    } // namespace

    void bind_axes3d(nb::module_& m) {
        static const std::string bar3d_doc = opts_doc<sextant::Bar3DOptions>(
            "bar3d(orient, u, v, heights, *, bottoms=None, **opts)",
            "Bars on the grid u x v, standing along orient's normal ('xy' -> along z). heights (and bottoms)\n"
            "are (len(u), len(v)) or flat, u major.");
        static const std::string surface_doc = opts_doc<sextant::SurfaceOptions>(
            "surface(orient, u, v, heights, **opts)",
            "A surface over the grid u x v (at least 2 x 2), rising along orient's normal; heights as bar3d.");
        static const std::string surface_tri_doc = opts_doc<sextant::SurfaceTriOptions>(
            "surface_tri(x, y, z, triangles=None, *, orient=None, colors=None, **opts)",
            "A sheet on a triangulated mesh. Give exactly one of triangles ((n, 3) or flat, any integer dtype)\n"
            "and orient (Delaunay triangulation of the points projected onto that plane). colors, one per\n"
            "vertex, colormaps the mesh.");
        static const std::string scatter3d_doc = opts_doc<sextant::Scatter3DOptions>(
            "scatter3d(x, y, z, *, colors=None, err=None, **opts)",
            "Markers at the points; colors (one per point) colormaps them. In 3D z is a coordinate.");
        static const std::string line3d_doc = opts_doc<sextant::Line3DOptions>(
            "line3d(x, y, z, *, colors=None, err=None, **opts)", "A path through the points in order.");
        static const std::string plane_doc = opts_doc<sextant::Plane2DOptions>(
            "plane(orient, offset=0.0, **opts)",
            "A Plane2D spanning orient's two axes at offset (data units) along the third.");
        static const std::string grid_doc = opts_doc<sextant::GridOptions>(
            "grid(enable=True, **opts)", "Grid lines on the three back panes (on by default).");
        static const std::string style_doc = opts_doc<sextant::AxesStyle>(
            "set_axes_style(**opts)", "Axis annotation, shared with 2D. Unnamed fields reset to their defaults.");
        static const std::string box_doc = opts_doc<sextant::Box3DStyle>(
            "set_box_style(**opts)", "The box's panes and margin. Unnamed fields reset to their defaults.");
        static const std::string legend_doc = opts_doc<sextant::LegendOptions>(
            "legend(**opts)", "Keys every named series of this axes, then of its planes.");
        static const std::string colorbar_doc = opts_doc<sextant::ColorbarOptions>(
            "set_colorbar_style(**opts)", "Style of every colorbar in the cell, planes' included.");
        static const std::string camera_doc = opts_doc<sextant::Camera3D>(
            "set_camera(**opts)", "Replace the camera; unnamed fields take their defaults. "
                                  "ax.set_camera(**ax.camera()) round-trips.");
        static const std::string default_camera_doc = opts_doc<sextant::Camera3D>(
            "set_default_camera(**opts)", "The camera a double-click in the window resets to.");

        nb::class_<PyAxes3D>(m, "Axes3D", nb::is_weak_referenceable())
            // --- plotting ------------------------------------------------
            .def("bar3d",
                 [](Self self, PlaneOrientation orient, const Vec& u, const Vec& v, const Grid& heights,
                    const std::optional<Grid>& bottoms, const nb::kwargs& kw) {
                     auto o = options<sextant::Bar3DOptions>("bar3d()", kw);
                     const std::size_t nu = u.a.size(), nv = v.a.size();
                     const auto h = grid_values(heights, nu, nv, "bar3d(): heights");
                     const auto b = bottoms ? grid_values(*bottoms, nu, nv, "bar3d(): bottoms")
                                            : std::span<const double>{};
                     return chain(self, [&](Axes3D& ax) {
                         if (bottoms) ax.bar3d(orient, u.span(), v.span(), h, b, std::move(o));
                         else ax.bar3d(orient, u.span(), v.span(), h, std::move(o));
                     });
                 },
                 "orient"_a, "u"_a, "v"_a, "heights"_a, nb::kw_only(), "bottoms"_a = nb::none(), "opts"_a,
                 bar3d_doc.c_str())
            .def("surface",
                 [](Self self, PlaneOrientation orient, const Vec& u, const Vec& v, const Grid& heights,
                    const nb::kwargs& kw) {
                     auto o = options<sextant::SurfaceOptions>("surface()", kw);
                     const auto h = grid_values(heights, u.a.size(), v.a.size(), "surface(): heights");
                     return chain(self, [&](Axes3D& ax) { ax.surface(orient, u.span(), v.span(), h, std::move(o)); });
                 },
                 "orient"_a, "u"_a, "v"_a, "heights"_a, "opts"_a, surface_doc.c_str())
            .def("surface_tri",
                 [](Self self, const Vec& x, const Vec& y, const Vec& z, const U32& tri,
                    std::optional<PlaneOrientation> orient, const std::optional<Vec>& colors, const nb::kwargs& kw) {
                     if (tri.is_none() == !orient.has_value())
                         throw nb::type_error("surface_tri(): pass exactly one of triangles and orient");
                     auto o = options<sextant::SurfaceTriOptions>("surface_tri()", kw);
                     std::optional<Triangles> t;
                     if (!tri.is_none()) t = triangles(tri, "surface_tri()");
                     return chain(self, [&](Axes3D& ax) {
                         if (t && colors) ax.surface_tri(x.span(), y.span(), z.span(), t->s, colors->span(), std::move(o));
                         else if (t) ax.surface_tri(x.span(), y.span(), z.span(), t->s, std::move(o));
                         else if (colors) ax.surface_tri(x.span(), y.span(), z.span(), *orient, colors->span(), std::move(o));
                         else ax.surface_tri(x.span(), y.span(), z.span(), *orient, std::move(o));
                     });
                 },
                 "x"_a, "y"_a, "z"_a, "triangles"_a.none() = nb::none(), nb::kw_only(),
                 "orient"_a = nb::none(), "colors"_a = nb::none(), "opts"_a, surface_tri_doc.c_str())
            .def("scatter3d",
                 [](Self self, const Vec& x, const Vec& y, const Vec& z, const std::optional<Vec>& colors,
                    const PyErrorBar3D* err, const nb::kwargs& kw) {
                     auto o = options<sextant::Scatter3DOptions>("scatter3d()", kw);
                     const auto e = spans(err);
                     return chain(self, [&](Axes3D& ax) {
                         if (colors && err) ax.scatter3d(x.span(), y.span(), z.span(), colors->span(), e, std::move(o));
                         else if (colors) ax.scatter3d(x.span(), y.span(), z.span(), colors->span(), std::move(o));
                         else if (err) ax.scatter3d(x.span(), y.span(), z.span(), e, std::move(o));
                         else ax.scatter3d(x.span(), y.span(), z.span(), std::move(o));
                     });
                 },
                 "x"_a, "y"_a, "z"_a, nb::kw_only(), "colors"_a = nb::none(), "err"_a.none() = nb::none(),
                 "opts"_a, scatter3d_doc.c_str())
            .def("line3d",
                 [](Self self, const Vec& x, const Vec& y, const Vec& z, const std::optional<Vec>& colors,
                    const PyErrorBar3D* err, const nb::kwargs& kw) {
                     auto o = options<sextant::Line3DOptions>("line3d()", kw);
                     const auto e = spans(err);
                     return chain(self, [&](Axes3D& ax) {
                         if (colors && err) ax.line3d(x.span(), y.span(), z.span(), colors->span(), e, std::move(o));
                         else if (colors) ax.line3d(x.span(), y.span(), z.span(), colors->span(), std::move(o));
                         else if (err) ax.line3d(x.span(), y.span(), z.span(), e, std::move(o));
                         else ax.line3d(x.span(), y.span(), z.span(), std::move(o));
                     });
                 },
                 "x"_a, "y"_a, "z"_a, nb::kw_only(), "colors"_a = nb::none(), "err"_a.none() = nb::none(),
                 "opts"_a, line3d_doc.c_str())

            // --- planes ----------------------------------------------------
            .def("plane",
                 [](PyAxes3D& a, PlaneOrientation orient, double offset, const nb::kwargs& kw) {
                     auto o = options<sextant::Plane2DOptions>("plane()", kw);
                     auto p = read(a, [&](Axes3D& ax) { return ax.plane(orient, offset, o); });
                     return wrap<PyPlane>(a.st, std::move(p));
                 },
                 "orient"_a, "offset"_a = 0.0, "opts"_a, plane_doc.c_str())
            .def("plane_count", [](PyAxes3D& a) { return read(a, [](Axes3D& ax) { return ax.plane_count(); }); })
            .def("plane_at",
                 [](PyAxes3D& a, std::int64_t i) {
                     auto p = read(a, [&](Axes3D& ax) { return ax.plane_at(index(i, ax.plane_count())); });
                     return wrap<PyPlane>(a.st, std::move(p));
                 },
                 "i"_a, "The i-th plane plane() made -- the same object; the index a Pick event's plane names.")

            // --- decoration ------------------------------------------------
            .def("set_title",
                 [](Self self, std::string text, float fontsize) {
                     return chain(self, [&](Axes3D& ax) { ax.set_title(text, fontsize); });
                 },
                 "text"_a, "fontsize"_a = 18.0f)
            .def("set_xtitle",
                 [](Self self, std::string text, float fontsize) {
                     return chain(self, [&](Axes3D& ax) { ax.set_xtitle(text, fontsize); });
                 },
                 "text"_a, "fontsize"_a = 16.5f)
            .def("set_ytitle",
                 [](Self self, std::string text, float fontsize) {
                     return chain(self, [&](Axes3D& ax) { ax.set_ytitle(text, fontsize); });
                 },
                 "text"_a, "fontsize"_a = 16.5f)
            .def("set_ztitle",
                 [](Self self, std::string text, float fontsize) {
                     return chain(self, [&](Axes3D& ax) { ax.set_ztitle(text, fontsize); });
                 },
                 "text"_a, "fontsize"_a = 16.5f)
            .def("set_xlim",
                 [](Self self, double lo, double hi) { return chain(self, [&](Axes3D& ax) { ax.set_xlim(lo, hi); }); },
                 "lo"_a, "hi"_a)
            .def("set_ylim",
                 [](Self self, double lo, double hi) { return chain(self, [&](Axes3D& ax) { ax.set_ylim(lo, hi); }); },
                 "lo"_a, "hi"_a)
            .def("set_zlim",
                 [](Self self, double lo, double hi) { return chain(self, [&](Axes3D& ax) { ax.set_zlim(lo, hi); }); },
                 "lo"_a, "hi"_a)
            .def("set_xticks",
                 [](Self self, const Vec& positions, std::vector<std::string> labels) {
                     return chain(self, [&](Axes3D& ax) { ax.set_xticks(positions.span(), std::move(labels)); });
                 },
                 "positions"_a, "labels"_a = std::vector<std::string>{})
            .def("set_yticks",
                 [](Self self, const Vec& positions, std::vector<std::string> labels) {
                     return chain(self, [&](Axes3D& ax) { ax.set_yticks(positions.span(), std::move(labels)); });
                 },
                 "positions"_a, "labels"_a = std::vector<std::string>{})
            .def("set_zticks",
                 [](Self self, const Vec& positions, std::vector<std::string> labels) {
                     return chain(self, [&](Axes3D& ax) { ax.set_zticks(positions.span(), std::move(labels)); });
                 },
                 "positions"_a, "labels"_a = std::vector<std::string>{})
            .def("grid",
                 [](Self self, bool enable, const nb::kwargs& kw) {
                     auto o = options<sextant::GridOptions>("grid()", kw);
                     return chain(self, [&](Axes3D& ax) { ax.grid(enable, o); });
                 },
                 "enable"_a = true, "opts"_a, grid_doc.c_str())
            .def("set_axes_style",
                 [](Self self, const nb::kwargs& kw) {
                     auto o = options<sextant::AxesStyle>("set_axes_style()", kw);
                     return chain(self, [&](Axes3D& ax) { ax.set_axes_style(std::move(o)); });
                 },
                 "opts"_a, style_doc.c_str())
            .def("set_box_style",
                 [](Self self, const nb::kwargs& kw) {
                     auto o = options<sextant::Box3DStyle>("set_box_style()", kw);
                     return chain(self, [&](Axes3D& ax) { ax.set_box_style(o); });
                 },
                 "opts"_a, box_doc.c_str())
            .def("set_box_aspect",
                 [](Self self, sextant::BoxAspect aspect) {
                     return chain(self, [&](Axes3D& ax) { ax.set_box_aspect(aspect); });
                 },
                 "aspect"_a, "Box side lengths as an (x, y, z) tuple; (1, 1, 1) is a cube.")
            .def("legend",
                 [](Self self, const nb::kwargs& kw) {
                     auto o = options<sextant::LegendOptions>("legend()", kw);
                     return chain(self, [&](Axes3D& ax) { ax.legend(std::move(o)); });
                 },
                 "opts"_a, legend_doc.c_str())
            .def("set_colorbar_style",
                 [](Self self, const nb::kwargs& kw) {
                     auto o = options<sextant::ColorbarOptions>("set_colorbar_style()", kw);
                     return chain(self, [&](Axes3D& ax) { ax.set_colorbar_style(std::move(o)); });
                 },
                 "opts"_a, colorbar_doc.c_str())

            // --- camera ----------------------------------------------------
            .def("set_view",
                 [](Self self, double azimuth, double elevation) {
                     return chain(self, [&](Axes3D& ax) { ax.set_view(azimuth, elevation); });
                 },
                 "azimuth"_a, "elevation"_a, "Degrees; elevation is clamped to +-89.")
            .def("set_camera",
                 [](Self self, const nb::kwargs& kw) {
                     auto c = options<sextant::Camera3D>("set_camera()", kw);
                     return chain(self, [&](Axes3D& ax) { ax.set_camera(c); });
                 },
                 "opts"_a, camera_doc.c_str())
            .def("camera",
                 [](PyAxes3D& a) { return camera_dict(read(a, [](Axes3D& ax) { return ax.camera(); })); },
                 "The camera as a dict: azimuth, elevation, target (x, y, z), zoom, projection, fov.")
            .def("set_projection",
                 [](Self self, sextant::Projection mode) {
                     return chain(self, [&](Axes3D& ax) { ax.set_projection(mode); });
                 },
                 "mode"_a, "'orthographic' or 'perspective'.")
            .def("set_fov",
                 [](Self self, double degrees) { return chain(self, [&](Axes3D& ax) { ax.set_fov(degrees); }); },
                 "degrees"_a, "Perspective field of view, clamped to [5, 120].")
            .def("set_default_camera",
                 [](Self self, const nb::kwargs& kw) {
                     auto c = options<sextant::Camera3D>("set_default_camera()", kw);
                     return chain(self, [&](Axes3D& ax) { ax.set_default_camera(c); });
                 },
                 "opts"_a, default_camera_doc.c_str())
            .def("cla", [](Self self) { return chain(self, [](Axes3D& ax) { ax.cla(); }); })

            // --- read-back -------------------------------------------------
            .def("title", [](PyAxes3D& a) { return read(a, [](Axes3D& ax) { return ax.title(); }); })
            .def("xtitle", [](PyAxes3D& a) { return read(a, [](Axes3D& ax) { return ax.xtitle(); }); })
            .def("ytitle", [](PyAxes3D& a) { return read(a, [](Axes3D& ax) { return ax.ytitle(); }); })
            .def("ztitle", [](PyAxes3D& a) { return read(a, [](Axes3D& ax) { return ax.ztitle(); }); })
            .def("xlim", [](PyAxes3D& a) { return read(a, [](Axes3D& ax) { return ax.xlim(); }); })
            .def("ylim", [](PyAxes3D& a) { return read(a, [](Axes3D& ax) { return ax.ylim(); }); })
            .def("zlim", [](PyAxes3D& a) { return read(a, [](Axes3D& ax) { return ax.zlim(); }); })
            .def("bar3d_count", [](PyAxes3D& a) { return read(a, [](Axes3D& ax) { return ax.bar3d_count(); }); })
            .def("surface_count", [](PyAxes3D& a) { return read(a, [](Axes3D& ax) { return ax.surface_count(); }); })
            .def("surface_tri_count",
                 [](PyAxes3D& a) { return read(a, [](Axes3D& ax) { return ax.surface_tri_count(); }); })
            .def("scatter3d_count",
                 [](PyAxes3D& a) { return read(a, [](Axes3D& ax) { return ax.scatter3d_count(); }); })
            .def("line3d_count", [](PyAxes3D& a) { return read(a, [](Axes3D& ax) { return ax.line3d_count(); }); })
            .def("bar3d_data",
                 [](PyAxes3D& a, std::int64_t i) {
                     return to_python(read(a, [&](Axes3D& ax) { return ax.bar3d_data(index(i, ax.bar3d_count())); }));
                 },
                 "i"_a)
            .def("surface_data",
                 [](PyAxes3D& a, std::int64_t i) {
                     return to_python(
                         read(a, [&](Axes3D& ax) { return ax.surface_data(index(i, ax.surface_count())); }));
                 },
                 "i"_a)
            .def("surface_tri_data",
                 [](PyAxes3D& a, std::int64_t i) {
                     return to_python(
                         read(a, [&](Axes3D& ax) { return ax.surface_tri_data(index(i, ax.surface_tri_count())); }));
                 },
                 "i"_a)
            .def("scatter3d_data",
                 [](PyAxes3D& a, std::int64_t i) {
                     return to_python(
                         read(a, [&](Axes3D& ax) { return ax.scatter3d_data(index(i, ax.scatter3d_count())); }));
                 },
                 "i"_a)
            .def("line3d_data",
                 [](PyAxes3D& a, std::int64_t i) {
                     return to_python(read(a, [&](Axes3D& ax) { return ax.line3d_data(index(i, ax.line3d_count())); }));
                 },
                 "i"_a)

            // --- updating plotted data ------------------------------------
            .def("set_bar3d_data",
                 [](Self self, std::int64_t i, const PyBar3DData& d) {
                     const auto u = as_array<1>(d.u, "Bar3DData.u"), v = as_array<1>(d.v, "Bar3DData.v");
                     const auto hg = as_grid(d.heights, "Bar3DData.heights");
                     const auto h = grid_values(hg, u.a.size(), v.a.size(), "Bar3DData.heights");
                     std::optional<Grid> bg;
                     if (!d.bottoms.is_none()) bg = as_grid(d.bottoms, "Bar3DData.bottoms");
                     const auto b = bg ? grid_values(*bg, u.a.size(), v.a.size(), "Bar3DData.bottoms")
                                       : std::span<const double>{};
                     return chain(self, [&](Axes3D& ax) {
                         const std::size_t k = index(i, ax.bar3d_count());
                         if (bg) ax.set_bar3d_data(k, d.orient, u.span(), v.span(), h, b);
                         else ax.set_bar3d_data(k, d.orient, u.span(), v.span(), h);
                     });
                 },
                 "i"_a, "data"_a)
            .def("set_bar3d_data",
                 [](Self self, std::int64_t i, PlaneOrientation orient, const Vec& u, const Vec& v,
                    const Grid& heights, const std::optional<Grid>& bottoms) {
                     const auto h = grid_values(heights, u.a.size(), v.a.size(), "set_bar3d_data(): heights");
                     const auto b = bottoms ? grid_values(*bottoms, u.a.size(), v.a.size(), "set_bar3d_data(): bottoms")
                                            : std::span<const double>{};
                     return chain(self, [&](Axes3D& ax) {
                         const std::size_t k = index(i, ax.bar3d_count());
                         if (bottoms) ax.set_bar3d_data(k, orient, u.span(), v.span(), h, b);
                         else ax.set_bar3d_data(k, orient, u.span(), v.span(), h);
                     });
                 },
                 "i"_a, "orient"_a, "u"_a, "v"_a, "heights"_a, "bottoms"_a = nb::none())
            .def("set_surface_data",
                 [](Self self, std::int64_t i, const PySurfaceData& d) {
                     const auto u = as_array<1>(d.u, "SurfaceData.u"), v = as_array<1>(d.v, "SurfaceData.v");
                     const auto hg = as_grid(d.heights, "SurfaceData.heights");
                     const auto h = grid_values(hg, u.a.size(), v.a.size(), "SurfaceData.heights");
                     return chain(self, [&](Axes3D& ax) {
                         ax.set_surface_data(index(i, ax.surface_count()), d.orient, u.span(), v.span(), h);
                     });
                 },
                 "i"_a, "data"_a)
            .def("set_surface_data",
                 [](Self self, std::int64_t i, PlaneOrientation orient, const Vec& u, const Vec& v,
                    const Grid& heights) {
                     const auto h = grid_values(heights, u.a.size(), v.a.size(), "set_surface_data(): heights");
                     return chain(self, [&](Axes3D& ax) {
                         ax.set_surface_data(index(i, ax.surface_count()), orient, u.span(), v.span(), h);
                     });
                 },
                 "i"_a, "orient"_a, "u"_a, "v"_a, "heights"_a)
            .def("set_surface_tri_data",
                 [](Self self, std::int64_t i, const PySurfaceTriData& d) {
                     const auto x = as_array<1>(d.x, "SurfaceTriData.x"), y = as_array<1>(d.y, "SurfaceTriData.y"),
                                z = as_array<1>(d.z, "SurfaceTriData.z");
                     const auto t = triangles(d.tri, "SurfaceTriData.tri");
                     std::optional<Vec> c;
                     if (!d.colors.is_none()) c = as_array<1>(d.colors, "SurfaceTriData.colors");
                     return chain(self, [&](Axes3D& ax) {
                         const std::size_t k = index(i, ax.surface_tri_count());
                         if (c) ax.set_surface_tri_data(k, x.span(), y.span(), z.span(), t.s, c->span());
                         else ax.set_surface_tri_data(k, x.span(), y.span(), z.span(), t.s);
                     });
                 },
                 "i"_a, "data"_a)
            .def("set_surface_tri_data",
                 [](Self self, std::int64_t i, const Vec& x, const Vec& y, const Vec& z, const U32& tri,
                    const std::optional<Vec>& colors) {
                     const auto t = triangles(tri, "set_surface_tri_data()");
                     return chain(self, [&](Axes3D& ax) {
                         const std::size_t k = index(i, ax.surface_tri_count());
                         if (colors) ax.set_surface_tri_data(k, x.span(), y.span(), z.span(), t.s, colors->span());
                         else ax.set_surface_tri_data(k, x.span(), y.span(), z.span(), t.s);
                     });
                 },
                 "i"_a, "x"_a, "y"_a, "z"_a, "triangles"_a, "colors"_a = nb::none())
            .def("set_scatter3d_data",
                 [](Self self, std::int64_t i, const PyScatter3DData& d) {
                     const auto x = as_array<1>(d.x, "Scatter3DData.x"), y = as_array<1>(d.y, "Scatter3DData.y"),
                                z = as_array<1>(d.z, "Scatter3DData.z");
                     std::optional<Vec> c;
                     if (!d.colors.is_none()) c = as_array<1>(d.colors, "Scatter3DData.colors");
                     return chain(self, [&](Axes3D& ax) {
                         const std::size_t k = index(i, ax.scatter3d_count());
                         if (c) ax.set_scatter3d_data(k, x.span(), y.span(), z.span(), c->span());
                         else ax.set_scatter3d_data(k, x.span(), y.span(), z.span());
                     });
                 },
                 "i"_a, "data"_a)
            .def("set_scatter3d_data",
                 [](Self self, std::int64_t i, const Vec& x, const Vec& y, const Vec& z,
                    const std::optional<Vec>& colors) {
                     return chain(self, [&](Axes3D& ax) {
                         const std::size_t k = index(i, ax.scatter3d_count());
                         if (colors) ax.set_scatter3d_data(k, x.span(), y.span(), z.span(), colors->span());
                         else ax.set_scatter3d_data(k, x.span(), y.span(), z.span());
                     });
                 },
                 "i"_a, "x"_a, "y"_a, "z"_a, "colors"_a = nb::none())
            .def("set_line3d_data",
                 [](Self self, std::int64_t i, const PyLine3DData& d) {
                     const auto x = as_array<1>(d.x, "Line3DData.x"), y = as_array<1>(d.y, "Line3DData.y"),
                                z = as_array<1>(d.z, "Line3DData.z");
                     std::optional<Vec> c;
                     if (!d.colors.is_none()) c = as_array<1>(d.colors, "Line3DData.colors");
                     return chain(self, [&](Axes3D& ax) {
                         const std::size_t k = index(i, ax.line3d_count());
                         if (c) ax.set_line3d_data(k, x.span(), y.span(), z.span(), c->span());
                         else ax.set_line3d_data(k, x.span(), y.span(), z.span());
                     });
                 },
                 "i"_a, "data"_a)
            .def("set_line3d_data",
                 [](Self self, std::int64_t i, const Vec& x, const Vec& y, const Vec& z,
                    const std::optional<Vec>& colors) {
                     return chain(self, [&](Axes3D& ax) {
                         const std::size_t k = index(i, ax.line3d_count());
                         if (colors) ax.set_line3d_data(k, x.span(), y.span(), z.span(), colors->span());
                         else ax.set_line3d_data(k, x.span(), y.span(), z.span());
                     });
                 },
                 "i"_a, "x"_a, "y"_a, "z"_a, "colors"_a = nb::none());
    }
} // namespace sextant_py
