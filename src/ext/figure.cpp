#include "options.h"
#include "wrappers.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <cctype>
#include <cstdint>
#include <optional>
#include <string>

namespace sextant_py {
    namespace {
        // Where show() should not block: a REPL, `python -i`, IPython/Jupyter.
        bool interactive_session() {
            nb::module_ sys = nb::module_::import_("sys");
            if (nb::hasattr(sys, "ps1")) return true;
            if (nb::cast<int>(sys.attr("flags").attr("interactive"))) return true;
            nb::dict modules = nb::borrow<nb::dict>(sys.attr("modules"));
            if (modules.contains("IPython")) {
                nb::object get = nb::getattr(modules["IPython"], "get_ipython", nb::none());
                if (!get.is_none() && !get().is_none()) return true;
            }
            return false;
        }

        nb::bytes as_bytes(const std::vector<std::uint8_t>& v) {
            return nb::bytes(reinterpret_cast<const char*>(v.data()), v.size());
        }

        template <class F>
        auto call(PyFigure& self, F&& f) {
            return locked(*self.st, [&] { return f(*self.st->fig); });
        }

        template <class T>
        std::string opts_doc(const char* signature, const char* text) {
            return std::string(signature) + "\n\n" + text + "\n\nKeyword options (" + Fields<T>::name +
                   "): " + field_names<T>() + ".";
        }

        // A window shown in an interactive session keeps working between
        // statements: in IPython through its inputhook machinery
        // (sextant/_interactive.py), in the plain REPL through PyOS_InputHook.
        void keep_interactive_windows_live() {
            if (nb::cast<bool>(nb::module_::import_("sextant._interactive").attr("enable_ipython")())) return;
            install_input_hook();
        }

        // What _repr_png_/_repr_svg_ offer; set_repr_formats(). GIL-protected.
        bool g_repr_png = true, g_repr_svg = false;

        enum class Format { Png, Svg };

        Format parse_format(std::string f) {
            for (char& c : f) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (!f.empty() && f[0] == '.') f.erase(0, 1);
            if (f == "png") return Format::Png;
            if (f == "svg") return Format::Svg;
            throw nb::value_error(("unknown format '" + f + "'; sextant writes 'png' and 'svg'").c_str());
        }

        nb::object write_rendered(nb::handle file, Format fmt, PyFigure& self, int width, int height,
                                  const nb::kwargs& kw) {
            if (fmt == Format::Png) {
                auto o = options<sextant::PngExportOptions>("savefig()", kw);
                auto png = call(self, [&](sextant::Figure& f) { return f.render_png(o, width, height); });
                file.attr("write")(as_bytes(png));
                return nb::none();
            }
            auto o = options<sextant::SvgExportOptions>("savefig()", kw);
            auto r = call(self, [&](sextant::Figure& f) { return f.render_svg(o, width, height); });
            // A text stream takes str, anything else bytes.
            nb::object io = nb::module_::import_("io");
            if (nb::isinstance(file, io.attr("TextIOBase")))
                file.attr("write")(nb::str(r.svg.data(), r.svg.size()));
            else
                file.attr("write")(nb::bytes(r.svg.data(), r.svg.size()));
            return nb::cast(std::move(r.report));
        }
    } // namespace

    void bind_figure(nb::module_& m) {
        using sextant::Figure;
        using sextant::SubplotSpan;

        static const std::string init_doc = opts_doc<sextant::FigureOptions>(
            "Figure(**opts)", "A figure. Nothing is shown until show(); savefig() works without a window.");
        static const std::string suptitle_style_doc = opts_doc<sextant::SuptitleOptions>(
            "set_suptitle_style(**opts)", "Unnamed fields reset to their defaults.");
        static const std::string margins_doc = opts_doc<sextant::FigureMargins>(
            "set_margins(**opts)", "Figure edge to subplot grid, in pixels. Unnamed sides reset to 10.");
        static const std::string rgba_doc = opts_doc<sextant::PngExportOptions>(
            "render_rgba(*, width=0, height=0, **opts)",
            "The rendered figure as a (height, width, 4) uint8 array, top row first; width/height <= 0 use "
            "the figure's size.");
        static const std::string png_doc = opts_doc<sextant::PngExportOptions>(
            "render_png(*, width=0, height=0, **opts)", "The PNG file savefig() would write, as bytes.");
        static const std::string svg_doc = opts_doc<sextant::SvgExportOptions>(
            "render_svg(*, width=0, height=0, **opts)",
            "(svg_text, SvgSaveReport): the SVG savefig() would write, and whether its 3D order is exact.");
        static const std::string savefig_doc =
            "savefig(fname, *, format=None, width=0, height=0, **opts)\n\n"
            "Write the figure to fname: a path (str or os.PathLike) or a file object with write().\n"
            "format is 'png' or 'svg'; by default the path's extension, and PNG for a file object.\n"
            "A text file object gets SVG as str, a binary one bytes. width/height <= 0 use the\n"
            "figure's size. Keyword options: PngExportOptions (" + field_names<sextant::PngExportOptions>() +
            ") or SvgExportOptions (" + field_names<sextant::SvgExportOptions>() + ").\n"
            "Returns the SvgSaveReport for SVG, None for PNG.";

        nb::class_<sextant::SvgSaveReport>(m, "SvgSaveReport",
                                           "scene_order_exact False: part of a 3D scene was left in plain depth\n"
                                           "order (a budget ran out); `warning` says which and how to raise it.")
            .def_ro("scene_order_exact", &sextant::SvgSaveReport::scene_order_exact)
            .def_ro("splits", &sextant::SvgSaveReport::splits)
            .def_ro("tests", &sextant::SvgSaveReport::tests)
            .def_ro("warning", &sextant::SvgSaveReport::warning)
            .def("__repr__", [](const sextant::SvgSaveReport& r) {
                return "SvgSaveReport(scene_order_exact=" + std::string(r.scene_order_exact ? "True" : "False") +
                       ", splits=" + std::to_string(r.splits) + ", tests=" + std::to_string(r.tests) + ")";
            });

        m.def("set_repr_formats",
              [](nb::args formats) {
                  bool png = false, svg = false;
                  for (nb::handle f : formats)
                      (parse_format(nb::cast<std::string>(f)) == Format::Png ? png : svg) = true;
                  g_repr_png = png;
                  g_repr_svg = svg;
              },
              "set_repr_formats(*formats)\n\n"
              "What a Figure offers Jupyter/IPython to display it: 'png' (the default), 'svg', both,\n"
              "or none. SVG of a 3D scene can be slow to order, so it is opt-in.");

        nb::class_<sextant::FrameStats>(m, "FrameStats")
            .def_ro("frames", &sextant::FrameStats::frames)
            .def_ro("total_ms", &sextant::FrameStats::total_ms)
            .def_ro("last_ms", &sextant::FrameStats::last_ms)
            .def_ro("max_ms", &sextant::FrameStats::max_ms)
            .def("__repr__", [](const sextant::FrameStats& s) {
                return "FrameStats(frames=" + std::to_string(s.frames) + ", total_ms=" +
                       std::to_string(s.total_ms) + ", last_ms=" + std::to_string(s.last_ms) +
                       ", max_ms=" + std::to_string(s.max_ms) + ")";
            });

        static PyType_Slot figure_slots[] = {
            {Py_tp_traverse, reinterpret_cast<void*>(&figure_traverse)},
            {Py_tp_clear, reinterpret_cast<void*>(&figure_clear)},
            {0, nullptr}};
        nb::class_<PyFigure> cls(m, "Figure", nb::type_slots(figure_slots), nb::is_weak_referenceable());
        bind_events(m, cls);
        cls
            .def("__init__",
                 [](PyFigure* self, const nb::kwargs& kw) {
                     auto opts = options<sextant::FigureOptions>("Figure()", kw);
                     new (self) PyFigure{register_figure(Figure::create(std::move(opts)))};
                 },
                 "opts"_a, init_doc.c_str())

            // --- subplots ----------------------------------------------------
            .def("axes",
                 [](PyFigure& self) { return wrap<PyAxes>(self.st, call(self, [](Figure& f) { return f.axes(); })); },
                 "The single axes of a 1x1 grid (made on first use).")
            .def("add_subplot",
                 [](PyFigure& self, int rows, int cols, int index) {
                     return wrap<PyAxes>(self.st, call(self, [&](Figure& f) { return f.add_subplot(rows, cols, index); }));
                 },
                 "rows"_a, "cols"_a, "index"_a,
                 "The subplot at 1-based `index` of a rows x cols grid; `index` may be a (first, last) span.\n"
                 "The first call fixes the grid; returns the existing subplot at that cell or span.")
            .def("add_subplot",
                 [](PyFigure& self, int rows, int cols, SubplotSpan span) {
                     return wrap<PyAxes>(self.st, call(self, [&](Figure& f) { return f.add_subplot(rows, cols, span); }));
                 },
                 "rows"_a, "cols"_a, "index"_a)
            .def("add_subplot",
                 [](PyFigure& self, int index) {
                     return wrap<PyAxes>(self.st, call(self, [&](Figure& f) { return f.add_subplot(index); }));
                 },
                 "index"_a, "On the grid an earlier call fixed.")
            .def("add_subplot",
                 [](PyFigure& self, SubplotSpan span) {
                     return wrap<PyAxes>(self.st, call(self, [&](Figure& f) { return f.add_subplot(span); }));
                 },
                 "index"_a)
            .def("add_subplot3d",
                 [](PyFigure& self, int rows, int cols, int index) {
                     return wrap<PyAxes3D>(self.st,
                                           call(self, [&](Figure& f) { return f.add_subplot3d(rows, cols, index); }));
                 },
                 "rows"_a, "cols"_a, "index"_a,
                 "A 3D subplot on the same grid as add_subplot(), same rules; a cell holding a 2D axes throws.")
            .def("add_subplot3d",
                 [](PyFigure& self, int rows, int cols, SubplotSpan span) {
                     return wrap<PyAxes3D>(self.st,
                                           call(self, [&](Figure& f) { return f.add_subplot3d(rows, cols, span); }));
                 },
                 "rows"_a, "cols"_a, "index"_a)
            .def("add_subplot3d",
                 [](PyFigure& self, int index) {
                     return wrap<PyAxes3D>(self.st, call(self, [&](Figure& f) { return f.add_subplot3d(index); }));
                 },
                 "index"_a)
            .def("add_subplot3d",
                 [](PyFigure& self, SubplotSpan span) {
                     return wrap<PyAxes3D>(self.st, call(self, [&](Figure& f) { return f.add_subplot3d(span); }));
                 },
                 "index"_a)

            // --- layout --------------------------------------------------------
            .def("suptitle",
                 [](PyFigure& self, std::string text, float fontsize) {
                     call(self, [&](Figure& f) { f.suptitle(text, fontsize); });
                 },
                 "text"_a, "fontsize"_a = 21.0f)
            .def("set_suptitle_style",
                 [](PyFigure& self, const nb::kwargs& kw) {
                     auto o = options<sextant::SuptitleOptions>("set_suptitle_style()", kw);
                     call(self, [&](Figure& f) { f.set_suptitle_style(std::move(o)); });
                 },
                 "opts"_a, suptitle_style_doc.c_str())
            .def("set_margins",
                 [](PyFigure& self, const nb::kwargs& kw) {
                     auto o = options<sextant::FigureMargins>("set_margins()", kw);
                     call(self, [&](Figure& f) { f.set_margins(o); });
                 },
                 "opts"_a, margins_doc.c_str())
            .def("set_col_ratios",
                 [](PyFigure& self, std::vector<float> r) {
                     call(self, [&](Figure& f) { f.set_col_ratios(std::move(r)); });
                 },
                 "ratios"_a, "Relative column widths, e.g. [2, 1]; [] = equal.")
            .def("set_row_ratios",
                 [](PyFigure& self, std::vector<float> r) {
                     call(self, [&](Figure& f) { f.set_row_ratios(std::move(r)); });
                 },
                 "ratios"_a)
            .def("col_ratios", [](PyFigure& self) { return call(self, [](Figure& f) { return f.col_ratios(); }); })
            .def("row_ratios", [](PyFigure& self) { return call(self, [](Figure& f) { return f.row_ratios(); }); })
            .def("resize",
                 [](PyFigure& self, int width, int height) {
                     call(self, [&](Figure& f) { f.resize(width, height); });
                 },
                 "width"_a, "height"_a, "Resize the plot area (what savefig() writes).")
            .def("size_for_frame",
                 [](PyFigure& self, int frame_w, int frame_h, int slot_index) {
                     return call(self, [&](Figure& f) { return f.size_for_frame(frame_w, frame_h, slot_index); });
                 },
                 "frame_w"_a, "frame_h"_a, "slot_index"_a = 1,
                 "(width, height) at which subplot slot_index's data frame is frame_w x frame_h.")
            .def("resize_to_frame",
                 [](PyFigure& self, int frame_w, int frame_h, int slot_index) {
                     call(self, [&](Figure& f) { f.resize_to_frame(frame_w, frame_h, slot_index); });
                 },
                 "frame_w"_a, "frame_h"_a, "slot_index"_a = 1)

            // --- window --------------------------------------------------------
            .def("show",
                 [](PyFigure& self, std::optional<bool> block) {
                     // Never show(true): it reads the console.
                     call(self, [](Figure& f) { f.show(false); });
                     const bool interactive = interactive_session();
                     if (interactive) keep_interactive_windows_live();
                     if (block.value_or(!interactive)) wait_closed(*self.st, -1);
                 },
                 "block"_a = nb::none(),
                 "Open the window. block=None blocks until it closes, except in an\n"
                 "interactive session (REPL, IPython, Jupyter), where the window keeps\n"
                 "working between statements.")
            .def("close", [](PyFigure& self) { call(self, [](Figure& f) { f.close(); }); })
            .def("is_open", [](PyFigure& self) { return self.st->fig->is_open(); })
            .def("wait_closed",
                 [](PyFigure& self, std::optional<double> timeout) {
                     return wait_closed(*self.st, timeout.value_or(-1.0));
                 },
                 "timeout"_a = nb::none(),
                 "Wait until the window has closed (True) or timeout seconds passed (False).")
            .def("refresh", [](PyFigure& self) { call(self, [](Figure& f) { f.refresh(); }); },
                 "Publish changes to an open window.")
            .def("frame_stats", [](PyFigure& self) { return self.st->fig->frame_stats(); })

            // --- output ----------------------------------------------------------
            .def("savefig",
                 [](PyFigure& self, nb::object fname, std::optional<std::string> format, int width, int height,
                    const nb::kwargs& kw) -> nb::object {
                     const bool is_file = nb::hasattr(fname, "write");
                     std::string path;
                     if (!is_file) {
                         nb::object fs = nb::module_::import_("os").attr("fspath")(fname);
                         if (!nb::isinstance<nb::str>(fs))
                             throw nb::type_error("savefig(): a str path or a text-named PathLike, please");
                         path = nb::cast<std::string>(fs);
                     }
                     Format fmt = Format::Png;
                     if (format) {
                         fmt = parse_format(*format);
                     } else if (!is_file) {
                         const auto dot = path.find_last_of('.');
                         const auto sep = path.find_last_of("/\\");
                         if (dot == std::string::npos || (sep != std::string::npos && dot < sep))
                             throw nb::value_error(("savefig(): '" + path +
                                                    "' has no extension; give format='png' or 'svg'").c_str());
                         fmt = parse_format(path.substr(dot));
                     }
                     if (is_file) return write_rendered(fname, fmt, self, width, height, kw);
                     if (fmt == Format::Png) {
                         auto o = options<sextant::PngExportOptions>("savefig()", kw);
                         call(self, [&](Figure& f) { f.savefig_png(path, o, width, height); });
                         return nb::none();
                     }
                     auto o = options<sextant::SvgExportOptions>("savefig()", kw);
                     return nb::cast(call(self, [&](Figure& f) { return f.savefig_svg(path, o, width, height); }));
                 },
                 "fname"_a, nb::kw_only(), "format"_a = nb::none(), "width"_a = 0, "height"_a = 0, "opts"_a,
                 savefig_doc.c_str())
            .def("render_png",
                 [](PyFigure& self, int width, int height, const nb::kwargs& kw) {
                     auto o = options<sextant::PngExportOptions>("render_png()", kw);
                     return as_bytes(call(self, [&](Figure& f) { return f.render_png(o, width, height); }));
                 },
                 nb::kw_only(), "width"_a = 0, "height"_a = 0, "opts"_a, png_doc.c_str())
            .def("render_svg",
                 [](PyFigure& self, int width, int height, const nb::kwargs& kw) {
                     auto o = options<sextant::SvgExportOptions>("render_svg()", kw);
                     auto r = call(self, [&](Figure& f) { return f.render_svg(o, width, height); });
                     return nb::make_tuple(nb::str(r.svg.data(), r.svg.size()), std::move(r.report));
                 },
                 nb::kw_only(), "width"_a = 0, "height"_a = 0, "opts"_a, svg_doc.c_str())
            .def("_repr_png_",
                 [](PyFigure& self) -> nb::object {
                     if (!g_repr_png) return nb::none();
                     return as_bytes(call(self, [](Figure& f) { return f.render_png(); }));
                 })
            .def("_repr_svg_",
                 [](PyFigure& self) -> nb::object {
                     if (!g_repr_svg) return nb::none();
                     auto r = call(self, [](Figure& f) { return f.render_svg(); });
                     return nb::str(r.svg.data(), r.svg.size());
                 })
            .def("render_rgba",
                 [](PyFigure& self, int width, int height, const nb::kwargs& kw) {
                     auto o = options<sextant::PngExportOptions>("render_rgba()", kw);
                     auto img = call(self, [&](Figure& f) { return f.render_rgba(o, width, height); });
                     const auto h = static_cast<std::size_t>(img.height), w = static_cast<std::size_t>(img.width);
                     return to_numpy(std::move(img.pixels), {h, w, 4});
                 },
                 nb::kw_only(), "width"_a = 0, "height"_a = 0, "opts"_a, rgba_doc.c_str())

            .def("__enter__", [](nb::handle self) { return nb::borrow(self); })
            .def("__exit__", [](PyFigure& self, nb::args) { call(self, [](Figure& f) { f.close(); }); });
    }
} // namespace sextant_py
