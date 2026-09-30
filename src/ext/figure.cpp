#include "state.h"
#include "wrappers.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

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
    } // namespace

    void bind_figure(nb::module_& m) {
        using sextant::Figure;

        nb::class_<PyFigure>(m, "Figure")
            .def("__init__",
                 [](PyFigure* self, int width, int height, std::string title) {
                     sextant::FigureOptions opts;
                     opts.width = width;
                     opts.height = height;
                     opts.title = std::move(title);
                     new (self) PyFigure{register_figure(Figure::create(std::move(opts)))};
                 },
                 "width"_a = 800, "height"_a = 600, "title"_a = "sextant")
            .def("axes",
                 [](PyFigure& self) {
                     auto ax = locked(*self.st, [&] { return self.st->fig->axes(); });
                     return wrap<PyAxes>(self.st, std::move(ax));
                 })
            .def("show",
                 [](PyFigure& self, std::optional<bool> block) {
                     // Never show(true): it reads the console.
                     locked(*self.st, [&] { self.st->fig->show(false); });
                     if (block.value_or(!interactive_session())) wait_closed(*self.st, -1);
                 },
                 "block"_a = nb::none(),
                 "Open the window. block=None blocks until it closes, except in an\n"
                 "interactive session (REPL, IPython, Jupyter).")
            .def("close", [](PyFigure& self) { locked(*self.st, [&] { self.st->fig->close(); }); })
            .def("is_open", [](PyFigure& self) { return self.st->fig->is_open(); })
            .def("wait_closed",
                 [](PyFigure& self, std::optional<double> timeout) {
                     return wait_closed(*self.st, timeout.value_or(-1.0));
                 },
                 "timeout"_a = nb::none(),
                 "Wait until the window has closed (True) or timeout seconds passed (False).")
            .def("savefig",
                 [](PyFigure& self, std::string path) {
                     locked(*self.st, [&] { self.st->fig->savefig(path); });
                 },
                 "path"_a)
            .def("render_png",
                 [](PyFigure& self) {
                     auto png = locked(*self.st, [&] { return self.st->fig->render_png(); });
                     return as_bytes(png);
                 })
            .def("__enter__", [](nb::handle self) { return nb::borrow(self); })
            .def("__exit__",
                 [](PyFigure& self, nb::args) { locked(*self.st, [&] { self.st->fig->close(); }); });
    }
} // namespace sextant_py
