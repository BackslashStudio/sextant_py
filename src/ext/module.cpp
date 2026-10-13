// The _sextant extension module: the C++ API mapped 1:1. Conversions that
// make it pleasant from Python belong in the sextant package's Python files,
// not here.
//
// Threading: every call into a figure's object graph goes through locked()
// (state.h); the waits run lock-free with the GIL released.
#include "options.h"
#include "wrappers.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <system_error>

NB_MODULE(_sextant, m) {
    namespace nb = nanobind;

    m.doc() = "sextant's C++ core; use the sextant package, not this module";

    // sextant reports file errors with the errno as the code, on every platform;
    // OSError(errno, msg) picks the matching subclass (FileNotFoundError, ...).
    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const std::system_error& e) {
            nb::object err = nb::handle(PyExc_OSError)(e.code().value(), e.what());
            PyErr_SetObject(reinterpret_cast<PyObject*>(Py_TYPE(err.ptr())), err.ptr());
        }
    });

    sextant_py::bind_data(m);
    sextant_py::bind_axes(m);
    sextant_py::bind_plane(m);
    sextant_py::bind_axes3d(m);
    sextant_py::bind_figure(m);
    sextant_py::bind_messages(m);
    sextant_py::bind_lifecycle(m);

    using namespace nb::literals;
    m.def("register_colormap",
          [](std::string name, std::vector<sextant::Color> colors, bool cyclic) {
              nb::gil_scoped_release nogil; // process-wide, under sextant's own lock
              return sextant::register_colormap(name, colors, cyclic);
          },
          "name"_a, "colors"_a, "cyclic"_a = false,
          "Add a colormap for the rest of the process, usable by name wherever cmap= is\n"
          "taken (with '_r' for reversed): `colors` spread evenly from vmin to vmax,\n"
          "interpolated linearly, alpha included. cyclic=True wraps values outside\n"
          "[vmin, vmax] instead of clamping (give it equal first and last colors). Returns\n"
          "the name. ValueError for fewer than two colors, an empty name, one ending in\n"
          "'_r', or one already taken (any case). A registered map cannot be removed.");
    m.def("colormaps", [] { return sextant::Colormap::names(); },
          "Every colormap's name, without '_r': the built-ins, then the registered ones in\n"
          "registration order.");

    // Tests and tools/gen_stubs.py: every option struct's fields and their types.
    m.def("_option_fields", &sextant_py::option_fields);
    // sextant/_enums.py (alias lookup) and tests: the enum name tables.
    m.def("_enum_tables", &sextant_py::enum_tables);
}
