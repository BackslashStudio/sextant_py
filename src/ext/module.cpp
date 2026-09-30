// The _sextant extension module: the C++ API mapped 1:1. Conversions that
// make it pleasant from Python belong in the sextant package's Python files,
// not here.
//
// Threading: every call into a figure's object graph goes through locked()
// (state.h); the waits run lock-free with the GIL released.
#include "state.h"

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

    sextant_py::bind_axes(m);
    sextant_py::bind_figure(m);
    sextant_py::bind_messages(m);
    sextant_py::bind_lifecycle(m);
}
