#include "wrappers.h"

#include <nanobind/ndarray.h>

#include <span>

namespace sextant_py {
    namespace {
        // Zero-copy when the caller passes C-contiguous float64 (sextant copies at
        // ingest, so the array only has to outlive the call); nanobind converts
        // anything else it can.
        using Array1D = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

        std::span<const double> as_span(const Array1D& a) { return {a.data(), a.shape(0)}; }

        using Self = nb::pointer_and_handle<PyAxes>;

        // Run a graph call on the axes and return the same Python object, for
        // chaining.
        template <class F>
        nb::object chain(Self self, F&& f) {
            PyAxes& a = *self.p;
            locked(*a.st, [&] { f(*a.ax); });
            return nb::borrow(self.h);
        }
    } // namespace

    void bind_axes(nb::module_& m) {
        using sextant::Axes;

        nb::class_<PyAxes>(m, "Axes", nb::is_weak_referenceable())
            .def("line",
                 [](Self self, const Array1D& x, const Array1D& y) {
                     return chain(self, [&](Axes& ax) { ax.line(as_span(x), as_span(y)); });
                 },
                 "x"_a, "y"_a)
            .def("line",
                 [](Self self, const Array1D& y) {
                     return chain(self, [&](Axes& ax) { ax.line(as_span(y)); });
                 },
                 "y"_a)
            .def("line_count", [](PyAxes& a) {
                return locked(*a.st, [&] { return a.ax->line_count(); });
            });
    }
} // namespace sextant_py
