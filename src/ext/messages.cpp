// sextant's diagnostics as Python warnings, or a Python handler.
#include "state.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <cstdio>
#include <string>
#include <thread>

namespace sextant_py {
    namespace {
        // handler(message: str), for the stubs; None is taken and returned too.
        using HandlerFn = nb::typed<nb::callable, void(std::string)>;
        using Handler = nb::typed<nb::object, HandlerFn>;
        using OptionalHandler = nb::typed<nb::object, std::optional<HandlerFn>>;

        // The user's handler; None = warnings.warn. Heap-allocated and never
        // freed, so no Python object is released after the interpreter is gone.
        // GIL-protected.
        nb::object& user_handler() {
            static auto* h = new nb::object(nb::none());
            return *h;
        }

        // Installed in sextant at import. Called on any thread, with no sextant
        // lock held.
        void deliver(std::string_view msg) {
            if (shutting_down()) {
                std::fprintf(stderr, "sextant: %.*s\n", static_cast<int>(msg.size()), msg.data());
                return;
            }
            nb::gil_scoped_acquire gil;
            try {
                nb::str text(msg.data(), msg.size());
                nb::object& h = user_handler();
                if (!h.is_none())
                    h(text);
                else
                    nb::module_::import_("warnings").attr("warn")(text, nb::handle(PyExc_RuntimeWarning));
            } catch (nb::python_error& e) {
                // Nowhere to raise it: it may be a window thread.
                e.discard_as_unraisable("sextant message handler");
            }
        }
    } // namespace

    void bind_messages(nb::module_& m) {
        sextant::Figure::set_message_handler(deliver);

        m.def("set_message_handler",
              [](Handler handler) -> OptionalHandler {
                  if (!handler.is_none() && !PyCallable_Check(handler.ptr()))
                      throw nb::type_error("handler must be callable or None");
                  nb::object prev = user_handler();
                  user_handler() = std::move(handler);
                  return prev;
              },
              nb::arg("handler").none(),
              "Route sextant's diagnostics to handler(message: str); None restores the default,\n"
              "warnings.warn(message, RuntimeWarning). Returns the previous handler.");

        // Tests only: call the handler sextant holds, as sextant would, on this
        // thread or a new one.
        m.def("_emit_message", [](std::string msg, bool other_thread) {
            sextant::MessageHandler h = sextant::Figure::set_message_handler({});
            sextant::Figure::set_message_handler(h);
            if (!h) return;
            nb::gil_scoped_release nogil;
            if (other_thread)
                std::thread([&] { h(msg); }).join();
            else
                h(msg);
        });
    }
} // namespace sextant_py
