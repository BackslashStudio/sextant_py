// Events: Figure.connect()/disconnect()/dispatch_events() and the Event class.
//
// sextant calls a callback on whichever thread drains the figure's queue --
// inside wait_closed(), poll_events(), run() or dispatch_events(), which the
// binding always enters with the GIL released -- holding no lock of its own.
// The thunk takes the GIL, so a Python callback runs like any Python code on
// that thread (the main thread, in a script).
#include "options.h"
#include "wrappers.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include <cmath>
#include <string>

namespace sextant_py {
    // The Python callable, shared by sextant's std::function and the PyFigure
    // that owns it. `fn` is reset (GIL held) when the PyFigure is cleared by
    // the GC or destroyed; the handler in sextant then does nothing until it
    // is disconnected. sextant may drop the handler on any thread
    // (disconnect(), ~Figure); the binding never holds the GIL while it calls
    // into sextant, so taking it here cannot deadlock.
    struct PyCallable {
        nb::object fn;
        explicit PyCallable(nb::object f) : fn(std::move(f)) {}
        PyCallable(const PyCallable&) = delete;
        PyCallable& operator=(const PyCallable&) = delete;
        ~PyCallable() {
            if (!fn.is_valid()) return;
            nb::gil_scoped_acquire gil;
            nb::object dead = std::move(fn); // released while the GIL is held
        }
    };

    PyFigure::~PyFigure() {
        if (callbacks.empty() || !st) return;
        std::vector<int> ids;
        for (auto& c : callbacks) {
            ids.push_back(c.id);
            std::erase(st->connections, c.id);
            nb::object dead = std::move(c.cb->fn);
        }
        callbacks.clear();
        if (shutting_down()) return; // the exit sweep disconnected everything
        nb::gil_scoped_release nogil;
        for (int id : ids) st->fig->disconnect(id);
    }

    int figure_traverse(PyObject* self, visitproc visit, void* arg) {
        Py_VISIT(Py_TYPE(self)); // a heap type's instances refer to it
        if (!nb::inst_ready(self)) return 0;
        for (const auto& c : nb::inst_ptr<PyFigure>(self)->callbacks)
            if (c.cb->fn.is_valid()) Py_VISIT(c.cb->fn.ptr());
        return 0;
    }

    int figure_clear(PyObject* self) {
        // Only drop the references: no call into sextant from inside the GC.
        // The handlers are disconnected when the object is destroyed.
        for (auto& c : nb::inst_ptr<PyFigure>(self)->callbacks) {
            nb::object dead = std::move(c.cb->fn);
        }
        return 0;
    }

    namespace {
        // A copy of the event (sextant's is valid during the call only), and the
        // figure it came from, for inaxes.
        struct PyEvent {
            sextant::Event e;
            std::weak_ptr<FigureState> st;
        };

        void deliver(const PyCallable& cb, const std::weak_ptr<FigureState>& weak, const sextant::Event& e) {
            if (shutting_down()) return;
            nb::gil_scoped_acquire gil;
            if (!cb.fn.is_valid()) return; // its Figure object is gone
            std::shared_ptr<FigureState> keep = weak.lock();
            try {
                // A reference of our own: the callback may clear its figure.
                nb::object fn = cb.fn;
                fn(nb::cast(PyEvent{e, weak}));
            } catch (nb::python_error& err) {
                // Nowhere to raise it: report it (as matplotlib does) and go on.
                // Ctrl+C is re-raised by the wait that delivered the event.
                if (err.matches(PyExc_KeyboardInterrupt)) note_interrupt();
                else err.discard_as_unraisable(cb.fn.is_valid() ? nb::handle(cb.fn) : nb::handle(Py_None));
            }
            // The callback may have dropped the last reference to its own figure.
            if (keep && keep.use_count() == 1) defer_release(std::move(keep));
        }

        // The Axes/Axes3D whose frame is under the event, as the same Python
        // object the program holds. The event names the subplot by its first
        // cell; add_subplot() returns the one there, and throws if it is 3D.
        nb::typed<nb::object, std::variant<PyAxes, PyAxes3D, nb::none>> inaxes(const PyEvent& ev) {
            if (ev.e.axes < 0) return nb::none();
            auto st = ev.st.lock();
            if (!st) return nb::none();
            const int cell = ev.e.axes;
            try {
                auto ax = locked(*st, [&] { return st->fig->add_subplot(cell); });
                return wrap<PyAxes>(st, std::move(ax));
            } catch (const std::exception&) {
            }
            try {
                auto ax = locked(*st, [&] { return st->fig->add_subplot3d(cell); });
                return wrap<PyAxes3D>(st, std::move(ax));
            } catch (const std::exception&) {
            }
            return nb::none();
        }

        nb::tuple modifiers(int mods) {
            nb::list out;
            if (mods & sextant::kModCtrl) out.append("ctrl");
            if (mods & sextant::kModShift) out.append("shift");
            if (mods & sextant::kModAlt) out.append("alt");
            if (mods & sextant::kModSuper) out.append("super");
            return nb::tuple(out);
        }

        std::string num(double v) {
            if (std::isnan(v)) return "nan";
            std::string s = std::to_string(v);
            s.erase(s.find_last_not_of('0') + 1);
            if (!s.empty() && s.back() == '.') s.pop_back();
            return s;
        }

        std::string repr(const PyEvent& ev) {
            using K = sextant::EventKind;
            const sextant::Event& e = ev.e;
            std::string out = "Event(kind='" + std::string(enum_to_string(e.kind)) + "'";
            const bool pointer = e.kind == K::MouseDown || e.kind == K::MouseUp || e.kind == K::MouseMove ||
                                 e.kind == K::Scroll || e.kind == K::Pick;
            if (pointer) {
                out += ", x=" + num(e.x) + ", y=" + num(e.y) + ", axes=" + std::to_string(e.axes);
                if (e.has_data)
                    out += ", xdata=" + num(e.xdata) + ", ydata=" + num(e.ydata) +
                           (std::isnan(e.zdata) ? "" : ", zdata=" + num(e.zdata)) +
                           (std::isnan(e.x2data) ? "" : ", x2data=" + num(e.x2data)) +
                           (std::isnan(e.y2data) ? "" : ", y2data=" + num(e.y2data));
            }
            if (e.kind == K::MouseDown || e.kind == K::MouseUp || e.kind == K::Pick)
                out += ", button=" + std::to_string(e.button) + (e.double_click ? ", double_click=True" : "");
            if (e.kind == K::Scroll) out += ", scroll_x=" + num(e.scroll_x) + ", scroll_y=" + num(e.scroll_y);
            if (e.kind == K::KeyDown || e.kind == K::KeyUp) out += ", key='" + e.key + "'";
            if (e.kind == K::Resize) out += ", width=" + std::to_string(e.width) + ", height=" + std::to_string(e.height);
            if (e.kind == K::Pick) {
                out += ", pick_kind='" + std::string(enum_to_string(e.pick_kind)) + "', pick_object=" +
                       std::to_string(e.pick_object) + ", pick_index=" + std::to_string(e.pick_index);
                if (e.pick_row >= 0) out += ", pick_row=" + std::to_string(e.pick_row) + ", pick_col=" + std::to_string(e.pick_col);
                if (e.pick_plane != 0) out += ", pick_plane=" + std::to_string(e.pick_plane);
                if (e.pick_twin != 0) out += ", pick_twin=" + std::to_string(e.pick_twin);
            }
            if (e.consumed != sextant::EventConsumed::None)
                out += ", consumed='" + std::string(enum_to_string(e.consumed)) + "'";
            return out + ")";
        }
    } // namespace

    void bind_events(nb::module_& m, nb::class_<PyFigure>& fig) {
        using sextant::Event;

        nb::class_<PyEvent>(m, "Event",
                            "One event, as a callback connected with Figure.connect() receives it. Every kind\n"
                            "has every field; the ones its kind does not fill keep their defaults (NaN for the\n"
                            "data coordinates, -1 for indices). Pixel coordinates are the figure's: logical\n"
                            "pixels from the top left of the plot area.")
            .def_prop_ro("kind", [](const PyEvent& ev) { return ev.e.kind; })
            .def_prop_ro("mods", [](const PyEvent& ev) { return ev.e.mods; },
                         "Modifier bits: 1 ctrl, 2 shift, 4 alt, 8 super (see modifiers).")
            .def_prop_ro("modifiers", [](const PyEvent& ev) { return modifiers(ev.e.mods); },
                         "The held modifiers by name: ('ctrl', 'shift', 'alt', 'super') in that order.")
            .def_prop_ro("button", [](const PyEvent& ev) { return ev.e.button; },
                         "mouse_down/mouse_up/pick: 0 left, 1 right, 2 middle.")
            .def_prop_ro("double_click", [](const PyEvent& ev) { return ev.e.double_click; })
            .def_prop_ro("x", [](const PyEvent& ev) { return ev.e.x; })
            .def_prop_ro("y", [](const PyEvent& ev) { return ev.e.y; })
            .def_prop_ro("axes", [](const PyEvent& ev) { return ev.e.axes; },
                         "The subplot under (x, y), as the index add_subplot() takes (a span: its first\n"
                         "cell), or -1. inaxes is the object itself.")
            .def_prop_ro("inaxes", &inaxes, "The Axes or Axes3D under (x, y), or None.")
            .def_prop_ro("has_data", [](const PyEvent& ev) { return ev.e.has_data; },
                         "Whether xdata/ydata(/zdata) are set: over a 2D axes, or where a 3D cursor ray\n"
                         "meets a Plane2D.")
            .def_prop_ro("xdata", [](const PyEvent& ev) { return ev.e.xdata; })
            .def_prop_ro("ydata", [](const PyEvent& ev) { return ev.e.ydata; })
            .def_prop_ro("zdata", [](const PyEvent& ev) { return ev.e.zdata; })
            .def_prop_ro("x2data", [](const PyEvent& ev) { return ev.e.x2data; },
                         "Over a 2D axes with a twiny(): the twin's x under (x, y); NaN without one.\n"
                         "xdata/ydata stay the host's (inaxes).")
            .def_prop_ro("y2data", [](const PyEvent& ev) { return ev.e.y2data; },
                         "Over a 2D axes with a twinx(): the twin's y under (x, y); NaN without one.")
            .def_prop_ro("scroll_x", [](const PyEvent& ev) { return ev.e.scroll_x; })
            .def_prop_ro("scroll_y", [](const PyEvent& ev) { return ev.e.scroll_y; },
                         "Wheel notches, positive away from the user; queued ones are summed.")
            .def_prop_ro("consumed", [](const PyEvent& ev) { return ev.e.consumed; },
                         "What sextant itself did with the input (informational; nothing is suppressed).")
            .def_prop_ro("key", [](const PyEvent& ev) { return ev.e.key; },
                         "key_down/key_up: 'a', 'A', 'ctrl+a', 'escape', 'f5', 'left', ...")
            .def_prop_ro("width", [](const PyEvent& ev) { return ev.e.width; })
            .def_prop_ro("height", [](const PyEvent& ev) { return ev.e.height; })
            .def_prop_ro("pick_kind", [](const PyEvent& ev) { return ev.e.pick_kind; })
            .def_prop_ro("pick_object", [](const PyEvent& ev) { return ev.e.pick_object; },
                         "The picked object's id (0 = none), as Axes.last_object() returned it.\n"
                         "Axes.find_object(id) (Plane2D.find_object(id) when pick_plane != 0) gives its\n"
                         "kind and index, the i of line_data(i), bar3d_data(i), ...; None if the object\n"
                         "was removed since.")
            .def_prop_ro("pick_index", [](const PyEvent& ev) { return ev.e.pick_index; },
                         "The element: point, bar, marker or vertex index; a surface sample; a heatmap's\n"
                         "or bar3d's flat cell index; a fill's segment i, between its points i and i + 1.\n"
                         "A bar, fill or heatmap is picked where the press is inside it (see xdata/ydata).")
            .def_prop_ro("pick_row", [](const PyEvent& ev) { return ev.e.pick_row; })
            .def_prop_ro("pick_col", [](const PyEvent& ev) { return ev.e.pick_col; })
            .def_prop_ro("pick_plane", [](const PyEvent& ev) { return ev.e.pick_plane; },
                         "The id of the plane the picked object is on (Plane2D.id(); Axes3D.plane_index(id) gives its\n"
                         "plane_at() index), or 0 for an object of the axes itself.")
            .def_prop_ro("pick_twin", [](const PyEvent& ev) { return ev.e.pick_twin; },
                         "Whose object pick_object indexes in a 2D cell with twins: 0 inaxes' own,\n"
                         "1 inaxes.twinx()'s, 2 inaxes.twiny()'s.")
            .def("__repr__", &repr);

        fig.def("connect",
                [](PyFigure& self, sextant::EventKind kind, nb::callable fn) {
                    auto cb = std::make_shared<PyCallable>(std::move(fn));
                    std::weak_ptr<FigureState> weak = self.st;
                    sextant::EventCallback thunk = [cb, weak](const Event& e) { deliver(*cb, weak, e); };
                    int id;
                    {
                        nb::gil_scoped_release nogil;
                        id = self.st->fig->connect(kind, std::move(thunk));
                    }
                    self.st->connections.push_back(id);
                    self.callbacks.push_back({id, std::move(cb)});
                    return id;
                },
                "kind"_a, "callback"_a,
                "Call callback(event) for events of `kind` (an EventKind, its name, or matplotlib's:\n"
                "'button_press_event', 'pick_event', ...). Returns an id for disconnect().\n"
                "Callbacks run on the thread that delivers events -- during show(), wait_closed(),\n"
                "run(), poll_events() or dispatch_events(), or at an interactive prompt -- so a\n"
                "script that never waits sees none. An exception is reported through\n"
                "sys.unraisablehook and delivery goes on; KeyboardInterrupt is raised by the wait.")
            .def("disconnect",
                 [](PyFigure& self, int id) {
                     std::erase(self.st->connections, id);
                     std::shared_ptr<PyCallable> cb;
                     for (auto it = self.callbacks.begin(); it != self.callbacks.end(); ++it)
                         if (it->id == id) {
                             cb = std::move(it->cb);
                             self.callbacks.erase(it);
                             break;
                         }
                     {
                         nb::gil_scoped_release nogil;
                         self.st->fig->disconnect(id);
                     }
                     // `cb`, if sextant's copy is gone too, ends here with the GIL held.
                 },
                 "id"_a, "Stop a callback, including for events already queued. Unknown ids are ignored.")
            .def("dispatch_events",
                 [](PyFigure& self) {
                     int n;
                     {
                         nb::gil_scoped_release nogil;
                         n = self.st->fig->dispatch_events();
                     }
                     drain_deferred();
                     check_signals();
                     return n;
                 },
                 "Deliver this figure's queued events now; returns how many.");
    }
} // namespace sextant_py
