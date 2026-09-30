// Per-figure state shared by the Python wrappers, and the locking rule every
// binding follows.
#pragma once

#include <sextant/sextant.h>

#include <nanobind/nanobind.h>

#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sextant_py {
    namespace nb = nanobind;

    // One per sextant::Figure. A figure and every axes it hands out form one
    // object graph that sextant leaves the caller to serialise; `graph` is that
    // exclusion, taken around every call into the graph.
    //
    // Only Python objects (the Figure and Axes wrappers) own a FigureState, so
    // the last owner always goes away with the GIL held -- which the destructor
    // relies on, since the stable ABI has no way to ask. Code that needs one
    // from another thread must hold a weak_ptr.
    struct FigureState {
        explicit FigureState(std::shared_ptr<sextant::Figure> f) : fig(std::move(f)) {}
        ~FigureState();
        FigureState(const FigureState&) = delete;
        FigureState& operator=(const FigureState&) = delete;

        std::mutex graph;
        // Reset only by the destructor, so the lock-free calls (is_open(),
        // wait_closed()) may read it without `graph`.
        std::shared_ptr<sextant::Figure> fig;
        // The Python wrapper handed out for each axes, so fig.axes() is
        // fig.axes(). Keyed by the C++ object; GIL-protected.
        std::unordered_map<const void*, nb::weakref> wrappers;
    };

    // f() with the GIL released and the graph lock held -- in that order: a
    // thread waiting for the lock must not hold the GIL, or it deadlocks
    // against a lock holder that needs the GIL (a message handler).
    template <class F>
    decltype(auto) locked(FigureState& st, F&& f) {
        nb::gil_scoped_release nogil;
        std::lock_guard lock(st.graph);
        return std::forward<F>(f)();
    }

    // A new FigureState, recorded for run() and the exit sweep. GIL held.
    std::shared_ptr<FigureState> register_figure(std::shared_ptr<sextant::Figure> fig);

    // Every figure whose wrappers are still alive. GIL held.
    std::vector<std::shared_ptr<FigureState>> live_figures();

    // Set by the exit sweep (_shutdown), after which Python may be finalizing.
    bool shutting_down();

    // Wait for st's window to close, `timeout` seconds at most (negative:
    // forever), with the GIL released in short slices and Ctrl+C checked
    // between them. Lock-free. GIL held on entry.
    bool wait_closed(FigureState& st, double timeout);

    void bind_figure(nb::module_& m);
    void bind_axes(nb::module_& m);
    void bind_messages(nb::module_& m);
    void bind_lifecycle(nb::module_& m);

    // Python wrapper for `ax`, the one already handed out if it is alive.
    template <class Wrapper, class T>
    nb::object wrap(const std::shared_ptr<FigureState>& st, std::shared_ptr<T> obj) {
        const void* key = obj.get();
        if (auto it = st->wrappers.find(key); it != st->wrappers.end()) {
            nb::object alive = it->second();
            if (!alive.is_none()) return alive;
        }
        nb::object o = nb::cast(Wrapper{st, std::move(obj)}, nb::rv_policy::move);
        st->wrappers.insert_or_assign(key, nb::weakref(o));
        return o;
    }
} // namespace sextant_py
