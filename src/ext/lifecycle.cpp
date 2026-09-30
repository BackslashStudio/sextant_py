// Figure registry, the exit sweep, and the process-wide event loop functions.
#include "state.h"

#include <algorithm>
#include <atomic>
#include <chrono>

namespace sextant_py {
    namespace {
        // GIL-protected.
        std::vector<std::weak_ptr<FigureState>> g_figures;
        std::atomic<bool> g_shutdown{false};

        // How long a blocking wait runs with the GIL released before it checks
        // for Ctrl+C.
        constexpr double kSlice = 0.1;

        void check_signals() {
            if (PyErr_CheckSignals() != 0) throw nb::python_error();
        }
    } // namespace

    FigureState::~FigureState() {
        wrappers.clear();
        if (!fig) return;
        if (g_shutdown) {
            // Closed by the sweep, so there is no thread left to join; and Python
            // may be finalizing, when giving up the GIL is best avoided.
            fig.reset();
            return;
        }
        // ~Figure closes the window and joins its thread.
        nb::gil_scoped_release nogil;
        std::lock_guard lock(graph);
        fig.reset();
    }

    std::shared_ptr<FigureState> register_figure(std::shared_ptr<sextant::Figure> fig) {
        std::erase_if(g_figures, [](const auto& w) { return w.expired(); });
        auto st = std::make_shared<FigureState>(std::move(fig));
        g_figures.push_back(st);
        return st;
    }

    std::vector<std::shared_ptr<FigureState>> live_figures() {
        std::vector<std::shared_ptr<FigureState>> out;
        for (const auto& w : g_figures)
            if (auto st = w.lock()) out.push_back(std::move(st));
        return out;
    }

    bool shutting_down() { return g_shutdown; }

    void bind_lifecycle(nb::module_& m) {
        m.def("poll_events", [] {
            nb::gil_scoped_release nogil;
            sextant::Figure::poll_events();
        }, "Pump window events once and deliver queued events on this thread.");

        // Figure::run() cannot be interrupted, so wait in slices instead.
        m.def("run", [] {
            for (;;) {
                std::shared_ptr<FigureState> open;
                for (auto& st : live_figures())
                    if (st->fig->is_open()) { open = std::move(st); break; }
                if (!open) return;
                {
                    nb::gil_scoped_release nogil;
                    open->fig->wait_closed(kSlice);
                }
                check_signals();
            }
        }, "Block until every open figure has closed.");

        // Registered with atexit by the package: close every window while the
        // interpreter is still whole, and stop routing diagnostics to Python.
        m.def("_shutdown", [] {
            if (g_shutdown) return;
            for (auto& st : live_figures())
                locked(*st, [&] { st->fig->close(); });
            g_shutdown = true;
            sextant::Figure::set_message_handler({});
        });
    }

    bool wait_closed(FigureState& st, double timeout) {
        using clock = std::chrono::steady_clock;
        const auto start = clock::now();
        for (;;) {
            double slice = kSlice;
            if (timeout >= 0) {
                const double left =
                    timeout - std::chrono::duration<double>(clock::now() - start).count();
                slice = left > 0 ? std::min(left, kSlice) : 0.0;
            }
            bool closed;
            {
                nb::gil_scoped_release nogil;
                closed = st.fig->wait_closed(slice);
            }
            if (closed) return true;
            check_signals();
            if (timeout >= 0 && slice < kSlice) return false;
        }
    }
} // namespace sextant_py
