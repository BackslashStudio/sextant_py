// Figure registry, the exit sweep, the process-wide event loop functions and
// the REPL input hook.
#include "state.h"

#include <algorithm>
#include <atomic>
#include <chrono>

#if defined(_WIN32)
#include <conio.h>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/select.h>
#endif

namespace sextant_py {
    namespace {
        // GIL-protected.
        std::vector<std::weak_ptr<FigureState>> g_figures;
        std::vector<std::shared_ptr<FigureState>> g_deferred;
        // Shown figures, kept while their windows are open (keep_shown()): the
        // Python object, which owns the callbacks, and its state.
        struct Shown {
            nb::object fig;
            std::shared_ptr<FigureState> st;
        };
        std::vector<Shown> g_shown;
        bool g_interrupt = false;

        std::atomic<bool> g_shutdown{false};

        // How long a blocking wait runs with the GIL released before it checks
        // for Ctrl+C.
        constexpr double kSlice = 0.1;

        // Whether a line (or EOF) is waiting on stdin, waiting up to `ms`.
        // Anything that cannot be waited on counts as ready, so the prompt is
        // never held up.
        bool stdin_ready(int ms) {
#if defined(_WIN32)
            HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
            if (h == nullptr || h == INVALID_HANDLE_VALUE || GetFileType(h) != FILE_TYPE_CHAR) return true;
            if (_kbhit()) return true;
            Sleep(static_cast<DWORD>(ms));
            return _kbhit() != 0;
#else
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(0, &fds);
            timeval tv{0, ms * 1000};
            return select(1, &fds, nullptr, nullptr, &tv) != 0;
#endif
        }

        // PyOS_InputHook: the REPL calls it on the main thread, without the GIL,
        // while it waits for a line (some REPLs once per line, readline many
        // times). It runs until a line is typed, pumping windows -- on macOS
        // what keeps them responsive -- and delivering events, and uninstalls
        // itself once no figure is open.
        int input_hook() {
            for (;;) {
                bool any_open = false;
                {
                    nb::gil_scoped_acquire gil;
                    if (!g_shutdown)
                        for (auto& st : live_figures())
                            if (st->fig->is_open()) any_open = true;
                    // Ctrl+C at the prompt: let the REPL see it.
                    if (PyOS_InterruptOccurred()) {
                        PyErr_SetInterrupt();
                        return 0;
                    }
                }
                try {
                    sextant::Figure::poll_events();
                } catch (...) {
                }
                {
                    nb::gil_scoped_acquire gil;
                    drain_deferred();
                    if (!any_open) {
                        uninstall_input_hook();
                        return 0;
                    }
                }
                if (stdin_ready(10)) return 0;
            }
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

    void defer_release(std::shared_ptr<FigureState> st) { g_deferred.push_back(std::move(st)); }

    void keep_shown(nb::handle fig, std::shared_ptr<FigureState> st) {
        for (const auto& s : g_shown)
            if (s.st == st) return;
        g_shown.push_back({nb::borrow(fig), std::move(st)});
    }

    void forget_shown(const FigureState& st) {
        // Moved out first: the last reference may go, and with it a wrapper
        // whose destruction runs Python.
        std::vector<Shown> gone;
        for (auto it = g_shown.begin(); it != g_shown.end();) {
            if (it->st.get() == &st) {
                gone.push_back(std::move(*it));
                it = g_shown.erase(it);
            } else {
                ++it;
            }
        }
    }

    void drain_deferred() {
        // The shown figures whose windows have closed go, once their last
        // events (the Close) are delivered.
        std::vector<Shown> closed;
        for (auto it = g_shown.begin(); it != g_shown.end();) {
            if (it->st->fig->is_open()) {
                ++it;
            } else {
                closed.push_back(std::move(*it));
                it = g_shown.erase(it);
            }
        }
        if (!closed.empty() && !g_shutdown) {
            nb::gil_scoped_release nogil;
            for (auto& s : closed) s.st->fig->dispatch_events();
        }
        closed.clear();
        // Swapped out first: releasing one may run Python (a wrapper's weakref
        // callback) that defers another.
        while (!g_deferred.empty()) {
            std::vector<std::shared_ptr<FigureState>> batch;
            batch.swap(g_deferred);
        }
    }

    void note_interrupt() { g_interrupt = true; }

    void check_signals() {
        if (g_interrupt) {
            g_interrupt = false;
            PyErr_SetNone(PyExc_KeyboardInterrupt);
            throw nb::python_error(); // takes the error just set
        }
        if (PyErr_CheckSignals() != 0) throw nb::python_error();
    }

    bool install_input_hook() {
        if (PyOS_InputHook == &input_hook) return true;
        if (PyOS_InputHook != nullptr) return false; // someone else's (tkinter's): leave it
        PyOS_InputHook = &input_hook;
        return true;
    }

    void uninstall_input_hook() {
        if (PyOS_InputHook == &input_hook) PyOS_InputHook = nullptr;
    }

    void bind_lifecycle(nb::module_& m) {
        m.def("poll_events",
              [] {
                  {
                      nb::gil_scoped_release nogil;
                      sextant::Figure::poll_events();
                  }
                  drain_deferred();
                  check_signals();
              },
              "Pump window events once and deliver every figure's queued events on this thread.\n"
              "On macOS this is the pump and must run on the main thread (RuntimeError elsewhere).");

        // Figure::run() cannot be interrupted, so wait in slices instead,
        // sharing each among the open figures so every one's events flow.
        m.def("run",
              [] {
                  for (;;) {
                      std::vector<std::shared_ptr<FigureState>> open;
                      auto live = live_figures();
                      for (auto& st : live)
                          if (st->fig->is_open()) open.push_back(st);
                      if (open.empty()) {
                          // The last window may have closed after its slice: its
                          // Close event is still queued.
                          {
                              nb::gil_scoped_release nogil;
                              for (auto& st : live) st->fig->dispatch_events();
                          }
                          live.clear();
                          drain_deferred();
                          check_signals();
                          return;
                      }
                      live.clear();
                      const double slice = kSlice / static_cast<double>(open.size());
                      for (auto& st : open) {
                          nb::gil_scoped_release nogil;
                          st->fig->wait_closed(slice);
                      }
                      open.clear();
                      drain_deferred();
                      check_signals();
                  }
              },
              "Block until every open figure has closed, delivering their events.");

        m.def("_any_open", [] {
            for (auto& st : live_figures())
                if (st->fig->is_open()) return true;
            return false;
        });
        m.def("_install_input_hook", &install_input_hook);
        m.def("_uninstall_input_hook", &uninstall_input_hook);
        m.def("_input_hook_installed", [] { return PyOS_InputHook == &input_hook; });

        // Registered with atexit by the package: close every window while the
        // interpreter is still whole, take every Python callable out of
        // sextant, and stop routing diagnostics to Python.
        m.def("_shutdown", [] {
            if (g_shutdown) return;
            uninstall_input_hook();
            for (auto& st : live_figures()) {
                locked(*st, [&] { st->fig->close(); });
                std::vector<int> ids;
                ids.swap(st->connections);
                nb::gil_scoped_release nogil;
                for (int id : ids) st->fig->disconnect(id);
            }
            g_shutdown = true;
            sextant::Figure::set_message_handler({});
            std::vector<Shown> gone;
            gone.swap(g_shown);
            gone.clear();
            drain_deferred();
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
            drain_deferred();
            check_signals();
            if (closed) return true;
            if (timeout >= 0 && slice < kSlice) return false;
        }
    }
} // namespace sextant_py
