// The C++ types behind the Python classes. Each holds its figure's state, so
// any wrapper keeps the figure (and its window) alive, as in matplotlib.
#pragma once

#include "state.h"

namespace sextant_py {
    using namespace nb::literals;

    struct PyFigure {
        std::shared_ptr<FigureState> st;
    };

    struct PyAxes {
        std::shared_ptr<FigureState> st;
        std::shared_ptr<sextant::Axes> ax;
    };
} // namespace sextant_py
