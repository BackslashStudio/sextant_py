#include "options.h"

#include <cmath>
#include <exception>

namespace sextant_py {
    bool numbers_from_python(nb::handle h, std::size_t n, double* out) noexcept {
        PyObject* o = h.ptr();
        if (!PyTuple_Check(o) && !PyList_Check(o)) return false;
        if (static_cast<std::size_t>(PySequence_Size(o)) != n) return false;
        for (std::size_t i = 0; i < n; ++i) {
            PyObject* item = PySequence_GetItem(o, static_cast<Py_ssize_t>(i));
            if (!item) {
                PyErr_Clear();
                return false;
            }
            const bool number = PyFloat_Check(item) || PyLong_Check(item);
            out[i] = number ? PyFloat_AsDouble(item) : 0.0;
            Py_DECREF(item);
            if (!number || PyErr_Occurred()) {
                PyErr_Clear();
                return false;
            }
        }
        return true;
    }

    std::optional<sextant::Color> color_from_python(nb::handle h) noexcept {
        if (PyUnicode_Check(h.ptr())) {
            Py_ssize_t n = 0;
            const char* s = PyUnicode_AsUTF8AndSize(h.ptr(), &n);
            if (!s) {
                PyErr_Clear();
                return std::nullopt;
            }
            try {
                return sextant::Color::from_name({s, static_cast<std::size_t>(n)});
            } catch (const std::exception&) {
                return std::nullopt;
            }
        }
        double v[4] = {0, 0, 0, 1};
        if (!numbers_from_python(h, 4, v) && !numbers_from_python(h, 3, v)) return std::nullopt;
        for (double c : v)
            if (!std::isfinite(c)) return std::nullopt;
        return sextant::Color{static_cast<float>(v[0]), static_cast<float>(v[1]),
                              static_cast<float>(v[2]), static_cast<float>(v[3])};
    }

    PyObject* enum_member(const char* cls, std::string_view value) noexcept {
        PyObject* str = PyUnicode_FromStringAndSize(value.data(), static_cast<Py_ssize_t>(value.size()));
        if (!str) return nullptr;
        // Only while sextant/__init__.py is still importing can this fail; the
        // plain string then stands in, equal to the member.
        PyObject* mod = PyImport_ImportModule("sextant._enums");
        PyObject* type = mod ? PyObject_GetAttrString(mod, cls) : nullptr;
        Py_XDECREF(mod);
        PyObject* member = type ? PyObject_CallFunctionObjArgs(type, str, nullptr) : nullptr;
        Py_XDECREF(type);
        if (!member) {
            PyErr_Clear();
            return str;
        }
        Py_DECREF(str);
        return member;
    }

    namespace {
        template <class E>
        void add_enum(nb::dict& d) {
            nb::list names;
            for (const auto& [name, v] : EnumNames<E>::names) names.append(nb::str(name.data(), name.size()));
            nb::dict aliases;
            for (const auto& [alias, v] : EnumNames<E>::aliases) {
                const std::string_view canonical = enum_to_string(v);
                aliases[nb::str(alias.data(), alias.size())] = nb::str(canonical.data(), canonical.size());
            }
            nb::dict entry;
            entry["names"] = names;
            entry["aliases"] = aliases;
            d[EnumNames<E>::py_name] = entry;
        }
    } // namespace

    nb::dict enum_tables() {
        nb::dict d;
        add_enum<sextant::LineStyle>(d);
        add_enum<sextant::MarkerStyle>(d);
        add_enum<sextant::Colormap>(d);
        add_enum<sextant::CapStyle>(d);
        add_enum<sextant::AxisPosition>(d);
        add_enum<sextant::LegendAnchor>(d);
        add_enum<sextant::ColorbarAnchor>(d);
        add_enum<sextant::HAlign>(d);
        add_enum<sextant::PanelTheme>(d);
        add_enum<sextant::Projection>(d);
        add_enum<sextant::PlaneOrientation>(d);
        return d;
    }

    namespace {
        template <class T>
        void add_fields(nb::dict& d) {
            nb::list names;
            for (const auto& f : Fields<T>::list()) names.append(f.name);
            d[Fields<T>::name] = names;
        }
    } // namespace

    nb::dict option_fields() {
        nb::dict d;
        add_fields<sextant::ErrorBarOptions>(d);
        add_fields<sextant::FigureMargins>(d);
        add_fields<sextant::LineOptions>(d);
        add_fields<sextant::ScatterOptions>(d);
        add_fields<sextant::ScatterZOptions>(d);
        add_fields<sextant::BarOptions>(d);
        add_fields<sextant::HeatmapOptions>(d);
        add_fields<sextant::GridOptions>(d);
        add_fields<sextant::AxesStyle>(d);
        add_fields<sextant::LegendOptions>(d);
        add_fields<sextant::ColorbarOptions>(d);
        add_fields<sextant::SuptitleOptions>(d);
        add_fields<sextant::PngExportOptions>(d);
        add_fields<sextant::FigureOptions>(d);
        add_fields<sextant::ErrorBar3DOptions>(d);
        add_fields<sextant::Plane2DOptions>(d);
        add_fields<sextant::Bar3DOptions>(d);
        add_fields<sextant::SurfaceOptions>(d);
        add_fields<sextant::SurfaceTriOptions>(d);
        add_fields<sextant::Scatter3DOptions>(d);
        add_fields<sextant::Line3DOptions>(d);
        add_fields<sextant::Box3DStyle>(d);
        add_fields<sextant::Camera3D>(d);
        return d;
    }
} // namespace sextant_py
