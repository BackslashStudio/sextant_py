#include "options.h"

#include <cctype>
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

    bool doubles_from_python(nb::handle h, std::vector<double>& out) {
        if (h.is_none()) {
            out.clear();
            return true;
        }
        if (PyUnicode_Check(h.ptr()) || PyBytes_Check(h.ptr())) return false;
        try {
            nb::object np = nb::module_::import_("numpy");
            // ascontiguousarray: C order, and a scalar comes back as one value.
            nb::object a = np.attr("ascontiguousarray")(h, nb::arg("dtype") = np.attr("float64"));
            Vec v;
            if (nb::cast<int>(a.attr("ndim")) != 1 || !nb::try_cast(a, v)) return false;
            out.assign(v.span().begin(), v.span().end());
            return true;
        } catch (const nb::python_error&) {
            return false;
        }
    }

    bool bools_from_python(nb::handle h, std::vector<bool>& out) {
        if (h.is_none()) {
            out.clear();
            return true;
        }
        if (PyUnicode_Check(h.ptr()) || PyBytes_Check(h.ptr())) return false;
        try {
            nb::object np = nb::module_::import_("numpy");
            nb::object a = np.attr("ascontiguousarray")(h);
            // Booleans, or numbers read as numpy reads them (0 is False); not strings.
            const std::string kind = nb::cast<std::string>(a.attr("dtype").attr("kind"));
            if (kind != "b" && kind != "i" && kind != "u" && kind != "f") return false;
            if (nb::cast<int>(a.attr("ndim")) != 1) return false;
            a = np.attr("ascontiguousarray")(a, nb::arg("dtype") = np.attr("bool_"));
            nb::ndarray<const bool, nb::ndim<1>, nb::c_contig, nb::device::cpu> v;
            if (!nb::try_cast(a, v)) return false;
            out.assign(v.data(), v.data() + v.size());
            return true;
        } catch (const nb::python_error&) {
            return false;
        }
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

    const std::vector<sextant::Colormap>& builtin_colormaps() {
        using C = sextant::Colormap;
        static const std::vector<C> all{
            C::Viridis, C::Plasma, C::Inferno, C::Magma, C::Cividis, C::Turbo, C::Coolwarm, C::Gray,
            C::Blues, C::Greens, C::Reds, C::Oranges, C::Purples, C::Greys, C::YlOrRd, C::YlGnBu,
            C::Hot, C::Jet, C::RdBu, C::Spectral, C::BrBG, C::PiYG,
            C::Twilight, C::TwilightShifted, C::Hsv, C::Phase};
        return all;
    }

    std::optional<sextant::Colormap> colormap_from_python(nb::handle h) noexcept {
        PyObject* o = h.ptr();
        PyObject* name = nullptr; // new reference when not `o` itself
        if (!PyUnicode_Check(o)) {
            // A matplotlib Colormap (or anything else carrying its name).
            name = PyObject_GetAttrString(o, "name");
            if (!name || !PyUnicode_Check(name)) {
                Py_XDECREF(name);
                PyErr_Clear();
                return std::nullopt;
            }
            o = name;
        }
        Py_ssize_t n = 0;
        const char* s = PyUnicode_AsUTF8AndSize(o, &n);
        std::string text = s ? std::string(s, static_cast<std::size_t>(n)) : std::string();
        Py_XDECREF(name);
        if (!s) {
            PyErr_Clear();
            return std::nullopt;
        }
        // matplotlib's spelling of gray, which sextant does not take.
        std::string lower = text;
        for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (lower == "grey") text = "gray";
        else if (lower == "grey_r") text = "gray_r";
        try {
            return sextant::Colormap::from_name(text);
        } catch (const std::exception&) {
            return std::nullopt;
        }
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

        // Colormap is a class, not an enum: its table holds the built-ins (the
        // _enums members) and is open, since any registered name is taken too.
        void add_colormaps(nb::dict& d) {
            nb::list names;
            for (const auto& c : builtin_colormaps()) names.append(c.name());
            nb::dict aliases;
            aliases["grey"] = "gray";
            nb::dict entry;
            entry["names"] = names;
            entry["aliases"] = aliases;
            entry["open"] = true;
            d["Colormap"] = entry;
        }
    } // namespace

    nb::dict enum_tables() {
        nb::dict d;
        add_enum<sextant::LineStyle>(d);
        add_enum<sextant::MarkerStyle>(d);
        add_enum<sextant::DrawStyle>(d);
        add_enum<sextant::Orientation>(d);
        add_enum<sextant::StackBaseline>(d);
        add_enum<sextant::QuiverPivot>(d);
        add_colormaps(d);
        add_enum<sextant::CapStyle>(d);
        add_enum<sextant::AxisPosition>(d);
        add_enum<sextant::Scale>(d);
        add_enum<sextant::Adjustable>(d);
        add_enum<sextant::LegendAnchor>(d);
        add_enum<sextant::ColorbarAnchor>(d);
        add_enum<sextant::HAlign>(d);
        add_enum<sextant::VAlign>(d);
        add_enum<sextant::Coords>(d);
        add_enum<sextant::AxLineMode>(d);
        add_enum<sextant::ArrowHead>(d);
        add_enum<sextant::PanelTheme>(d);
        add_enum<sextant::MathFontset>(d);
        add_enum<sextant::Projection>(d);
        add_enum<sextant::PlaneOrientation>(d);
        add_enum<sextant::EventKind>(d);
        add_enum<sextant::PickKind>(d);
        add_enum<sextant::EventConsumed>(d);
        return d;
    }

    namespace {
        template <class T>
        void add_fields(nb::dict& d) {
            nb::dict fields;
            for (const auto& f : Fields<T>::list()) fields[f.name] = f.type();
            d[Fields<T>::name] = fields;
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
        add_fields<sextant::AxLineOptions>(d);
        add_fields<sextant::AxSpanOptions>(d);
        add_fields<sextant::FillOptions>(d);
        add_fields<sextant::PolyOptions>(d);
        add_fields<sextant::CircleOptions>(d);
        add_fields<sextant::WedgeOptions>(d);
        add_fields<sextant::StackOptions>(d);
        add_fields<sextant::QuiverOptions>(d);
        add_fields<sextant::HeatmapOptions>(d);
        add_fields<sextant::GridOptions>(d);
        add_fields<sextant::ScaleOptions>(d);
        add_fields<sextant::AxesStyle>(d);
        add_fields<sextant::LegendOptions>(d);
        add_fields<sextant::ColorbarOptions>(d);
        add_fields<sextant::SuptitleOptions>(d);
        add_fields<sextant::TextOptions>(d);
        add_fields<sextant::ArrowOptions>(d);
        add_fields<sextant::PngExportOptions>(d);
        add_fields<sextant::SvgExportOptions>(d);
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
