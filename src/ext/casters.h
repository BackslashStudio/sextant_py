// How sextant's value types look from Python: arrays are anything numpy can
// make float64 of, enums are strings, a Color is a name, "#rrggbb[aa]" or an
// (r, g, b[, a]) tuple, a Range or a SubplotSpan a 2-tuple.
#pragma once

#include <sextant/sextant.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sextant_py {
    namespace nb = nanobind;

    // An N-dimensional float64 array argument. Zero-copy for a C-contiguous
    // float64 ndarray (sextant copies at ingest, so it only has to outlive the
    // call); anything else numpy.asarray() turns into one (lists, other dtypes,
    // strided views), which this then owns.
    template <class Inner>
    struct ArrayOf {
        Inner a;
        std::span<const double> span() const { return {a.data(), a.size()}; }
        std::size_t shape(std::size_t i) const { return a.shape(i); }
        std::size_t ndim() const { return a.ndim(); }
    };
    template <int N>
    using Array = ArrayOf<nb::ndarray<const double, nb::ndim<N>, nb::c_contig, nb::device::cpu>>;
    using Vec = Array<1>;
    using Mat = Array<2>;
    // Any number of dimensions, read flat (a bar3d/surface `heights` grid).
    using Grid = ArrayOf<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>;

    // A name for the type stubs only: NumpyObject converts as a plain object
    // (anything in, the object out) and reads "ArrayLike" as an argument,
    // "numpy.typing.NDArray[numpy.<T>]" as a result; OrNone adds "| None" to the
    // result (an argument that takes None says so with .none()).
    template <class T, bool OrNone = false>
    struct NumpyOf {};
    template <class T, bool OrNone = false>
    using NumpyObject = nb::typed<nb::object, NumpyOf<T, OrNone>>;

    // Also stubs only: nb::typed<nb::object, Named<"...">> is an object the
    // stubs call "...".
    template <std::size_t N>
    struct StubName {
        char text[N];
        constexpr StubName(const char (&s)[N]) { std::copy_n(s, N, text); }
    };
    template <StubName S>
    struct Named {};

    // A new numpy array that owns `v` (moved, not copied).
    template <class T>
    NumpyObject<T> to_numpy(std::vector<T>&& v, std::initializer_list<std::size_t> shape) {
        auto* heap = new std::vector<T>(std::move(v));
        nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
        return nb::cast(nb::ndarray<nb::numpy, T>(heap->data(), shape, owner));
    }

    // --- enums as strings ----------------------------------------------------

    // Per enum: `py_name`, its class in sextant/_enums.py; `names`, one
    // canonical spelling per value (the Python member's value); `aliases`,
    // matched verbatim (matplotlib's "--", "o", ...).
    template <class E>
    struct EnumNames;

    template <class E>
    using NameList = std::vector<std::pair<std::string_view, E>>;

    // Lower case without '_', '-' or ' ': "inside_tl", "InsideTL" and
    // "inside-tl" are one name.
    inline std::string fold(std::string_view s) {
        std::string out;
        for (char c : s)
            if (c != '_' && c != '-' && c != ' ')
                out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return out;
    }

    template <class E>
    std::optional<E> enum_from_string(std::string_view s) {
        for (const auto& [alias, v] : EnumNames<E>::aliases)
            if (alias == s) return v;
        const std::string f = fold(s);
        for (const auto& [name, v] : EnumNames<E>::names)
            if (fold(name) == f) return v;
        return std::nullopt;
    }

    template <class E>
    std::string_view enum_to_string(E e) {
        for (const auto& [name, v] : EnumNames<E>::names)
            if (v == e) return name;
        return "?";
    }

    template <class E>
    std::string enum_choices() {
        std::string out;
        for (const auto& [name, v] : EnumNames<E>::names)
            out += (out.empty() ? "'" : ", '") + std::string(name) + "'";
        for (const auto& [alias, v] : EnumNames<E>::aliases)
            out += ", '" + std::string(alias) + "'";
        return out;
    }

    template <class E>
    constexpr bool is_named_enum = requires { EnumNames<E>::names; };

#define SEXTANT_PY_UNPAREN(...) __VA_ARGS__
#define SEXTANT_PY_ENUM(E, NAMES, ALIASES)                                        \
    template <>                                                                   \
    struct EnumNames<sextant::E> {                                                \
        using enum sextant::E;                                                    \
        static constexpr const char* py_name = #E;                                \
        static inline const NameList<sextant::E> names{SEXTANT_PY_UNPAREN NAMES}; \
        static inline const NameList<sextant::E> aliases{SEXTANT_PY_UNPAREN ALIASES}; \
    };

    SEXTANT_PY_ENUM(LineStyle,
                    ({"solid", Solid}, {"dashed", Dashed}, {"dotted", Dotted},
                     {"dashdot", DashDot}, {"none", None}),
                    ({"-", Solid}, {"--", Dashed}, {":", Dotted}, {"-.", DashDot}, {"", None}))
    SEXTANT_PY_ENUM(MarkerStyle,
                    ({"none", None}, {"circle", Circle}, {"square", Square},
                     {"triangle", Triangle}, {"cross", Cross}, {"plus", Plus},
                     {"diamond", Diamond}),
                    ({"o", Circle}, {"s", Square}, {"^", Triangle}, {"x", Cross},
                     {"+", Plus}, {"D", Diamond}, {"d", Diamond}, {"", None}))
    SEXTANT_PY_ENUM(Colormap,
                    ({"viridis", Viridis}, {"plasma", Plasma}, {"inferno", Inferno},
                     {"magma", Magma}, {"cividis", Cividis}, {"turbo", Turbo},
                     {"coolwarm", Coolwarm}, {"gray", Gray}),
                    ({"grey", Gray}))
    SEXTANT_PY_ENUM(CapStyle, ({"flat", Flat}, {"arrow", Arrow}), ())
    SEXTANT_PY_ENUM(AxisPosition,
                    ({"auto", Auto}, {"low", Low}, {"mid", Mid}, {"high", High}), ())
    SEXTANT_PY_ENUM(LegendAnchor,
                    ({"inside_tl", InsideTL}, {"inside_tr", InsideTR},
                     {"inside_bl", InsideBL}, {"inside_br", InsideBR},
                     {"outside_tl", OutsideTL}, {"outside_tr", OutsideTR},
                     {"outside_bl", OutsideBL}, {"outside_br", OutsideBR},
                     {"outside_lt", OutsideLT}, {"outside_lb", OutsideLB},
                     {"outside_rt", OutsideRT}, {"outside_rb", OutsideRB}),
                    ())
    SEXTANT_PY_ENUM(ColorbarAnchor,
                    ({"left", Left}, {"right", Right}, {"top", Top}, {"bottom", Bottom}), ())
    SEXTANT_PY_ENUM(HAlign,
                    ({"left", Left}, {"center", Center}, {"right", Right}), ({"centre", Center}))
    SEXTANT_PY_ENUM(VAlign,
                    ({"top", Top}, {"center", Center}, {"baseline", Baseline}, {"bottom", Bottom}),
                    ({"centre", Center}))
    // Alias: matplotlib's xycoords name for a frame fraction.
    SEXTANT_PY_ENUM(Coords, ({"data", Data}, {"fraction", Fraction}), ({"axes fraction", Fraction}))
    SEXTANT_PY_ENUM(ArrowHead, ({"none", None}, {"open", Open}, {"filled", Filled}, {"bar", Bar}), ())
    SEXTANT_PY_ENUM(PanelTheme,
                    ({"dark", Dark}, {"light", Light}, {"classic", Classic}), ())
    SEXTANT_PY_ENUM(Projection,
                    ({"orthographic", Orthographic}, {"perspective", Perspective}),
                    ({"ortho", Orthographic}, {"persp", Perspective}))
    SEXTANT_PY_ENUM(PlaneOrientation, ({"xy", XY}, {"yz", YZ}, {"zx", ZX}), ())
    // Aliases: matplotlib's mpl_connect() names.
    SEXTANT_PY_ENUM(EventKind,
                    ({"close", Close}, {"mouse_down", MouseDown}, {"mouse_up", MouseUp},
                     {"mouse_move", MouseMove}, {"scroll", Scroll}, {"key_down", KeyDown},
                     {"key_up", KeyUp}, {"resize", Resize}, {"pick", Pick}),
                    ({"close_event", Close}, {"button_press_event", MouseDown},
                     {"button_release_event", MouseUp}, {"motion_notify_event", MouseMove},
                     {"scroll_event", Scroll}, {"key_press_event", KeyDown},
                     {"key_release_event", KeyUp}, {"resize_event", Resize}, {"pick_event", Pick}))
    SEXTANT_PY_ENUM(PickKind,
                    ({"none", None}, {"line", Line}, {"scatter", Scatter}, {"scatter_z", ScatterZ},
                     {"bar", Bar}, {"heatmap", Heatmap}, {"bar3d", Bar3D}, {"surface", Surface},
                     {"surface_tri", SurfaceTri}, {"scatter3d", Scatter3D}, {"line3d", Line3D}),
                    ())
    SEXTANT_PY_ENUM(EventConsumed,
                    ({"none", None}, {"select", Select}, {"navigate", Navigate}, {"grid_drag", GridDrag}),
                    ())
#undef SEXTANT_PY_ENUM
#undef SEXTANT_PY_UNPAREN

    // "#rrggbb", "#rrggbbaa", a name (sextant's), or 3/4 floats in [0, 1].
    std::optional<sextant::Color> color_from_python(nb::handle h) noexcept;
    constexpr const char* kColorHint =
        "a color: a name (red, blue, green, orange, purple, cyan, black, white, gray), "
        "'#rrggbb', '#rrggbbaa' or an (r, g, b[, a]) tuple of floats in [0, 1]";

    // A tuple or list of exactly `n` numbers.
    bool numbers_from_python(nb::handle h, std::size_t n, double* out) noexcept;

    // The member of sextant._enums.<cls> whose value is `value` (a new
    // reference); the plain string if that class cannot be had.
    PyObject* enum_member(const char* cls, std::string_view value) noexcept;

    // For _enums.py and tests: {class name: {"names": [...], "aliases": {...}}}.
    nb::dict enum_tables();
} // namespace sextant_py

namespace nanobind::detail {
    template <class Inner>
    struct type_caster<sextant_py::ArrayOf<Inner>> {
        NB_TYPE_CASTER(sextant_py::ArrayOf<Inner>, const_name("ArrayLike"))

        bool from_python(handle src, uint32_t flags, cleanup_list* cleanup) noexcept {
            make_caster<Inner> c;
            if (c.from_python(src, flags, cleanup)) {
                value.a = c.value;
                return true;
            }
            if (!(flags & static_cast<uint32_t>(cast_flags::convert))) return false;
            try {
                object np = module_::import_("numpy");
                object arr = np.attr("asarray")(src, arg("dtype") = np.attr("float64"));
                if (!c.from_python(arr, flags, cleanup)) return false;
                value.a = c.value; // holds a reference to arr
                return true;
            } catch (python_error&) {
                return false; // not array-like: nanobind reports the mismatch
            }
        }
    };

    // Name only: typed<object, NumpyOf<...>> converts through object's caster.
    template <class T, bool OrNone>
    struct type_caster<sextant_py::NumpyOf<T, OrNone>> {
        static constexpr auto Dtype = const_name<std::is_same_v<T, double>>(
            const_name("float64"),
            const_name<std::is_same_v<T, std::uint8_t>>(const_name("uint8"), const_name("uint32")));
        static constexpr auto None = const_name<OrNone>(const_name(" | None"), const_name(""));
        static constexpr auto Name = const_name('@') + const_name("ArrayLike") + const_name('@') +
                                     const_name("numpy.typing.NDArray[numpy.") + Dtype + const_name("]") +
                                     None + const_name('@');
    };

    template <sextant_py::StubName S>
    struct type_caster<sextant_py::Named<S>> {
        static constexpr auto Name = const_name(S.text);
    };

    // Full specializations below: nanobind's own enum caster is a partial one.
    template <class E>
    struct sextant_enum_caster {
        NB_TYPE_CASTER(E, const_name("str"))

        bool from_python(handle src, uint32_t, cleanup_list*) noexcept {
            if (!PyUnicode_Check(src.ptr())) return false;
            Py_ssize_t n = 0;
            const char* s = PyUnicode_AsUTF8AndSize(src.ptr(), &n);
            if (!s) {
                PyErr_Clear();
                return false;
            }
            auto v = sextant_py::enum_from_string<E>({s, static_cast<std::size_t>(n)});
            if (!v) return false;
            value = *v;
            return true;
        }

        static handle from_cpp(E v, rv_policy, cleanup_list*) noexcept {
            return sextant_py::enum_member(sextant_py::EnumNames<E>::py_name, sextant_py::enum_to_string(v));
        }
    };

    // Named for the stubs: `LineStyleLike` in (the member or any listed string,
    // an alias the stub generator writes), the _enums class out.
#define SEXTANT_PY_ENUM_CASTER(E)                                               \
    template <> struct type_caster<sextant::E> : sextant_enum_caster<sextant::E> { \
        static constexpr auto Name = io_name(#E "Like", "sextant._enums." #E);   \
    };
    SEXTANT_PY_ENUM_CASTER(LineStyle)
    SEXTANT_PY_ENUM_CASTER(MarkerStyle)
    SEXTANT_PY_ENUM_CASTER(Colormap)
    SEXTANT_PY_ENUM_CASTER(CapStyle)
    SEXTANT_PY_ENUM_CASTER(AxisPosition)
    SEXTANT_PY_ENUM_CASTER(LegendAnchor)
    SEXTANT_PY_ENUM_CASTER(ColorbarAnchor)
    SEXTANT_PY_ENUM_CASTER(HAlign)
    SEXTANT_PY_ENUM_CASTER(VAlign)
    SEXTANT_PY_ENUM_CASTER(Coords)
    SEXTANT_PY_ENUM_CASTER(ArrowHead)
    SEXTANT_PY_ENUM_CASTER(PanelTheme)
    SEXTANT_PY_ENUM_CASTER(Projection)
    SEXTANT_PY_ENUM_CASTER(PlaneOrientation)
    SEXTANT_PY_ENUM_CASTER(EventKind)
    SEXTANT_PY_ENUM_CASTER(PickKind)
    SEXTANT_PY_ENUM_CASTER(EventConsumed)
#undef SEXTANT_PY_ENUM_CASTER

    // Vec3 and BoxAspect: (x, y, z).
    template <class T>
    struct xyz_caster {
        NB_TYPE_CASTER(T, const_name("tuple[float, float, float]"))

        bool from_python(handle src, uint32_t, cleanup_list*) noexcept {
            double v[3];
            if (!sextant_py::numbers_from_python(src, 3, v)) return false;
            value = {v[0], v[1], v[2]};
            return true;
        }

        static handle from_cpp(const T& t, rv_policy, cleanup_list*) noexcept {
            return make_tuple(t.x, t.y, t.z).release();
        }
    };
    template <> struct type_caster<sextant::Vec3> : xyz_caster<sextant::Vec3> {};
    template <> struct type_caster<sextant::BoxAspect> : xyz_caster<sextant::BoxAspect> {};

    template <>
    struct type_caster<sextant::Color> {
        NB_TYPE_CASTER(sextant::Color, const_name("str | tuple[float, ...]"))

        bool from_python(handle src, uint32_t, cleanup_list*) noexcept {
            auto c = sextant_py::color_from_python(src);
            if (!c) return false;
            value = *c;
            return true;
        }

        static handle from_cpp(const sextant::Color& c, rv_policy, cleanup_list*) noexcept {
            return make_tuple(c.r, c.g, c.b, c.a).release();
        }
    };

    template <>
    struct type_caster<sextant::Range> {
        NB_TYPE_CASTER(sextant::Range, const_name("tuple[float, float]"))

        bool from_python(handle src, uint32_t, cleanup_list*) noexcept {
            double v[2];
            if (!sextant_py::numbers_from_python(src, 2, v)) return false;
            value = {v[0], v[1]};
            return true;
        }

        static handle from_cpp(const sextant::Range& r, rv_policy, cleanup_list*) noexcept {
            return make_tuple(r.lo, r.hi).release();
        }
    };

    template <>
    struct type_caster<sextant::SubplotSpan> {
        NB_TYPE_CASTER(sextant::SubplotSpan, const_name("tuple[int, int]"))

        bool from_python(handle src, uint32_t, cleanup_list*) noexcept {
            if (!PyTuple_Check(src.ptr()) || PyTuple_Size(src.ptr()) != 2) return false;
            const long a = PyLong_AsLong(PyTuple_GetItem(src.ptr(), 0));
            const long b = PyLong_AsLong(PyTuple_GetItem(src.ptr(), 1));
            if (PyErr_Occurred()) {
                PyErr_Clear();
                return false;
            }
            value = {static_cast<int>(a), static_cast<int>(b)};
            return true;
        }

        static handle from_cpp(const sextant::SubplotSpan& s, rv_policy, cleanup_list*) noexcept {
            return make_tuple(s.first, s.last).release();
        }
    };

    template <>
    struct type_caster<sextant::FigureSize> {
        NB_TYPE_CASTER(sextant::FigureSize, const_name("tuple[int, int]"))

        bool from_python(handle, uint32_t, cleanup_list*) noexcept { return false; }

        static handle from_cpp(const sextant::FigureSize& s, rv_policy, cleanup_list*) noexcept {
            return make_tuple(s.width, s.height).release();
        }
    };
} // namespace nanobind::detail
