// sextant's option structs as keyword arguments: one field table per struct,
// and apply() that sets fields from a kwargs/dict by name. A nested struct
// (LineOptions::errorbar, FigureOptions::margins) takes a dict.
//
// Every field of a struct must be listed here: a field added to a sextant
// header is a binding edit (tests/test_options.py sets every listed name).
#pragma once

#include "casters.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <vector>

namespace sextant_py {
    template <class T>
    struct Field {
        const char* name;
        // Sets the field from `v`; `ctx` ("line()") and `path` ("errorbar.capsize")
        // are for the message when `v` does not fit.
        void (*set)(T& o, nb::handle v, const std::string& ctx, const std::string& path);
        // The Python annotation of what `set` takes, for the type stubs.
        std::string (*type)();
    };

    template <class T>
    struct Fields; // specialised per struct: `name`, `list()`

    template <class T>
    constexpr bool has_fields = requires { Fields<T>::list(); };

    // A caster name's input side: nanobind writes io_name(in, out) as "@in@out@".
    inline std::string input_side(const char* text) {
        std::string out;
        for (const char* p = text; *p; ++p) {
            if (*p != '@') {
                out += *p;
                continue;
            }
            while (*++p && *p != '@') out += *p;     // in
            while (*p && *++p && *p != '@') {}       // out
        }
        return out;
    }

    // What a field of type F takes, as a Python annotation. A nested struct is
    // the TypedDict of its name, which the stub generator writes.
    template <class F>
    std::string annotation() {
        if constexpr (has_fields<F>) return Fields<F>::name;
        else return input_side(nb::detail::make_caster<F>::Name.text);
    }

    template <class T>
    std::string field_names() {
        std::string out;
        for (const auto& f : Fields<T>::list()) out += (out.empty() ? "" : ", ") + std::string(f.name);
        return out;
    }

    // What a value for a field of type F looks like, for error messages.
    template <class F>
    std::string expected() {
        if constexpr (is_named_enum<F>) return "one of " + enum_choices<F>();
        else if constexpr (std::is_same_v<F, sextant::Color> || std::is_same_v<F, std::optional<sextant::Color>>)
            return kColorHint;
        else return input_side(nb::detail::make_caster<F>::Name.text);
    }

    template <class T>
    void apply(T& o, nb::handle mapping, const std::string& ctx, const std::string& prefix = "");

    template <class F>
    void assign(F& dst, nb::handle v, const std::string& ctx, const std::string& path) {
        if constexpr (has_fields<F>) {
            if (!PyDict_Check(v.ptr()))
                throw nb::type_error((ctx + ": '" + path + "' takes a dict of " + Fields<F>::name +
                                      " fields: " + field_names<F>()).c_str());
            apply(dst, v, ctx, path + ".");
        } else {
            try {
                dst = nb::cast<F>(v);
            } catch (const nb::cast_error&) {
                throw nb::type_error((ctx + ": bad value for '" + path + "': " +
                                      nb::cast<std::string>(nb::repr(v)) + " (expected " +
                                      expected<F>() + ")").c_str());
            }
        }
    }

    template <class T>
    void apply(T& o, nb::handle mapping, const std::string& ctx, const std::string& prefix) {
        for (auto [k, v] : nb::borrow<nb::dict>(mapping)) {
            const std::string key = nb::cast<std::string>(k);
            const Field<T>* hit = nullptr;
            for (const auto& f : Fields<T>::list())
                if (key == f.name) hit = &f;
            if (!hit) {
                const std::string where = prefix.empty()
                    ? ctx + " got an unexpected keyword argument '" + key + "'"
                    : ctx + ": '" + prefix.substr(0, prefix.size() - 1) + "' has no field '" + key + "'";
                throw nb::type_error((where + " (" + Fields<T>::name + " fields: " +
                                      field_names<T>() + ")").c_str());
            }
            hit->set(o, v, ctx, prefix + key);
        }
    }

    // A T from `kwargs` over `init` (T's defaults unless given).
    template <class T>
    T options(const std::string& ctx, const nb::kwargs& kwargs, T init = {}) {
        apply(init, kwargs, ctx);
        return init;
    }

#define SEXTANT_PY_FIELD(f)                                                          \
    Field<Self>{#f, [](Self& o, nb::handle v, const std::string& ctx, const std::string& path) { \
        assign(o.f, v, ctx, path);                                                   \
    }, [] { return annotation<decltype(Self::f)>(); }}
#define SEXTANT_PY_OPTIONS(T, ...)                                                   \
    template <>                                                                      \
    struct Fields<sextant::T> {                                                      \
        using Self = sextant::T;                                                     \
        static constexpr const char* name = #T;                                      \
        static const std::vector<Field<Self>>& list() {                              \
            static const std::vector<Field<Self>> l{__VA_ARGS__};                    \
            return l;                                                                \
        }                                                                            \
    };
#define F SEXTANT_PY_FIELD

    // Nested ones first: their Fields must exist where a parent's assign() is
    // instantiated.
    SEXTANT_PY_OPTIONS(ErrorBarOptions,
                       F(color), F(linewidth), F(capsize), F(capstyle), F(boxwidth), F(box_alpha))
    SEXTANT_PY_OPTIONS(FigureMargins, F(left), F(right), F(top), F(bottom))
    SEXTANT_PY_OPTIONS(ErrorBar3DOptions,
                       F(color), F(linewidth), F(capsize), F(capstyle), F(boxwidth), F(box_alpha),
                       F(edge_alpha))

    SEXTANT_PY_OPTIONS(LineOptions,
                       F(color), F(linewidth), F(linestyle), F(name), F(show_legend), F(alpha),
                       F(loop), F(errorbar), F(hint_labels))
    SEXTANT_PY_OPTIONS(ScatterOptions,
                       F(color), F(size), F(marker), F(name), F(show_legend), F(alpha),
                       F(edgecolor), F(edge_alpha), F(edge_linewidth), F(errorbar), F(hint_labels))
    SEXTANT_PY_OPTIONS(ScatterZOptions,
                       F(cmap), F(size), F(marker), F(alpha), F(edgecolor), F(edge_alpha), F(edge_linewidth),
                       F(vmin), F(vmax), F(colorbar), F(name), F(show_legend), F(errorbar), F(hint_labels))
    SEXTANT_PY_OPTIONS(BarOptions,
                       F(color), F(width), F(alpha), F(name), F(show_legend), F(edgecolor),
                       F(linewidth), F(errorbar), F(hint_labels))
    SEXTANT_PY_OPTIONS(HeatmapOptions,
                       F(cmap), F(vmin), F(vmax), F(colorbar), F(name), F(origin), F(contours),
                       F(contour_color), F(contour_linewidth), F(contour_labels),
                       F(contour_fontsize), F(hint_labels))
    SEXTANT_PY_OPTIONS(GridOptions, F(color), F(linestyle), F(linewidth))
    SEXTANT_PY_OPTIONS(AxesStyle,
                       F(background), F(spine_color), F(spine_linewidth), F(spine_bottom), F(spine_left),
                       F(spine_top), F(spine_right), F(xaxis_y), F(xaxis_z), F(yaxis_x),
                       F(yaxis_z), F(zaxis_x), F(zaxis_y), F(origin_x), F(origin_y), F(origin_z),
                       F(frame_margin), F(show_xticks), F(show_yticks), F(show_zticks), F(tick_color),
                       F(tick_length), F(tick_linewidth),
                       F(label_color), F(label_fontsize), F(title_color), F(title_fontsize),
                       F(xtitle_color), F(xtitle_fontsize), F(ytitle_color), F(ytitle_fontsize),
                       F(ztitle_color), F(ztitle_fontsize), F(font_path))
    SEXTANT_PY_OPTIONS(LegendOptions,
                       F(anchor), F(margin), F(offset_x), F(offset_y), F(fontsize), F(frameon),
                       F(text_color), F(frame_color), F(border_color), F(border_linewidth),
                       F(font_path))
    SEXTANT_PY_OPTIONS(ColorbarOptions,
                       F(anchor), F(width), F(margin), F(offset_x), F(offset_y), F(fontsize),
                       F(text_color), F(border_color), F(border_linewidth), F(font_path))
    SEXTANT_PY_OPTIONS(SuptitleOptions,
                       F(fontsize), F(color), F(font_path), F(align), F(offset_x), F(offset_y))
    SEXTANT_PY_OPTIONS(PngExportOptions, F(peel_layers), F(dpi))
    SEXTANT_PY_OPTIONS(SvgExportOptions, F(max_splits), F(max_tests))
    SEXTANT_PY_OPTIONS(FigureOptions,
                       F(width), F(height), F(title), F(resizable), F(dpi), F(subplot_col_gap),
                       F(subplot_row_gap), F(margins), F(background), F(panel_width), F(supersample), F(vsync),
                       F(theme))

    // 3D
    SEXTANT_PY_OPTIONS(Plane2DOptions, F(alpha), F(visible))
    SEXTANT_PY_OPTIONS(Bar3DOptions,
                       F(color), F(alpha), F(width), F(depth), F(bottom), F(shading), F(edges),
                       F(edgecolor), F(edge_alpha), F(edge_linewidth), F(name), F(show_legend),
                       F(hint_labels))
    SEXTANT_PY_OPTIONS(SurfaceOptions,
                       F(color), F(colormap), F(cmap), F(vmin), F(vmax), F(colorbar), F(name),
                       F(show_legend), F(alpha), F(shading), F(edges), F(edgecolor), F(edge_alpha),
                       F(edge_linewidth), F(hint_labels))
    SEXTANT_PY_OPTIONS(SurfaceTriOptions,
                       F(color), F(cmap), F(vmin), F(vmax), F(colorbar), F(name), F(show_legend),
                       F(alpha), F(shading), F(edges), F(edgecolor), F(edge_alpha), F(edge_linewidth),
                       F(hint_labels))
    SEXTANT_PY_OPTIONS(Scatter3DOptions,
                       F(color), F(size), F(marker), F(alpha), F(edgecolor), F(edge_alpha), F(edge_linewidth),
                       F(depthshade), F(cmap), F(vmin), F(vmax), F(colorbar), F(name), F(show_legend), F(errorbar),
                       F(hint_labels))
    SEXTANT_PY_OPTIONS(Line3DOptions,
                       F(color), F(linewidth), F(alpha), F(loop), F(depthshade), F(cmap), F(vmin),
                       F(vmax), F(colorbar), F(name), F(show_legend), F(errorbar), F(hint_labels))
    SEXTANT_PY_OPTIONS(Box3DStyle, F(panes), F(pane_color), F(pane_edge_color), F(margin))
    SEXTANT_PY_OPTIONS(Camera3D, F(azimuth), F(elevation), F(target), F(zoom), F(projection), F(fov))
#undef F
#undef SEXTANT_PY_OPTIONS
#undef SEXTANT_PY_FIELD

    // For tests and tools/gen_stubs.py: {struct: {field: annotation}}, in
    // declaration order.
    nb::dict option_fields();
} // namespace sextant_py
