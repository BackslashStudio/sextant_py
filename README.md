# sextant for Python

Python bindings for [sextant](https://github.com/BackslashStudio/sextant), a C++20 library for scientific
plotting: 2D and 3D figures in an interactive OpenGL window, and PNG/SVG export with or without a display.
Two ways in: the core API, which maps sextant's C++ API name for name, and `sextant.pyplot`, which runs
matplotlib-style code.

## Install

```sh
python -m pip install pysextant
```

On PyPI it is **`pysextant`** (`sextant` there is an unrelated project); it imports as `sextant`. The current
release is 1.1.0. Wheels cover Python 3.10 or later on Windows (x86-64), Linux (x86-64, glibc 2.27+) and macOS 11+ (Apple Silicon and Intel);
the only dependency is numpy. **[doc/install.md](https://github.com/BackslashStudio/sextant_py/blob/v1.1.0/doc/install.md)** has everything else: building
[from source](https://github.com/BackslashStudio/sextant_py/blob/v1.1.0/doc/install.md#from-source), [conda](https://github.com/BackslashStudio/sextant_py/blob/v1.1.0/doc/install.md#in-a-conda-environment),
[what each platform needs at run time](https://github.com/BackslashStudio/sextant_py/blob/v1.1.0/doc/install.md#running-it) (X11/EGL on Linux, the main thread on macOS,
OpenGL 3.3 on Windows) and [troubleshooting](https://github.com/BackslashStudio/sextant_py/blob/v1.1.0/doc/install.md#troubleshooting).

## Documentation

[doc/README.md](https://github.com/BackslashStudio/sextant_py/blob/v1.1.0/doc/README.md) indexes it: the [core API guide](https://github.com/BackslashStudio/sextant_py/blob/v1.1.0/doc/api.md), [3D plots](https://github.com/BackslashStudio/sextant_py/blob/v1.1.0/doc/3d.md),
[`sextant.pyplot`](https://github.com/BackslashStudio/sextant_py/blob/v1.1.0/doc/pyplot.md) and the [option reference](https://github.com/BackslashStudio/sextant_py/blob/v1.1.0/doc/reference.md).

## Examples

The core API: keyword arguments are the C++ option structs' fields, enums are strings, and plotting calls
chain.

```python
import numpy as np
import sextant

x = np.linspace(0, 2 * np.pi, 200)
fig = sextant.Figure(width=800, height=500, title="waves")
ax = fig.axes()
ax.line(x, np.sin(x), color="blue", name="sin").line(x, np.cos(x), color="red", linestyle="--", name="cos")
ax.set_title("sin and cos").legend()
fig.savefig("waves.png")   # needs no window
fig.show()                 # in a script, blocks until the window is closed
```

matplotlib-style code through `sextant.pyplot`:

```python
import numpy as np
import sextant.pyplot as plt

x = np.linspace(0, 10, 100)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, np.sin(x), "r-", label="sin")
ax.scatter(x[::10], np.cos(x[::10]), label="cos, sampled")
ax.set_xlabel("x")
ax.legend()
plt.savefig("mpl.png")
plt.show()
```

`pyplot` covers the common calls (`plot`, `scatter`, `bar`, `hist`, `errorbar`, `imshow`, `pcolormesh`, 3D
`plot_surface`/`scatter`/`bar3d`, `text`/`annotate`, `$...$` math in labels, colorbars, `mpl_connect` events,
`pause`/`ion`); what sextant cannot draw
raises an error rather than being ignored. In Jupyter a figure displays inline as PNG.

Windows are interactive (pan, zoom, a panel to edit titles, ticks and data). In a script `show()` blocks until
the window closes; in a REPL or IPython it returns and the window stays live.

The package is typed (`py.typed`): keyword options and enum names are checked by a type checker (tested with mypy).

## Development

Building needs the toolchain in [doc/install.md](https://github.com/BackslashStudio/sextant_py/blob/v1.1.0/doc/install.md#1-the-c-toolchain).

```sh
git clone --recursive https://github.com/BackslashStudio/sextant_py.git
cd sextant_py
uv sync
uv run pytest
```

The install is editable: after a change to the C++ sources, the next `import sextant` rebuilds the extension.
On Windows any shell works (Visual Studio with the C++ workload is found by itself).

- sextant comes from the `extern/sextant` submodule, pinned to a commit. After a `git pull` that moves the
  pin, run `git submodule update` (or pull with `--recurse-submodules`).
- To build against another sextant checkout, set `SEXTANT_SOURCE_DIR` to its absolute path and run
  `uv sync --reinstall-package pysextant`.
- After changing a signature, an option field or an enum name, regenerate the type stub with
  `uv run python tools/gen_stubs.py`; `tests/test_stubs.py` fails until it matches.
- Release wheels are built by `cibuildwheel` with the settings in `pyproject.toml`, locally the same as in CI
  (`.github/workflows/wheels.yml`).

## License

MIT. The extension links sextant and the code it bundles (FreeType, libpng, zlib, GLFW, Dear ImGui, NanoVG,
stb, glad); their notices are in `extern/sextant/THIRD_PARTY_NOTICES.md`, which every wheel and sdist carries
(`*.dist-info/licenses/`). Portions of this software are copyright © 2026 The FreeType Project
(www.freetype.org). All rights reserved.
