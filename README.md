# sextant for Python

Python bindings for [sextant](https://github.com/BackslashStudio/sextant), a C++20 library for scientific
plotting. Work in progress: not yet published.

## Building from source

```sh
git clone --recursive <this repo>
uv sync
uv run pytest
```

A C++20 compiler and CMake 3.21+ are required (on Windows, Visual Studio with the C++ workload; any shell
works). The install is editable: after a change to the C++ sources, the next `import sextant` rebuilds the
extension. To build against another sextant checkout instead of the `extern/sextant` submodule, set
`SEXTANT_SOURCE_DIR` to its absolute path and run `uv sync --reinstall-package sextant`.
