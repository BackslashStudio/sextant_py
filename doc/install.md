# Installing sextant for Python

On PyPI the package is **`pysextant`** (`sextant` there is an unrelated project). It imports as `sextant`.

```sh
python -m pip install pysextant
```

The current release is **1.1.0** (`pysextant==1.1.0` to ask for it by number).

PyPI has wheels, which install without a compiler, for Python 3.10 or later on Windows x86-64, Linux x86-64
(glibc 2.27 or later) and macOS 11 or later (Apple Silicon and Intel). Anywhere else pip builds from the source
distribution, which needs the [C++ toolchain](#1-the-c-toolchain). The rest of this page covers installing
from the repository or a wheel file, conda, and what each platform needs at run time.

- [Requirements](#requirements)
- [From source](#from-source): [the C++ toolchain](#1-the-c-toolchain) · [installing](#2-installing) ·
  [checking it works](#3-checking-it-works)
- [From a wheel](#from-a-wheel)
- [In a conda environment](#in-a-conda-environment)
- [Running it](#running-it): [Linux](#linux) · [macOS](#macos) · [Windows](#windows)
- [Upgrading and uninstalling](#upgrading-and-uninstalling)
- [Troubleshooting](#troubleshooting)

For working on the binding itself (an editable install that rebuilds on import, the test suite), see
*Development* in the [README](../README.md#development).

---

## Requirements

| | |
|---|---|
| Python | 3.10 or later (CPython; the free-threaded 3.13t build is not supported) |
| Python packages | numpy ≥ 1.23, which pip installs for you |
| Platforms | Windows 10+ x86_64; Linux x86_64 with glibc 2.27+; macOS 11+ (arm64 or x86_64) |
| Graphics | OpenGL 3.3 for windows and PNG export; SVG export needs no OpenGL |

Building from source also needs:

- a **C++20 compiler**: MSVC (Visual Studio or its Build Tools, with the *Desktop development with C++*
  workload), GCC 13 or later, or Apple Clang from Xcode or its command line tools. Tested with MSVC 14.51,
  GCC 13, 14 and 16, and Apple Clang;
- **CMake 3.21** or later and **git**;
- **network access** during the build: GLFW, FreeType, libpng and zlib are downloaded and compiled into the
  package.

Building takes a few minutes, mostly compiling those libraries. The result is self-contained: one native
module, statically linked, with no `sextant.dll` or `libsextant.so` to keep next to it.

---

## From source

### 1. The C++ toolchain

**Windows.** Install Visual Studio or the *Build Tools for Visual Studio* with the *Desktop
development with C++* workload. Any shell works afterwards; the build finds the compiler by itself, so you do
not need a *Developer Command Prompt*. CMake comes with that workload, or install it from cmake.org or
`pip install cmake`.

**Linux.** A compiler, CMake, and the X11 and OpenGL development headers:

```sh
# Debian / Ubuntu (24.04 or later for GCC 13 as the default compiler)
sudo apt install build-essential cmake git \
    libgl-dev libegl-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxext-dev

# Fedora / RHEL / AlmaLinux
sudo dnf install gcc-c++ cmake git \
    mesa-libGL-devel mesa-libEGL-devel libX11-devel libXrandr-devel libXinerama-devel \
    libXcursor-devel libXi-devel libXext-devel

# Arch
sudo pacman -S base-devel cmake git mesa libx11 libxrandr libxinerama libxcursor libxi libxext
```

**macOS.** The Xcode command line tools (`xcode-select --install`) and CMake (`brew install cmake`, or
`pip install cmake`).

### 2. Installing

Use a virtual environment, and run pip through the Python you will use (`python -m pip`), so the package lands
there.

Straight from GitHub; pip clones the repository and its `extern/sextant` submodule itself:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
python -m pip install "git+https://github.com/BackslashStudio/sextant_py.git"
```

Or from a clone, which lets you pick a branch or commit first. **Clone with `--recursive`**: the C++ library
is a git submodule, and without it the build stops with *No sextant sources at …*.

```sh
git clone --recursive https://github.com/BackslashStudio/sextant_py.git
python -m pip install ./sextant_py
```

If you already cloned without `--recursive`, run `git submodule update --init` inside the clone.

pip builds in an isolated environment of its own, fetching the build tools (scikit-build-core, nanobind) for
that build only; nothing beyond numpy is added to your environment. Add `-v` to watch the compiler output.

### 3. Checking it works

```sh
python -c "import sextant, numpy as np; f = sextant.Figure(); f.axes().line(np.sin(np.arange(60) / 6)); f.savefig('check.png'); print('ok')"
```

That writes `check.png` without opening a window. To see a window:

```sh
python -c "import sextant; f = sextant.Figure(); f.axes().line([1, 3, 2, 4]); f.show()"
```

The second command blocks until you close the window.

---

## From a wheel

A wheel is a pre-built package: installing one needs no compiler. `pip install pysextant` picks the
right one from PyPI by itself; a wheel file is for a machine without access to PyPI, or a build between
releases. Wheel files come from:

- the *Artifacts* section of a run of the repository's `wheels` workflow on GitHub (one zip per OS:
  `wheels-Windows`, `wheels-Linux`, `wheels-macOS`), or `gh run download <run-id> -R
  BackslashStudio/sextant_py -n wheels-Linux`;
- your own build: `python -m pip wheel ./sextant_py -w dist` builds one for the Python that runs it,
  which you can then copy to machines of the same OS, architecture and Python version. On Linux, such a
  wheel needs at least the glibc of the machine that built it, unlike the `wheels` workflow's, which run on
  glibc 2.27 and later.

Pick the file whose tags match your Python and platform:

| Your Python | Wheel |
|---|---|
| 3.10 | `pysextant-<version>-cp310-cp310-<platform>.whl` |
| 3.11 | `pysextant-<version>-cp311-cp311-<platform>.whl` |
| 3.12 or later | `pysextant-<version>-cp312-abi3-<platform>.whl` (one wheel for every later version) |

`<platform>` is `win_amd64`, a `manylinux_…_x86_64` tag, `macosx_11_0_arm64` or `macosx_11_0_x86_64`.

```sh
python -m pip install path/to/pysextant-1.1.0-cp312-abi3-win_amd64.whl
```

pip refuses a wheel that does not fit your interpreter or platform ("is not a supported wheel on this
platform"), so a wrong pick fails safely.

---

## In a conda environment

conda environments install sextant with pip, like any package that is not on conda-forge. Two rules:

- **Let conda install numpy first**, so pip finds it satisfied and does not add a PyPI build beside it.
- **Use the environment's pip** (`python -m pip` with the environment active).

From PyPI:

```sh
conda create -n sextant python=3.12 numpy
conda activate sextant
python -m pip install pysextant
```

From a wheel file, the same with `python -m pip install path/to/pysextant-<version>-cp312-abi3-<platform>.whl`.

From source, with CMake from conda and the compiler from the system (on Linux, the system GCC and the
development packages from [step 1](#1-the-c-toolchain); conda-forge's own compilers use a separate sysroot
that lacks the X11 and OpenGL headers):

```sh
conda create -n sextant python=3.12 numpy cmake ninja git
conda activate sextant
python -m pip install "git+https://github.com/BackslashStudio/sextant_py.git"
```

---

## Running it

Exporting (`savefig()`, `render_png()`, `render_svg()`) works on any machine the package installs on.
Windows need a little more, depending on the platform.

### Linux

- **Windows use X11.** Under Wayland they open through XWayland, so `DISPLAY` must be set (it is, in a normal
  desktop session).
- **With no display** (ssh, a container, CI), exports still work: PNG renders through EGL. That needs Mesa's
  EGL and a renderer, and a font, which a minimal system may lack. `show()` raises `RuntimeError` there.

  ```sh
  # Debian / Ubuntu
  sudo apt install libgl1 libegl1 libegl-mesa0 libgl1-mesa-dri fonts-dejavu-core
  # Fedora / RHEL / AlmaLinux
  sudo dnf install mesa-libGL mesa-libEGL mesa-dri-drivers dejavu-sans-fonts
  ```

### macOS

Windows belong to the main thread. A window responds while the main thread is inside sextant (`show()`,
`wait_closed()`, `run()`, `poll_events()`) or waiting at an interactive prompt (`python -i`, IPython). A script
that opens a window and then computes for a minute without calling one of those has a frozen window for that
minute. See [Live updates](api.md#live-updates).

### Windows

An OpenGL 3.3 driver is needed for windows and PNG. Every current GPU driver has one. A virtual machine or
remote-desktop session without one can use Mesa's software renderer, the `opengl32.dll` of a
[mesa-dist-win](https://github.com/pal1000/mesa-dist-win) release, whose deployment script can install it for
one program (`python.exe`).

---

## Upgrading and uninstalling

```sh
python -m pip install --upgrade pysextant
python -m pip uninstall pysextant
```

From the repository instead, `python -m pip install --force-reinstall --no-deps
"git+https://github.com/BackslashStudio/sextant_py.git"`: `--force-reinstall` because the version number stays
the same between commits, so pip may otherwise find that version installed and keep it; `--no-deps` leaves
numpy alone.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `pip install sextant` installs something else | That is an unrelated project on PyPI: `pip uninstall sextant`, then `pip install pysextant` |
| `pip install pysextant` finds no matching version | Python is older than 3.10, or pip is too old to read the wheel tags: `python -m pip install --upgrade pip` |
| *No sextant sources at …/extern/sextant* | The clone has no submodule: `git submodule update --init`, or clone again with `--recursive` |
| CMake cannot find a C++ compiler, or rejects C++20 | Install the toolchain of [step 1](#1-the-c-toolchain); on Linux check `g++ --version` is 13 or later |
| *Could NOT find X11* or *OpenGL* during the build (Linux) | The development packages of [step 1](#1-the-c-toolchain) are missing |
| The build fails while downloading GLFW, FreeType, libpng or zlib | No network, or a proxy: the build fetches them from GitHub |
| `ImportError: … version 'GLIBCXX_…' not found` on Linux, often in a conda environment | The module was built by a newer GCC than the `libstdc++` the environment loads. Current sources link libstdc++ statically, so [reinstall](#upgrading-and-uninstalling) from the repository's current `main`. A `GLIBC_…` version in the message instead means a wheel built on a newer system than this one: build from source here, or use a wheel from the `wheels` workflow |
| `RuntimeError` from `show()` on Linux, mentioning `DISPLAY` | No X server; set `DISPLAY` or export to a file instead |
| PNG export fails on a headless Linux machine | Mesa's EGL or a font is missing; see [Linux](#linux) |
| A blank or failed window on Windows in a VM | No OpenGL 3.3 driver; see [Windows](#windows) |
| A `RuntimeWarning` about a missing font | No system font was found; install one (DejaVu is enough) |
