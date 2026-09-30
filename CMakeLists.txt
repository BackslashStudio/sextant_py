cmake_minimum_required(VERSION 3.21...4.3)
project(sextant_py LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# sextant is compiled in this build and linked statically into the extension:
# one module cannot disagree with itself about the C++ ABI (the API passes
# std::string/std::vector across), and the wheel carries no second binary.
set(SEXTANT_SOURCE_DIR "" CACHE PATH "sextant source tree; empty = the extern/sextant submodule")
if(NOT SEXTANT_SOURCE_DIR)
    set(SEXTANT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/sextant")
endif()
if(NOT EXISTS "${SEXTANT_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "No sextant sources at ${SEXTANT_SOURCE_DIR}: "
                        "run `git submodule update --init`, or set SEXTANT_SOURCE_DIR")
endif()
message(STATUS "sextant sources: ${SEXTANT_SOURCE_DIR}")

# Link libGL.so.1, not GLVND's libOpenGL.so.0 + libGLX.so.0: manylinux allows
# only the former, and auditwheel would bundle the latter into the wheel.
set(OpenGL_GL_PREFERENCE LEGACY)

# Static C++ runtime for everything in the build (sextant, its bundled deps,
# nanobind): python.org's Python ships no msvcp140.dll, and a process that
# loaded an older one first would break std::mutex in a /MD module.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(SEXTANT_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(SEXTANT_BUNDLE_DEPS  ON  CACHE BOOL "" FORCE)
set(SEXTANT_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
# sextant pins GLFW to X11 on Linux, so its Wayland backend is dead code, and
# building it needs wayland-scanner/protocols and xkbcommon headers.
set(GLFW_BUILD_WAYLAND   OFF CACHE BOOL "" FORCE)
add_subdirectory("${SEXTANT_SOURCE_DIR}" sextant EXCLUDE_FROM_ALL)

find_package(Python 3.10 REQUIRED
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_sextant STABLE_ABI NB_STATIC
    src/ext/module.cpp
    src/ext/axes.cpp
    src/ext/axes3d.cpp
    src/ext/casters.cpp
    src/ext/data.cpp
    src/ext/events.cpp
    src/ext/figure.cpp
    src/ext/lifecycle.cpp
    src/ext/messages.cpp)
target_link_libraries(_sextant PRIVATE sextant::sextant_static)

# Export PyInit__sextant alone (see the map file). macOS needs nothing: with
# libc++ and hidden visibility it already exports that symbol only.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_sextant_map "${CMAKE_CURRENT_SOURCE_DIR}/src/ext/_sextant.map")
    target_link_options(_sextant PRIVATE "LINKER:--version-script=${_sextant_map}")
    set_property(TARGET _sextant APPEND PROPERTY LINK_DEPENDS "${_sextant_map}")
endif()

install(TARGETS _sextant LIBRARY DESTINATION sextant COMPONENT python)
