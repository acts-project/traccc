# Detray library, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the project.
cmake_minimum_required(VERSION 3.11)
project(detray VERSION 0.83.0 LANGUAGES CXX)

# Set up the used C++ standard(s).
set(CMAKE_CXX_STANDARD 20 CACHE STRING "The (host) C++ standard to use")
set(CMAKE_CXX_EXTENSIONS FALSE CACHE BOOL "Disable (host) C++ extensions")
set(CMAKE_CUDA_STANDARD 20 CACHE STRING "The (CUDA) C++ standard to use")
set(CMAKE_CUDA_EXTENSIONS FALSE CACHE BOOL "Disable (CUDA) C++ extensions")
set(CMAKE_SYCL_STANDARD 20 CACHE STRING "The (SYCL) C++ standard to use")

if(${CMAKE_CXX_STANDARD} LESS 20)
    message(
        SEND_ERROR
        "CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}, but detray requires C++>=20"
    )
endif()

# CMake include(s).
include(CMakeDependentOption)
include(GNUInstallDirs)
include(CTest)

# Flags controlling the meta-build system.
option(DETRAY_USE_SYSTEM_LIBS "Use system libraries be default" FALSE)

# Explicitly set the output directory for the binaries. Such that if this
# project is included by another project, the main project's configuration would
# win out.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
    CACHE PATH
    "Directory for the built binaries"
)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
    CACHE PATH
    "Directory for the built libraries"
)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
    CACHE PATH
    "Directory for the built static libraries"
)

# Include the Detray CMake code.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(detray-functions)

# Check if CUDA is available.
include(CheckLanguage)
check_language(CUDA)
set(DETRAY_BUILD_CUDA_DEFAULT FALSE)
if(CMAKE_CUDA_COMPILER)
    set(DETRAY_BUILD_CUDA_DEFAULT TRUE)
endif()

# Flags controlling which parts of Detray to build.
option(DETRAY_EIGEN_PLUGIN "Build Eigen math plugin" OFF)
option(DETRAY_SMATRIX_PLUGIN "Build ROOT/SMatrix math plugin" OFF)
option(DETRAY_VC_AOS_PLUGIN "Build Vc based AoS math plugin" OFF)
option(DETRAY_VC_SOA_PLUGIN "Build Vc based SoA math plugin" OFF)
option(DETRAY_SVG_DISPLAY "Build ActSVG display module" OFF)
option(DETRAY_BUILD_SYCL "Build the SYCL sources included in detray" OFF)
option(
    DETRAY_BUILD_CUDA
    "Build the CUDA sources included in detray"
    ${DETRAY_BUILD_CUDA_DEFAULT}
)
cmake_dependent_option(
    DETRAY_BUILD_HOST
    "Build the host sources included in detray"
    ON
    "DETRAY_BUILD_CUDA OR DETRAY_BUILD_SYCL"
    ON
)
option(DETRAY_BUILD_TEST_UTILS "Build the test utility library of Detray" OFF)
option(DETRAY_BUILD_VALIDATION_TOOLS "Build the validation tools of Detray" OFF)
option(DETRAY_BUILD_UNITTESTS "Build the unit tests of Detray" OFF)
option(
    DETRAY_BUILD_INTEGRATIONTESTS
    "Build the integration tests of Detray"
    OFF
)
option(DETRAY_BUILD_ALL_TESTS "Build unit and integrations tests of Detray" OFF)
option(DETRAY_BUILD_BENCHMARKS "Build the benchmark tests" OFF)
option(DETRAY_BUILD_CLI_TOOLS "Build the command line tools of Detray" OFF)
option(DETRAY_BUILD_TUTORIALS "Build the tutorial executables of Detray" OFF)
option(
    DETRAY_FAIL_ON_WARNINGS
    "Make the build fail on compiler/linker warnings"
    FALSE
)

# Need test functionality and svg display for command line tools
if(DETRAY_BUILD_CLI_TOOLS)
    set(DETRAY_BUILD_TESTING ON)
    set(DETRAY_SVG_DISPLAY ON)
endif()

# Convenience option to build tests
if(DETRAY_BUILD_ALL_TESTS)
    set(DETRAY_BUILD_UNITTESTS ON)
    set(DETRAY_BUILD_INTEGRATIONTESTS ON)
endif()

# Alias flag to build the tests
if(BUILD_TESTING AND (DETRAY_BUILD_UNITTESTS OR DETRAY_BUILD_INTEGRATIONTESTS))
    set(DETRAY_BUILD_TESTING ON)
endif()

# Validation utilities are needed for integration tests and tutorials
if(DETRAY_BUILD_TESTING OR DETRAY_BUILD_TUTORIALS)
    set(DETRAY_BUILD_VALIDATION_TOOLS ON)
endif()

# Test utilities are needed for validation utilities and benchmarks
if(
    DETRAY_BUILD_TESTING
    OR DETRAY_BUILD_VALIDATION_TOOLS
    OR DETRAY_BUILD_BENCHMARKS
)
    set(DETRAY_BUILD_TEST_UTILS ON)
endif()

# Svg display is needed for the validation utilities
if(DETRAY_BUILD_VALIDATION_TOOLS)
    set(DETRAY_SVG_DISPLAY ON)
endif()

# Clean up.
unset(DETRAY_BUILD_CUDA_DEFAULT)

# Check CUDA and SYCL C++ standards
if(${DETRAY_BUILD_CUDA} AND ${CMAKE_CUDA_STANDARD} LESS 20)
    message(
        SEND_ERROR
        "CMAKE_CUDA_STANDARD=${CMAKE_CUDA_STANDARD}, but detray requires C++>=20"
    )
endif()

if(${DETRAY_BUILD_SYCL} AND ${CMAKE_SYCL_STANDARD} LESS 20)
    message(
        SEND_ERROR
        "CMAKE_SYCL_STANDARD=${CMAKE_SYCL_STANDARD}, but detray requires C++>=20"
    )
endif()

# Set up VecMem.
option(DETRAY_SETUP_VECMEM "Set up the VecMem target(s) explicitly" TRUE)
option(
    DETRAY_USE_SYSTEM_VECMEM
    "Pick up an existing installation of VecMem from the build environment"
    ${DETRAY_USE_SYSTEM_LIBS}
)
if(DETRAY_SETUP_VECMEM)
    if(DETRAY_USE_SYSTEM_VECMEM)
        find_package(vecmem REQUIRED)
    else()
        add_subdirectory(extern/vecmem)
        # Make the "VecMem language code" available for the whole project.
        include("${VECMEM_LANGUAGE_DIR}/vecmem-check-language.cmake")
    endif()
endif()

# Set up Algebra Plugins.
option(
    DETRAY_SETUP_ALGEBRA_PLUGINS
    "Set up the Algebra Plugins target(s) explicitly"
    TRUE
)
option(
    DETRAY_USE_SYSTEM_ALGEBRA_PLUGINS
    "Pick up an existing installation of Algebra Plugins from the build environment"
    ${DETRAY_USE_SYSTEM_LIBS}
)
if(DETRAY_SETUP_ALGEBRA_PLUGINS)
    if(DETRAY_USE_SYSTEM_ALGEBRA_PLUGINS)
        find_package(algebra-plugins REQUIRED)
    else()
        add_subdirectory(extern/algebra-plugins)
    endif()
endif()

# Set up ACTSVG for displaying
option(
    DETRAY_SETUP_ACTSVG
    "Set up the actsvg target(s) explicitly"
    ${DETRAY_SVG_DISPLAY}
)
option(
    DETRAY_USE_SYSTEM_ACTSVG
    "Pick up an existing installation of actsvg from the build environment"
    ${DETRAY_USE_SYSTEM_LIBS}
)
if(DETRAY_SETUP_ACTSVG)
    if(DETRAY_USE_SYSTEM_ACTSVG)
        find_package(actsvg REQUIRED COMPONENTS core meta)
    else()
        add_subdirectory(extern/actsvg)
    endif()
endif()

# Set up JSON for I/O
option(
    DETRAY_SETUP_NLOHMANN
    "Set up the nlohmann::json target(s) explicitly"
    TRUE
)
option(
    DETRAY_USE_SYSTEM_NLOHMANN
    "Pick up an existing installation of nlohman::json from the build environment"
    ${DETRAY_USE_SYSTEM_LIBS}
)
if(DETRAY_SETUP_NLOHMANN)
    if(DETRAY_USE_SYSTEM_NLOHMANN)
        find_package(nlohmann_json REQUIRED)
    else()
        add_subdirectory(extern/nlohmann_json)
    endif()
endif()

# Set up dfelibs.
option(DETRAY_SETUP_DFELIBS "Set up the dfelibs target(s) explicitly" TRUE)
option(
    DETRAY_USE_SYSTEM_DFELIBS
    "Pick up an existing installation of dfelibs from the build environment"
    ${DETRAY_USE_SYSTEM_LIBS}
)
if(DETRAY_SETUP_DFELIBS)
    if(DETRAY_USE_SYSTEM_DFELIBS)
        find_package(dfelibs REQUIRED)
    else()
        add_subdirectory(extern/dfelibs)
    endif()
endif()

# Set up GoogleTest.
option(
    DETRAY_SETUP_GOOGLETEST
    "Set up the GoogleTest target(s) explicitly"
    ${DETRAY_BUILD_TESTING}
)
option(
    DETRAY_USE_SYSTEM_GOOGLETEST
    "Pick up an existing installation of GoogleTest from the build environment"
    ${DETRAY_USE_SYSTEM_LIBS}
)
if(DETRAY_SETUP_GOOGLETEST)
    if(DETRAY_USE_SYSTEM_GOOGLETEST)
        find_package(GTest REQUIRED)
    else()
        add_subdirectory(extern/googletest)
    endif()
endif()

# Set up Google Benchmark.
option(
    DETRAY_SETUP_BENCHMARK
    "Set up the Google Benchmark target(s) explicitly"
    ${DETRAY_BUILD_BENCHMARKS}
)
option(
    DETRAY_USE_SYSTEM_BENCHMARK
    "Pick up an existing installation of Google Benchmark from the build environment"
    ${DETRAY_USE_SYSTEM_LIBS}
)
if(DETRAY_SETUP_BENCHMARK)
    if(DETRAY_USE_SYSTEM_BENCHMARK)
        find_package(benchmark REQUIRED)
    else()
        add_subdirectory(extern/benchmark)
    endif()
endif()

# Set up covfie.
option(DETRAY_SETUP_COVFIE "Set up the covfie target(s) explicitly" TRUE)
option(
    DETRAY_USE_SYSTEM_COVFIE
    "Pick up an existing installation of covfie from the build environment"
    ${DETRAY_USE_SYSTEM_LIBS}
)
if(DETRAY_SETUP_COVFIE)
    if(DETRAY_USE_SYSTEM_COVFIE)
        find_package(covfie 0.5.0 REQUIRED)
    else()
        add_subdirectory(extern/covfie)
    endif()
endif()

# Set up all of the libraries of the project.
add_subdirectory(core)
add_subdirectory(detectors)
add_subdirectory(io)
add_subdirectory(plugins)

# Set up the test utilities and test(s).
cmake_dependent_option(
    DETRAY_ENABLE_SANITIZER
    "Compile tests with sanitizers"
    OFF
    "BUILD_TESTING AND DETRAY_BUILD_TESTING"
    OFF
)
if(DETRAY_BUILD_TESTING OR DETRAY_BUILD_TEST_UTILS)
    add_subdirectory(tests)
endif()

# Set up the tutorial(s).
if(DETRAY_BUILD_TUTORIALS)
    add_subdirectory(tutorials)
endif()

# Set up the packaging of the project.
include(detray-packaging)
