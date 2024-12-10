# Detray library, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required(VERSION 3.14)
include(FetchContent)

# Silence FetchContent warnings with CMake >=3.24.
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# Tell the user what's happening.
message(STATUS "Building Algebra Plugins as part of the Detray project")

# Declare where to get Algebra Plugins from.
set(DETRAY_ALGEBRA_PLUGINS_SOURCE
    "URL;https://github.com/acts-project/algebra-plugins/archive/refs/tags/v0.25.0.tar.gz;URL_MD5;8201744f8b2a8ff7fbcf21bd1b4549b9"
    CACHE STRING
    "Source for Algebra Plugins, when built as part of this project"
)
mark_as_advanced(DETRAY_ALGEBRA_PLUGINS_SOURCE)

# Mark the import as a system library on modern CMake versions
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.25.0)
    set(DETRAY_ALGEBRA_PLUGINS_SOURCE_FULL
        "${DETRAY_ALGEBRA_PLUGINS_SOURCE};SYSTEM"
    )
else()
    set(DETRAY_ALGEBRA_PLUGINS_SOURCE_FULL "${DETRAY_ALGEBRA_PLUGINS_SOURCE}")
endif()
mark_as_advanced(DETRAY_ALGEBRA_PLUGINS_SOURCE_FULL)

FetchContent_Declare(AlgebraPlugins ${DETRAY_ALGEBRA_PLUGINS_SOURCE_FULL})

# Options used in the build of Algebra Plugins.
set(ALGEBRA_PLUGINS_BUILD_TESTING
    FALSE
    CACHE BOOL
    "Turn off the build of the Algebra Plugins unit tests"
)
set(ALGEBRA_PLUGINS_BUILD_BENCHMARKS
    FALSE
    CACHE BOOL
    "Turn off the build of the Algebra Plugins benchmarks"
)
set(ALGEBRA_PLUGINS_INCLUDE_EIGEN
    ${DETRAY_EIGEN_PLUGIN}
    CACHE BOOL
    "Turn on the build of algebra::eigen"
)
set(ALGEBRA_PLUGINS_INCLUDE_VC
    ${DETRAY_VC_AOS_PLUGIN}
    CACHE BOOL
    "Turn on the build of algebra::vc_aos"
)
set(ALGEBRA_PLUGINS_INCLUDE_VECMEM
    TRUE
    CACHE BOOL
    "Turn on the build of algebra::vecmem_array"
)

# Build smatrix plugin only when it needs to be
# ROOT is not provided in native github compilers :/
set(ALGEBRA_PLUGINS_INCLUDE_SMATRIX
    ${DETRAY_SMATRIX_PLUGIN}
    CACHE BOOL
    "Turn on/off the build of algebra::smatrix"
)

set(ALGEBRA_PLUGINS_SETUP_EIGEN3
    ${DETRAY_EIGEN_PLUGIN}
    CACHE BOOL
    "Have Algebra Plugins set up Eigen3 for itself"
)
set(ALGEBRA_PLUGINS_USE_SYSTEM_EIGEN3
    FALSE
    CACHE BOOL
    "Have Algebra Plugins pick up Eigen3 from the system"
)
set(ALGEBRA_PLUGINS_SETUP_VC
    ${DETRAY_VC_AOS_PLUGIN}
    OR
    ${DETRAY_VC_SOA_PLUGIN}
    CACHE BOOL
    "Have Algebra Plugins set up Vc for itself"
)
set(ALGEBRA_PLUGINS_USE_SYSTEM_VC
    FALSE
    CACHE BOOL
    "Have Algebra Plugins build Vc itself"
)
set(ALGEBRA_PLUGINS_SETUP_VECMEM
    FALSE
    CACHE BOOL
    "Do not set up VecMem in Algebra Plugins"
)
set(ALGEBRA_PLUGINS_SETUP_GOOGLETEST
    FALSE
    CACHE BOOL
    "Do not set up GoogleTest in Algebra Plugins"
)

# Disable a deprecation warning coming from the Vc build.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(
    "-Wno-deprecated-enum-enum-conversion"
    DETRAY_HAS_NO_DEPRECATED_ENUM_ENUM_CONVERSION
)
if(DETRAY_HAS_NO_DEPRECATED_ENUM_ENUM_CONVERSION)
    detray_add_flag( CMAKE_CXX_FLAGS "-Wno-deprecated-enum-enum-conversion" )
endif()

# Get it into the current directory.
FetchContent_MakeAvailable(AlgebraPlugins)
