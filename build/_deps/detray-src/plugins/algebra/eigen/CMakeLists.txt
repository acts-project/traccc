# Detray library, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

message(STATUS "Building the 'detray::algebra_eigen' plugin")

# A sanity check.
if(NOT ALGEBRA_PLUGINS_INCLUDE_EIGEN)
    message(
        WARNING
        "Eigen not available from Algebra Plugins. "
        "The configuration will likely fail."
    )
endif()

# Set up the library.
detray_add_library( detray_algebra_eigen algebra_eigen
   "include/detray/plugins/algebra/eigen_definitions.hpp"
)
target_link_libraries(
    detray_algebra_eigen
    INTERFACE algebra::eigen_eigen vecmem::core
)
target_compile_definitions(
    detray_algebra_eigen
    INTERFACE
        DETRAY_CUSTOM_SCALARTYPE=${DETRAY_CUSTOM_SCALARTYPE}
        DETRAY_ALGEBRA_EIGEN
)

# For some wicked reason CUDA keeps complaining about the Eigen headers, even
# though they are set up from a "system include path". So I had to explicitly
# disable the warning triggered by those headers, for anything that uses them.
# This is pretty bad, as we're now blind to these types of warnings/errors. :-(
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    target_compile_options(
        detray_algebra_eigen
        INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=20012>
    )
endif()

# Set up tests for the public header(s) of detray::algebra_eigen.
detray_test_public_headers( detray_algebra_eigen
   "detray/plugins/algebra/eigen_definitions.hpp"
)
