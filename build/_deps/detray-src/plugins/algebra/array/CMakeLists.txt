# Detray library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the library.
detray_add_library( detray_algebra_array algebra_array
   "include/detray/plugins/algebra/array_definitions.hpp"
)
target_link_libraries(
    detray_algebra_array
    INTERFACE algebra::array_cmath vecmem::core
)
target_compile_definitions(
    detray_algebra_array
    INTERFACE
        DETRAY_CUSTOM_SCALARTYPE=${DETRAY_CUSTOM_SCALARTYPE}
        DETRAY_ALGEBRA_ARRAY
)

# Set up tests for the public header(s) of detray::algebra_array.
detray_test_public_headers( detray_algebra_array
   "detray/plugins/algebra/array_definitions.hpp"
)
