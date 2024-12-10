# Detray library, part of the ACTS project (R&D line)
#
# (c) 2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

message(STATUS "Building the 'detray::algebra_vc_soa' plugin")

# A sanity check.
if(NOT ALGEBRA_PLUGINS_INCLUDE_VC)
    message(
        WARNING
        "Vc not available from Algebra Plugins. "
        "The configuration will likely fail."
    )
endif()

include(detray-compiler-options-cpp)

# Use the preferred compiler flags from Vc for the entire project.
vc_set_preferred_compiler_flags()

# Set up the SoA library.
detray_add_library( detray_algebra_vc_soa algebra_vc_soa
   "include/detray/plugins/algebra/vc_soa_definitions.hpp"
)
target_link_libraries(
    detray_algebra_vc_soa
    INTERFACE algebra::vc_soa algebra::vc_soa_storage vecmem::core
)
target_compile_definitions(
    detray_algebra_vc_soa
    INTERFACE
        DETRAY_CUSTOM_SCALARTYPE=${DETRAY_CUSTOM_SCALARTYPE}
        DETRAY_ALGEBRA_VC_SOA
        ${Vc_DEFINITIONS}
)
target_compile_options(
    detray_algebra_vc_soa
    INTERFACE ${Vc_COMPILE_FLAGS} ${Vc_ARCHITECTURE_FLAGS}
)

# Set up tests for the public header(s) of detray::algebra_vc_soa.
detray_test_public_headers( detray_algebra_vc_soa
   "detray/plugins/algebra/vc_soa_definitions.hpp"
)
