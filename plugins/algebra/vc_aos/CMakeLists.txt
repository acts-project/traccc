# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the "build" of the traccc::vc_aos library.
traccc_add_library( traccc_vc_aos vc_aos TYPE INTERFACE
  "include/traccc/plugins/algebra/vc_aos_definitions.hpp" )
target_link_libraries( traccc_vc_aos
  INTERFACE algebra::vc_aos detray::algebra_vc_aos vecmem::core )
target_compile_definitions( traccc_vc_aos
  INTERFACE TRACCC_CUSTOM_SCALARTYPE=${TRACCC_CUSTOM_SCALARTYPE} )
