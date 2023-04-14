# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the "build" of the traccc::vc library.
traccc_add_library( traccc_vc vc TYPE INTERFACE
  "include/traccc/plugins/algebra/vc_definitions.hpp" )
target_link_libraries( traccc_vc
  INTERFACE algebra::vc_vc detray::algebra_vc vecmem::core )
target_compile_definitions( traccc_vc
  INTERFACE TRACCC_CUSTOM_SCALARTYPE=${TRACCC_CUSTOM_SCALARTYPE} )
