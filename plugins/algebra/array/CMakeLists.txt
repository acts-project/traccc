# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the "build" of the traccc::array library.
traccc_add_library( traccc_array array TYPE INTERFACE
  "include/traccc/plugins/algebra/array_definitions.hpp" )
target_link_libraries( traccc_array
  INTERFACE algebra::array_cmath detray::algebra_array vecmem::core )
target_compile_definitions( traccc_array
  INTERFACE TRACCC_CUSTOM_SCALARTYPE=${TRACCC_CUSTOM_SCALARTYPE} )
