# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the "build" of the traccc::vecmem library.
traccc_add_library( traccc_vecmem vecmem TYPE INTERFACE
  "include/traccc/plugins/algebra/vecmem_definitions.hpp" )
target_link_libraries( traccc_vecmem
  INTERFACE algebra::vecmem_cmath vecmem::core )
target_compile_definitions( traccc_vecmem
  INTERFACE TRACCC_CUSTOM_SCALARTYPE=${TRACCC_CUSTOM_SCALARTYPE} )
