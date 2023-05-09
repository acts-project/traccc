# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the "build" of the traccc::eigen library.
traccc_add_library( traccc_eigen eigen TYPE INTERFACE
  "include/traccc/plugins/algebra/eigen_definitions.hpp" )
target_link_libraries( traccc_eigen
  INTERFACE algebra::eigen_eigen detray::algebra_eigen vecmem::core )
target_compile_definitions( traccc_eigen
  INTERFACE TRACCC_CUSTOM_SCALARTYPE=${TRACCC_CUSTOM_SCALARTYPE} )
