# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the "build" of the traccc::smatrix library.
traccc_add_library( traccc_smatrix smatrix TYPE INTERFACE
  "include/traccc/plugins/algebra/smatrix_definitions.hpp" )
target_link_libraries( traccc_smatrix
  INTERFACE algebra::smatrix_smatrix detray::algebra_smatrix vecmem::core )
target_compile_definitions( traccc_smatrix
  INTERFACE TRACCC_CUSTOM_SCALARTYPE=${TRACCC_CUSTOM_SCALARTYPE} )
