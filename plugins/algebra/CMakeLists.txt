# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Declare the traccc::algebra library.
traccc_add_library( traccc_algebra algebra TYPE INTERFACE )

# Make use of algebra::array_cmath in all cases.
add_subdirectory( array )
target_link_libraries( traccc_algebra INTERFACE traccc::array )

# Make use of algebra::vecmem_cmath in all cases.
add_subdirectory( vecmem )
target_link_libraries( traccc_algebra INTERFACE traccc::vecmem )

# Make use of algebra::eigen_eigen, if it is available.
if( ALGEBRA_PLUGINS_INCLUDE_EIGEN )
  add_subdirectory( eigen )
  target_link_libraries( traccc_algebra INTERFACE traccc::eigen )
endif()

# Make use of algebra::smatrix_smatrix, if it is available.
if( ALGEBRA_PLUGINS_INCLUDE_SMATRIX )
  add_subdirectory( smatrix )
  target_link_libraries( traccc_algebra INTERFACE traccc::smatrix )
endif()

# Make use of algebra::vc_aos, if it is available.
if( ALGEBRA_PLUGINS_INCLUDE_VC )
  add_subdirectory( vc_aos )
  target_link_libraries( traccc_algebra INTERFACE traccc::vc_aos )
endif()
