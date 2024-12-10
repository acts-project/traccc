# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up all enabled libraries.
add_subdirectory( array_cmath )

if( ALGEBRA_PLUGINS_INCLUDE_EIGEN )
   add_subdirectory( eigen_cmath )
   add_subdirectory( eigen_eigen )
endif()

if( ALGEBRA_PLUGINS_INCLUDE_SMATRIX )
   add_subdirectory( smatrix_cmath )
   add_subdirectory( smatrix_smatrix )
endif()

if( ALGEBRA_PLUGINS_INCLUDE_VC )
   add_subdirectory( vc_cmath )
   add_subdirectory( vc_vc )
   if( NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "AppleClang" )
      add_subdirectory( vc_soa )
   endif()
endif()

if( ALGEBRA_PLUGINS_INCLUDE_FASTOR )
   add_subdirectory( fastor_fastor )
endif()

if( ALGEBRA_PLUGINS_INCLUDE_VECMEM )
   add_subdirectory( vecmem_cmath )
endif()
