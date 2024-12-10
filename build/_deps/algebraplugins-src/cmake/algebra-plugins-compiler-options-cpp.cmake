# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Include the helper function(s).
include( algebra-plugins-functions )

# Turn on a number of warnings for the "known compilers".
if( ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" ) OR
    ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" ) )

   # Basic flags for all build modes.
   algebra_add_flag( CMAKE_CXX_FLAGS "-Wall" )
   algebra_add_flag( CMAKE_CXX_FLAGS "-Wextra" )
   algebra_add_flag( CMAKE_CXX_FLAGS "-Wshadow" )
   algebra_add_flag( CMAKE_CXX_FLAGS "-Wunused-local-typedefs" )
   algebra_add_flag( CMAKE_CXX_FLAGS "-pedantic" )

   # Fail on warnings, if asked for that behaviour.
   if( ALGEBRA_PLUGINS_FAIL_ON_WARNINGS )
      algebra_add_flag( CMAKE_CXX_FLAGS "-Werror" )
   endif()

elseif( "${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC" )

   # Basic flags for all build modes.
   string( REGEX REPLACE "/W[0-9]" "" CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS}" )
   algebra_add_flag( CMAKE_CXX_FLAGS "/W4" )

   # Fail on warnings, if asked for that behaviour.
   if( ALGEBRA_PLUGINS_FAIL_ON_WARNINGS )
      algebra_add_flag( CMAKE_CXX_FLAGS "/WX" )
   endif()

   # Turn on the correct setting for the __cplusplus macro with MSVC.
   algebra_add_flag( CMAKE_CXX_FLAGS "/Zc:__cplusplus" )

endif()
