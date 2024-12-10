# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Include the helper function(s).
include( vecmem-functions )

# Set up the used C++ standard(s).
set( CMAKE_CXX_STANDARD 17 CACHE STRING "The (host) C++ standard to use" )

# Turn on a number of warnings for the "known compilers".
if( ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" ) OR
    ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" ) )

   # Turn on a schizophrenic amount of warnings.
   vecmem_add_flag( CMAKE_CXX_FLAGS "-Wall" )
   vecmem_add_flag( CMAKE_CXX_FLAGS "-Wextra" )
   vecmem_add_flag( CMAKE_CXX_FLAGS "-Wshadow" )
   vecmem_add_flag( CMAKE_CXX_FLAGS "-Wunused-local-typedefs" )
   vecmem_add_flag( CMAKE_CXX_FLAGS "-pedantic" )
   vecmem_add_flag( CMAKE_CXX_FLAGS "-Wconversion" )

   # Fail on warnings, if asked for that behaviour.
   if( VECMEM_FAIL_ON_WARNINGS )
      vecmem_add_flag( CMAKE_CXX_FLAGS "-Werror" )
   endif()

elseif( "${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC" )

   # Basic flags for all build modes.
   string( REGEX REPLACE "/W[0-9]" "" CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS}" )
   vecmem_add_flag( CMAKE_CXX_FLAGS "/W4" )

   # Fail on warnings, if asked for that behaviour.
   if( VECMEM_FAIL_ON_WARNINGS )
      vecmem_add_flag( CMAKE_CXX_FLAGS "/WX" )
   endif()

endif()

# Do not allow symbols to be missing from shared libraries.
if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang" )
   # From AppleClang 15 onwards, the default is to fail on undefined symbols.
   if( "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS "15" )
      vecmem_add_flag( CMAKE_SHARED_LINKER_FLAGS "-Wl,-undefined,error" )
   endif()
elseif( ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" ) OR
        ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" ) )
   vecmem_add_flag( CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined" )
endif()
