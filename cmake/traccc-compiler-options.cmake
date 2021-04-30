# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Guard against multiple includes.
include_guard( GLOBAL )

# Include the helper function(s).
include( traccc-functions )

# Set up the used C++ standard(s).
set( CMAKE_CXX_STANDARD 17 CACHE STRING "The (host) C++ standard to use" )

# Turn on a number of warnings for the "known compilers".
if( ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" ) OR
    ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" ) )

   # Basic flags for all major build modes.
   foreach( mode RELEASE RELWITHDEBINFO MINSIZEREL DEBUG )
      traccc_add_flag( CMAKE_CXX_FLAGS_${mode} "-Wall" )
      traccc_add_flag( CMAKE_CXX_FLAGS_${mode} "-Wextra" )
   endforeach()
endif()
