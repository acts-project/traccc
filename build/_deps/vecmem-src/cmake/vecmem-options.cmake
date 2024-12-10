# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Guard against multiple includes.
include_guard( GLOBAL )

# Look for the supported GPU languages. But only if the user didn't specify
# explicitly if (s)he wants them used. If they are turned on/off through
# cache variables explicitly, then skip looking for them at this point.
include( vecmem-check-language )
foreach( lang CUDA HIP SYCL )
   if( NOT DEFINED VECMEM_BUILD_${lang}_LIBRARY )
      vecmem_check_language( ${lang} )
   endif()
endforeach()

# Helper function for setting up the library building flags.
function( vecmem_lib_option language descr )

   # Figure out what the default value should be for the variable.
   set( _default FALSE )
   if( CMAKE_${language}_COMPILER )
      set( _default TRUE )
   endif()

   # Set up the configuration option.
   option( VECMEM_BUILD_${language}_LIBRARY "${descr}" ${_default} )

endfunction( vecmem_lib_option )

# Flag specifying whether CUDA support should be built.
vecmem_lib_option( CUDA "Build the vecmem::cuda library" )

# Flag specifying whether HIP support should be built.
vecmem_lib_option( HIP "Build the vecmem::hip library" )

# Flag specifying whether SYCL support should be built.
vecmem_lib_option( SYCL "Build the vecmem::sycl library" )

# Use folders for organizing targets in IDEs.
set_property( GLOBAL PROPERTY USE_FOLDERS ON )

# Debug message output level in the code.
set( VECMEM_DEBUG_MSG_LVL 0 CACHE STRING
   "Debug message output level" )

# Set the default library type to build.
option( BUILD_SHARED_LIBS
   "Flag for building shared/static libraries" TRUE )

# Decide whether warnings in the code should be treated as errors.
option( VECMEM_FAIL_ON_WARNINGS
   "Make the build fail on compiler/linker warnings" FALSE )

# Decide whether runtime asynchronous synhronization errors should be fatal.
option( VECMEM_FAIL_ON_ASYNC_ERRORS
   "Make the build fail on asynchronous synchronization errors" FALSE )

# Build the project's documentation.
option( VECMEM_BUILD_DOCS "Build the project's documentation" OFF )
