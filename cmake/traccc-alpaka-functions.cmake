# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

cmake_minimum_required( VERSION 3.16 )

# Guard against multiple includes.
include_guard( GLOBAL )

# Function for declaring the libraries of the project.
# This version calls the alpaka_add_library() function to create the library,
# which is setup to use the correct compiler flags depending on the build type.
#
# Usage: traccc_add_alpaka_library( traccc_core core
#                                   [TYPE SHARED/INTERFACE/STATIC]
#                                   include/source1.hpp source2.cpp )
#
function( traccc_add_alpaka_library fullname basename )

   # Parse the function's options.
   cmake_parse_arguments( ARG "" "TYPE" "" ${ARGN} )

   # Decide what sources to give to the library.
   set( _sources ${ARG_UNPARSED_ARGUMENTS} )
   if( "${ARG_TYPE}" STREQUAL "INTERFACE" )
      set( _sources )
   endif()

   # Create the library.
   alpaka_add_library( ${fullname} ${ARG_TYPE} ${_sources} )

   # Set up how clients should find its headers.
   if( IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" )
      set( _depType PUBLIC )
      if( "${ARG_TYPE}" STREQUAL "INTERFACE" )
         set( _depType INTERFACE )
      endif()
      target_include_directories( ${fullname} ${_depType}
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> )
      unset( _depType )
   endif()

   # Make sure that the library is available as "traccc::${basename}" in every
   # situation.
   set_target_properties( ${fullname} PROPERTIES EXPORT_NAME ${basename} )
   add_library( traccc::${basename} ALIAS ${fullname} )

   # Specify the (SO)VERSION of the library.
   if( NOT "${ARG_TYPE}" STREQUAL "INTERFACE" )
      set_target_properties( ${fullname} PROPERTIES
         VERSION ${PROJECT_VERSION}
         SOVERSION ${PROJECT_VERSION_MAJOR} )
   endif()

   # Set up the installation of the library and its headers.
   install( TARGETS ${fullname}
      EXPORT traccc-exports
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" )
   if( IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" )
      install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
         DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" )
   endif()

endfunction( traccc_add_alpaka_library )

macro (traccc_enable_language_alpaka)
   #enable_language cannot be called by a function: put it in a macro
   if(alpaka_ACC_GPU_CUDA_ENABLE)
     enable_language(CUDA)
     include( traccc-compiler-options-cuda )
   elseif(alpaka_ACC_GPU_HIP_ENABLE)
     enable_language(HIP)
   elseif(alpaka_ACC_SYCL_ENABLE)
     enable_language(SYCL)
     include( traccc-compiler-options-sycl )
   endif()
endmacro(traccc_enable_language_alpaka)
