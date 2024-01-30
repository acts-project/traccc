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

# Function for declaring the (installable) executables of the project
# This version calls the alpaka_add_executable() function to create the executable,
# which is setup to use the correct compiler flags depending on the build type.
#
# Usage: traccc_add_alpaka_executable( super_exe super_source.cpp )
#
function( traccc_add_alpaka_executable name )

   # Parse the function's options.
   cmake_parse_arguments( ARG "" "" "LINK_LIBRARIES" ${ARGN} )

   # Set up the executable.
   set( exe_name "traccc_${name}" )
   add_executable( ${exe_name} ${ARG_UNPARSED_ARGUMENTS} )
   if( ARG_LINK_LIBRARIES )
      target_link_libraries( ${exe_name} PRIVATE ${ARG_LINK_LIBRARIES} )
   endif()

   target_compile_definitions(${exe_name} PRIVATE ALPAKA_HOST_ONLY)

   # Make sure that the executable is available as "traccc::${name}" in every
   # situation.
   set_target_properties( ${exe_name} PROPERTIES EXPORT_NAME ${name} )
   add_executable( traccc::${name} ALIAS ${exe_name} )

   # Set up the installation of the executable.
   install( TARGETS ${exe_name}
      EXPORT traccc-exports
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" )

endfunction( traccc_add_alpaka_executable )

# Helper function for setting up the traccc tests.
#
# Usage: traccc_add_test( core_containers source1.cpp source2.cpp
#                         LINK_LIBRARIES traccc::core )
#
function( traccc_add_alpaka_test name )

   # Parse the function's options.
   cmake_parse_arguments( ARG "" "" "LINK_LIBRARIES" ${ARGN} )

   # Create the test executable.
   set( test_exe_name "traccc_test_${name}" )
   alpaka_add_executable( ${test_exe_name} ${ARG_UNPARSED_ARGUMENTS} )
   if( ARG_LINK_LIBRARIES )
      target_link_libraries( ${test_exe_name} PRIVATE ${ARG_LINK_LIBRARIES} )
   endif()

   # Discover all of the tests from the execuable, and set them up as individual
   # CTest tests. All the while ensuring that they would find their data files.
   gtest_discover_tests( ${test_exe_name}
      PROPERTIES ENVIRONMENT
                 TRACCC_TEST_DATA_DIR=${PROJECT_SOURCE_DIR}/data
      DISCOVERY_TIMEOUT 20 )

endfunction( traccc_add_alpaka_test )
