# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Guard against multiple includes.
include_guard( GLOBAL )

# CMake include(s).
include( CMakeParseArguments )

# Helper function for setting up the algebra plugins libraries.
#
# Usage: algebra_add_library( algebra_array array "header1.hpp"... )
#
function( algebra_add_library fullname basename )

   # Create the library.
   add_library( ${fullname} INTERFACE ${ARG_UNPARSED_ARGUMENTS} )

   # Set up how clients should find its headers.
   target_include_directories( ${fullname} INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> )

   # Make sure that the library is available as "algebra::${basename}" in every
   # situation.
   set_target_properties( ${fullname} PROPERTIES EXPORT_NAME ${basename} )
   add_library( algebra::${basename} ALIAS ${fullname} )

   # Set up the installation of the library and its headers.
   install( TARGETS ${fullname}
      EXPORT algebra-plugins-exports
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" )
   install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" )

endfunction( algebra_add_library )

# Helper function testing the algebra plugins public headers.
#
# It can be used to test that public headers would include everything
# that they need to work, and that the CMake library targets would take
# care of declaring all of their dependencies correctly for the public
# headers to work.
#
# Usage: algebra_test_public_headers( algebra_array_cmath
#                                     include/header1.hpp ... )
#
function( algebra_test_public_headers library )

   # If testing is not turned on, don't do anything.
   if( ( NOT BUILD_TESTING ) OR ( NOT ALGEBRA_PLUGINS_BUILD_TESTING ) )
      return()
   endif()

   # All arguments are treated as header file names.
   foreach( _headerName ${ARGN} )

      # Make the header filename into a "string".
      string( REPLACE "/" "_" _headerNormName "${_headerName}" )
      string( REPLACE "." "_" _headerNormName "${_headerNormName}" )

      # Write a small source file that would test that the public
      # header can be used as-is.
      set( _testFileName
         "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/test_${_headerNormName}.cpp" )
      file( WRITE "${_testFileName}"
         "#include \"${_headerName}\"\n"
         "int main() { return 0; }" )

      # Set up an executable that would build it. But hide it, don't put it
      # into ${CMAKE_BINARY_DIR}/bin.
      add_executable( "test_${_headerNormName}" "${_testFileName}" )
      target_link_libraries( "test_${_headerNormName}" PRIVATE ${library} )
      set_target_properties( "test_${_headerNormName}" PROPERTIES
         RUNTIME_OUTPUT_DIRECTORY
         "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}" )

   endforeach()

endfunction( algebra_test_public_headers )

# Helper function for setting up the algebra plugin tests.
#
# Usage: algebra_add_test( array source1.cpp source2.cpp
#                          LINK_LIBRARIES algebra::array )
#
function( algebra_add_test name )

   # Parse the function's options.
   cmake_parse_arguments( ARG "" "" "LINK_LIBRARIES" ${ARGN} )

   # Create the test executable.
   set( test_exe_name "algebra_test_${name}" )
   add_executable( ${test_exe_name} ${ARG_UNPARSED_ARGUMENTS} )
   if( ARG_LINK_LIBRARIES )
      target_link_libraries( ${test_exe_name} PRIVATE ${ARG_LINK_LIBRARIES} )
   endif()

   # Run the executable as the test.
   add_test( NAME ${test_exe_name}
      COMMAND $<TARGET_FILE:${test_exe_name}> )

endfunction( algebra_add_test )

# Helper function for setting up the algebra plugin benchmarks.
#
# Usage: algebra_add_benchmark( array source1.cpp source2.cpp
#                               LINK_LIBRARIES algebra::array )
#
function( algebra_add_benchmark name )

   # Parse the function's options.
   cmake_parse_arguments( ARG "" "" "LINK_LIBRARIES" ${ARGN} )

   # Create the test executable.
   set( bench_exe_name "algebra_benchmark_${name}" )
   add_executable( ${bench_exe_name} ${ARG_UNPARSED_ARGUMENTS} )
   if( ARG_LINK_LIBRARIES )
      target_link_libraries( ${bench_exe_name} PRIVATE ${ARG_LINK_LIBRARIES} )
   endif()

endfunction( algebra_add_benchmark )

# Helper function for adding individual flags to "flag variables".
#
# Usage: algebra_add_flag( CMAKE_CXX_FLAGS "-Wall" )
#
function( algebra_add_flag name value )

   # Escape special characters in the value:
   set( matchedValue "${value}" )
   foreach( c "*" "." "^" "$" "+" "?" )
      string( REPLACE "${c}" "\\${c}" matchedValue "${matchedValue}" )
   endforeach()

   # Check if the variable already has this value in it:
   if( "${${name}}" MATCHES "${matchedValue}" )
      return()
   endif()

   # If not, then let's add it now:
   set( ${name} "${${name}} ${value}" PARENT_SCOPE )

endfunction( algebra_add_flag )
