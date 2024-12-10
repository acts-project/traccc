# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
include( CMakeTestCompilerCommon )

# Start with the correct status message.
if( ${CMAKE_VERSION} VERSION_LESS 3.17 )
   PrintTestCompilerStatus( "SYCL" "" )
else()
   PrintTestCompilerStatus( "SYCL" )
endif()

# Try to use the HIP compiler.
file( WRITE
   "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/main.sycl"
   "#include <sycl/sycl.hpp>\n"
   "int main() {\n"
   "#if (!defined(CL_SYCL_LANGUAGE_VERSION)) &&"
   "    (!defined(SYCL_LANGUAGE_VERSION))\n"
   "#error \"SYCL language is not available!\"\n"
   "#endif\n"
   "return 0; }\n" )
try_compile( CMAKE_SYCL_COMPILER_WORKS "${CMAKE_BINARY_DIR}"
   "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/main.sycl"
   OUTPUT_VARIABLE __CMAKE_SYCL_COMPILER_OUTPUT )

# Move the results of the test into a regular variable.
set( CMAKE_SYCL_COMPILER_WORKS ${CMAKE_SYCL_COMPILER_WORKS} )
unset( CMAKE_SYCL_COMPILER_WORKS CACHE )

# Check the results of the test.
if( NOT CMAKE_SYCL_COMPILER_WORKS )
   if( ${CMAKE_VERSION} VERSION_LESS 3.17 )
      PrintTestCompilerStatus( "SYCL" " -- broken" )
   else()
      PrintTestCompilerResult( CHECK_FAIL "broken" )
   endif()
   message( FATAL_ERROR "The SYCL compiler\n"
      "  \"${CMAKE_SYCL_COMPILER}\"\n"
      "is not able to compile a simple test program.\n"
      "It fails with the following output:\n"
      "  ${__CMAKE_SYCL_COMPILER_OUTPUT}\n\n"
      "CMake will not be able to correctly generate this project." )
endif()
if( ${CMAKE_VERSION} VERSION_LESS 3.17 )
   PrintTestCompilerStatus( "SYCL" " -- works" )
else()
   PrintTestCompilerResult( CHECK_PASS "works" )
endif()
