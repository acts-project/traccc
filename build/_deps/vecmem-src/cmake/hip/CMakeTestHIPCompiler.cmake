# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
include( CMakeTestCompilerCommon )

# Start with the correct status message.
if( ${CMAKE_VERSION} VERSION_LESS 3.17 )
   PrintTestCompilerStatus( "HIP" "" )
else()
   PrintTestCompilerStatus( "HIP" )
endif()

# Try to use the HIP compiler.
file( WRITE
   "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/main.hip"
   "#include <hip/hip_runtime.h>\n"
   "int main() { return 0; }\n" )
try_compile( CMAKE_HIP_COMPILER_WORKS "${CMAKE_BINARY_DIR}"
   "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/main.hip"
   OUTPUT_VARIABLE __CMAKE_HIP_COMPILER_OUTPUT )

# Move the results of the test into a regular variable.
set( CMAKE_HIP_COMPILER_WORKS ${CMAKE_HIP_COMPILER_WORKS} )
unset( CMAKE_HIP_COMPILER_WORKS CACHE )

# Check the results of the test.
if( NOT CMAKE_HIP_COMPILER_WORKS )
   if( ${CMAKE_VERSION} VERSION_LESS 3.17 )
      PrintTestCompilerStatus( "HIP" " -- broken" )
   else()
      PrintTestCompilerResult( CHECK_FAIL "broken" )
   endif()
   message( FATAL_ERROR "The HIP compiler\n"
      "  \"${CMAKE_HIP_COMPILER}\"\n"
      "is not able to compile a simple test program.\n"
      "It fails with the following output:\n"
      "  ${__CMAKE_HIP_COMPILER_OUTPUT}\n\n"
      "CMake will not be able to correctly generate this project." )
endif()
if( ${CMAKE_VERSION} VERSION_LESS 3.17 )
   PrintTestCompilerStatus( "HIP" " -- works" )
else()
   PrintTestCompilerResult( CHECK_PASS "works" )
endif()
