# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake version requirement.
cmake_minimum_required( VERSION 3.14 )

# VecMem include(s).
include( vecmem-check-sycl-code-compiles )

# Check whether a given piece of SYCL code compiles.
#
# Usage: vecmem_check_sycl_source_compiles(<code> <resultVar>
#                                          [COMPILE_DEFINITIONS <definitions>]
#                                          [CMAKE_FLAGS <flags>])
#
function( vecmem_check_sycl_source_compiles code resultVar )

   # Return early, if the result variable already has a value.
   if( DEFINED ${resultVar} )
      return()
   endif()

   # Generate a source file with the given code.
   file( WRITE
      "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${resultVar}.sycl"
      "${code}" )

   # Use the "file based" version of the function.
   vecmem_check_sycl_code_compiles( ${resultVar}
      "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${resultVar}.sycl"
      ${ARGN} )

endfunction( vecmem_check_sycl_source_compiles )
