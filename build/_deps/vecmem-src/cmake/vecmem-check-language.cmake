# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
include( CheckLanguage )

# Cache the location of this directory.
set( VECMEM_LANGUAGE_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH
   "Directory holding the VecMem language files" )
mark_as_advanced( VECMEM_LANGUAGE_DIR )

# Teach CMake about VecMem's custom language files.
list( INSERT CMAKE_MODULE_PATH 0
   "${VECMEM_LANGUAGE_DIR}"
   "${VECMEM_LANGUAGE_DIR}/hip"
   "${VECMEM_LANGUAGE_DIR}/sycl" )

# Code mimicking CMake's CheckLanguage.cmake module. But making sure that the
# VecMem specific code is used while looking for the non-standard languages.
macro( vecmem_check_language lang )

   # If CMAKE_${lang}_COMPILER is already set (even to an invalid value),
   # don't look for it again. Also ignore anything but the HIP "language".
   if( NOT DEFINED CMAKE_${lang}_COMPILER )

      # Handle the HIP and SYCL cases.
      if( ( "${lang}" STREQUAL "HIP" ) OR
          ( "${lang}" STREQUAL "SYCL" ) )

         # Greet the user.
         set( _desc "Looking for a ${lang} compiler" )
         if( ${CMAKE_VERSION} VERSION_LESS 3.17 )
            message( STATUS "${_desc}" )
         else()
            message( CHECK_START "${_desc}" )
         endif()

         # Clean up.
         file( REMOVE_RECURSE
            "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/Check${lang}" )

         # Set up a test project, which will be used to check for the viability
         # of HIP.
         file( WRITE
            "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/Check${lang}/CMakeLists.txt"
            "cmake_minimum_required( VERSION ${CMAKE_VERSION} )\n"
            "project( Check${lang} LANGUAGES CXX )\n"
            "list( INSERT CMAKE_MODULE_PATH 0 "
            "      \"${VECMEM_LANGUAGE_DIR}\" "
            "      \"${VECMEM_LANGUAGE_DIR}/hip\" "
            "      \"${VECMEM_LANGUAGE_DIR}/sycl\" )\n"
            "enable_language( ${lang} )\n"
            "file( WRITE \"\${CMAKE_CURRENT_BINARY_DIR}/result.cmake\"\n"
            "   \"set( CMAKE_${lang}_COMPILER \\\"\${CMAKE_${lang}_COMPILER}\\\" )\" )" )
         execute_process(
            WORKING_DIRECTORY
               "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/Check${lang}"
            COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} .
            OUTPUT_VARIABLE _vcl_output
            ERROR_VARIABLE _vcl_output
            RESULT_VARIABLE _vcl_result )

         # Check if the call succeeded.
         include(
            "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/Check${lang}/result.cmake"
            OPTIONAL )
         if( CMAKE_${lang}_COMPILER AND "${_vcl_result}" STREQUAL "0" )
            set( _CHECK_COMPILER_STATUS CHECK_PASS )
         else()
            set( _CHECK_COMPILER_STATUS CHECK_FAIL )
            set( CMAKE_${lang}_COMPILER NOTFOUND )
         endif()

         # Let the user know what happened.
         if( ${CMAKE_VERSION} VERSION_LESS 3.17 )
            message( STATUS "${_desc} - ${CMAKE_${lang}_COMPILER}" )
         else()
            message( ${_CHECK_COMPILER_STATUS} "${CMAKE_${lang}_COMPILER}" )
         endif()
         set( CMAKE_${lang}_COMPILER "${CMAKE_${lang}_COMPILER}" CACHE FILEPATH
            "${lang} compiler" )
         mark_as_advanced( CMAKE_${lang}_COMPILER )

      else()

         # Fall back on CMake's code.
         check_language( ${lang} )

      endif()
   endif()

endmacro( vecmem_check_language )
