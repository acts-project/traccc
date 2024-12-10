# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# The SYCL language code only works with the Ninja, and the different kinds
# of Makefile generators.
if( NOT ( ( "${CMAKE_GENERATOR}" MATCHES "Make" ) OR
          ( "${CMAKE_GENERATOR}" MATCHES "Ninja" ) ) )
   message( FATAL_ERROR "SYCL language not currently supported by "
      "\"${CMAKE_GENERATOR}\" generator" )
endif()

# Use the SYCLCXX environment variable preferably as the SYCL compiler.
if( NOT "$ENV{SYCLCXX}" STREQUAL "" )

   # Interpret the contents of SYCLCXX.
   get_filename_component( CMAKE_SYCL_COMPILER_INIT "$ENV{SYCLCXX}"
      PROGRAM PROGRAM_ARGS CMAKE_SYCL_FLAGS_INIT )
   if( NOT EXISTS ${CMAKE_SYCL_COMPILER_INIT} )
      message( FATAL_ERROR
         "Could not find compiler set in environment variable SYCLCXX:\n$ENV{SYCLCXX}.\n${CMAKE_SYCL_COMPILER_INIT}")
   endif()

   # Determine the type and version of the SYCL compiler.
   foreach( _version_cmdl "--acpp-version" "--version" "--hipsycl-version" )
      execute_process( COMMAND "${CMAKE_SYCL_COMPILER_INIT}" "${_version_cmdl}"
         OUTPUT_VARIABLE _syclVersionOutput
         ERROR_VARIABLE _syclVersionError
         RESULT_VARIABLE _syclVersionResult )
      if( ${_syclVersionResult} EQUAL 0 )
         break()
      endif()
   endforeach()
   if( ${_syclVersionResult} EQUAL 0 )
      if( "${_syclVersionOutput}" MATCHES "ComputeCpp" )
         set( CMAKE_SYCL_COMPILER_ID "ComputeCpp" CACHE STRING
            "Identifier for the SYCL compiler in use" )
         set( _syclVersionRegex "([0-9\.]+) Device Compiler" )
      elseif( "${_syclVersionOutput}" MATCHES "oneAPI" )
         set( CMAKE_SYCL_COMPILER_ID "IntelLLVM" CACHE STRING
            "Identifier for the SYCL compiler in use" )
         set( _syclVersionRegex "DPC\\\+\\\+.*Compiler ([0-9\.]+)" )
      elseif( "${_syclVersionOutput}" MATCHES "intel/llvm" )
         set( CMAKE_SYCL_COMPILER_ID "IntelLLVM" CACHE STRING
            "Identifier for the SYCL compiler in use" )
         set( _syclVersionRegex "clang version ([0-9\.]+)" )
      elseif( "${_syclVersionOutput}" MATCHES "AdaptiveCpp" )
         set( CMAKE_SYCL_COMPILER_ID "AdaptiveCpp" CACHE STRING
            "Identifier for the SYCL compiler in use" )
         set( _syclVersionRegex "AdaptiveCpp version: ([0-9\.]+)" )
      elseif( "${_syclVersionOutput}" MATCHES "hipSYCL" )
         set( CMAKE_SYCL_COMPILER_ID "hipSYCL" CACHE STRING
            "Identifier for the SYCL compiler in use" )
         set( _syclVersionRegex "hipSYCL version: ([0-9\.]+)" )
      else()
         set( CMAKE_SYCL_COMPILER_ID "Unknown" CACHE STRING
            "Identifier for the SYCL compiler in use" )
         set( _syclVersionRegex "[a-zA-Z]+ version ([0-9\.]+)" )
      endif()
      string( REPLACE "\n" ";" _syclVersionOutputList "${_syclVersionOutput}" )
      foreach( _line ${_syclVersionOutputList} )
         if( _line MATCHES "${_syclVersionRegex}" )
            set( CMAKE_SYCL_COMPILER_VERSION "${CMAKE_MATCH_1}" )
            break()
         endif()
      endforeach()
      unset( _syclVersionOutputList )
      unset( _syclVersionRegex )
   else()
      message( WARNING
         "Failed to execute: ${CMAKE_SYCL_COMPILER_INIT} --version" )
      set( CMAKE_SYCL_COMPILER_VERSION "Unknown" )
   endif()
   unset( _syclVersionOutput )
   unset( _syclVersionResult )
else()
   # If not specified otherwise, try to use the C++ compiler.
   set( CMAKE_SYCL_COMPILER_INIT "${CMAKE_CXX_COMPILER}" )
   set( CMAKE_SYCL_FLAGS_INIT "-fsycl" )
   set( CMAKE_SYCL_COMPILER_VERSION "${CMAKE_CXX_COMPILER_VERSION}" )
endif()

# Set up the compiler with cache variables.
set( CMAKE_SYCL_COMPILER "${CMAKE_SYCL_COMPILER_INIT}" CACHE FILEPATH
   "The SYCL compiler to use. Normally the same as the C++ compiler." )

# Tell the user what was found for the SYCL compiler.
message( STATUS "The SYCL compiler identification is "
   "${CMAKE_SYCL_COMPILER_ID} ${CMAKE_SYCL_COMPILER_VERSION}" )

# Set up what source/object file names to use.
set( CMAKE_SYCL_SOURCE_FILE_EXTENSIONS "sycl" )
if( UNIX )
   set( CMAKE_SYCL_OUTPUT_EXTENSION ".o" )
else()
   set( CMAKE_SYCL_OUTPUT_EXTENSION ".obj" )
endif()
set( CMAKE_SYCL_COMPILER_ENV_VAR "SYCLCXX" )

# Set up the linker used for components holding SYCL source code.
set( CMAKE_SYCL_HOST_LINKER "${CMAKE_SYCL_COMPILER}" )

# Set up C++17 by default.
set( CMAKE_SYCL_STANDARD 17 CACHE STRING "C++ standard to use with SYCL" )
set_property( CACHE CMAKE_SYCL_STANDARD PROPERTY STRINGS 17 20 23 )

# Configure variables set in this file for fast reload later on.
configure_file( ${CMAKE_CURRENT_LIST_DIR}/CMakeSYCLCompiler.cmake.in
   ${CMAKE_PLATFORM_INFO_DIR}/CMakeSYCLCompiler.cmake )
