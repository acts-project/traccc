# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# The HIP language code only works with the Ninja, and the different kinds
# of Makefile generators.
if( NOT ( ( "${CMAKE_GENERATOR}" MATCHES "Make" ) OR
          ( "${CMAKE_GENERATOR}" MATCHES "Ninja" ) ) )
   message( FATAL_ERROR "HIP language not currently supported by "
      "\"${CMAKE_GENERATOR}\" generator" )
endif()

# Use the HIPCXX environment variable preferably as the HIP compiler.
if( NOT "$ENV{HIPCXX}" STREQUAL "" )
   # Interpret the contents of HIPCXX.
   get_filename_component( CMAKE_HIP_COMPILER_INIT "$ENV{HIPCXX}"
      PROGRAM PROGRAM_ARGS CMAKE_HIP_FLAGS_ENV_INIT )
   if( NOT EXISTS ${CMAKE_HIP_COMPILER_INIT} )
      message( FATAL_ERROR
         "Could not find compiler set in environment variable HIPCXX:\n$ENV{HIPCXX}.\n${CMAKE_HIP_COMPILER_INIT}")
   endif()
else()
   # Find the HIP compiler.
   find_program( CMAKE_HIP_COMPILER_INIT NAMES "hipcc"
      PATHS "${HIP_ROOT_DIR}"
            ENV ROCM_PATH
            ENV HIP_PATH
            "/opt/rocm"
            "/opt/rocm/hip"
      PATH_SUFFIXES "bin" )
   set( CMAKE_HIP_FLAGS_ENV_INIT "" )
endif()
if( CMAKE_HIP_COMPILER_INIT )
   # Determine the type and version of the SYCL compiler.
   execute_process( COMMAND "${CMAKE_HIP_COMPILER_INIT}" "--version"
      OUTPUT_VARIABLE _hipVersionOutput
      ERROR_VARIABLE _hipVersionError
      RESULT_VARIABLE _hipVersionResult )
   if( ${_hipVersionResult} EQUAL 0 )
      if( "${_hipVersionOutput}" MATCHES "HIP version:" )
         set( CMAKE_HIP_COMPILER_ID "AMD" CACHE STRING
            "Identifier for the HIP compiler in use" )
         set( _hipVersionRegex "HIP version: ([0-9\.]+)" )
      else()
         set( CMAKE_HIP_COMPILER_ID "Unknown" CACHE STRING
            "Identifier for the HIP compiler in use" )
         set( _hipVersionRegex "[a-zA-Z]+ version ([0-9\.]+)" )
      endif()
      string( REPLACE "\n" ";" _hipVersionOutputList "${_hipVersionOutput}" )
      foreach( _line ${_hipVersionOutputList} )
         if( _line MATCHES "${_hipVersionRegex}" )
            set( CMAKE_HIP_COMPILER_VERSION "${CMAKE_MATCH_1}" )
            break()
         endif()
      endforeach()
      unset( _hipVersionOutputList )
      unset( _hipVersionRegex )
   else()
      message( WARNING
         "Failed to execute: ${CMAKE_HIP_COMPILER_INIT} --version" )
      set( CMAKE_HIP_COMPILER_VERSION "Unknown" )
   endif()
   unset( _hipVersionOutput )
   unset( _hipVersionResult )
endif()

# Set up the compiler as a cache variable.
set( CMAKE_HIP_COMPILER "${CMAKE_HIP_COMPILER_INIT}" CACHE FILEPATH
   "The HIP compiler to use" )

# Tell the user what was found for the HIP compiler.
message( STATUS "The HIP compiler identification is "
   "${CMAKE_HIP_COMPILER_ID} ${CMAKE_HIP_COMPILER_VERSION}" )

# Set up what source/object file names to use.
set( CMAKE_HIP_SOURCE_FILE_EXTENSIONS "hip" )
set( CMAKE_HIP_OUTPUT_EXTENSION ".o" )
set( CMAKE_HIP_COMPILER_ENV_VAR "HIPCXX" )

# Set up the linker used for components holding HIP source code.
set( CMAKE_HIP_HOST_LINKER "${CMAKE_CXX_COMPILER}" )

# Decide whether to generate AMD or NVidia code using HIP.
set( CMAKE_HIP_PLATFORM_DEFAULT "hcc" )
if( NOT "$ENV{HIP_PLATFORM}" STREQUAL "" )
   set( CMAKE_HIP_PLATFORM_DEFAULT "$ENV{HIP_PLATFORM}" )
endif()
set( CMAKE_HIP_PLATFORM "${CMAKE_HIP_PLATFORM_DEFAULT}" CACHE STRING
   "Platform to build all HIP code for in the project" )
set_property( CACHE CMAKE_HIP_PLATFORM
   PROPERTY STRINGS "hcc" "nvcc" "amd" "nvidia" )

# Set up C++14 by default for HIP.
set( CMAKE_HIP_STANDARD 14 CACHE STRING "C++ standard to use with HIP" )
set_property( CACHE CMAKE_HIP_STANDARD PROPERTY STRINGS 11 14 17 20 23 )

# Look for the HIP toolkit. Its variables are needed for setting up the build
# of HIP source files.
find_package( HIPToolkit REQUIRED QUIET )

# Configure variables set in this file for fast reload later on.
configure_file( ${CMAKE_CURRENT_LIST_DIR}/CMakeHIPCompiler.cmake.in
   ${CMAKE_PLATFORM_INFO_DIR}/CMakeHIPCompiler.cmake )
