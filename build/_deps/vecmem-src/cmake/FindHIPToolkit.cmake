# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Decide whether AMD or NVidia code is generated using HIP.
set( CMAKE_HIP_PLATFORM_DEFAULT "hcc" )
if( NOT "$ENV{HIP_PLATFORM}" STREQUAL "" )
   set( CMAKE_HIP_PLATFORM_DEFAULT "$ENV{HIP_PLATFORM}" )
endif()
set( CMAKE_HIP_PLATFORM "${CMAKE_HIP_PLATFORM_DEFAULT}" CACHE STRING
   "Platform to build the HIP code for" )
set_property( CACHE CMAKE_HIP_PLATFORM
   PROPERTY STRINGS "hcc" "nvcc" "amd" "nvidia" )

# Set a helper variable.
set( _quietFlag )
if( HIPToolkit_FIND_QUIETLY )
   set( _quietFlag QUIET )
endif()

# Look for the CUDA toolkit if we are building NVidia code.
set( _requiredVars )
if( ( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvcc" ) OR
    ( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvidia" ) )
   find_package( CUDAToolkit ${_quietFlag} )
   list( APPEND _requiredVars CUDAToolkit_FOUND )
endif()

# Look for the ROCm/HIP header(s).
find_path( HIPToolkit_INCLUDE_DIR
   NAMES "hip/hip_runtime.h"
         "hip/hip_runtime_api.h"
   PATHS "${HIP_ROOT_DIR}"
         ENV ROCM_PATH
         ENV HIP_PATH
         "/opt/rocm"
         "/opt/rocm/hip"
   PATH_SUFFIXES "include"
   DOC "ROCm/HIP include directory" )
mark_as_advanced( HIPToolkit_INCLUDE_DIR )
set( HIPToolkit_INCLUDE_DIRS "${HIPToolkit_INCLUDE_DIR}" )
list( APPEND _requiredVars HIPToolkit_INCLUDE_DIR )

# Figure out the version of HIP.
find_file( HIPToolkit_VERSION_FILE
   NAMES "hip/hip_version.h"
   HINTS "${HIPToolkit_INCLUDE_DIR}"
   DOC "Path to hip/hip_version.h" )
mark_as_advanced( HIPToolkit_VERSION_FILE )
if( HIPToolkit_VERSION_FILE )
   file( READ "${HIPToolkit_VERSION_FILE}" _versionFileContents )
   if( "${_versionFileContents}" MATCHES "HIP_VERSION_MAJOR ([0-9]+)" )
      set( HIPToolkit_VERSION_MAJOR "${CMAKE_MATCH_1}" )
   endif()
   if( "${_versionFileContents}" MATCHES "HIP_VERSION_MINOR ([0-9]+)" )
      set( HIPToolkit_VERSION_MINOR "${CMAKE_MATCH_1}" )
   endif()
   if( "${_versionFileContents}" MATCHES "HIP_VERSION_PATCH ([0-9]+)" )
      set( HIPToolkit_VERSION_PATCH "${CMAKE_MATCH_1}" )
   endif()
   unset( _versionFileContents )
   set( HIPToolkit_VERSION
        "${HIPToolkit_VERSION_MAJOR}.${HIPToolkit_VERSION_MINOR}.${HIPToolkit_VERSION_PATCH}" )
endif()

# Look for the HIP runtime library.
set( HIPToolkit_LIBRARIES )
if( ( "${CMAKE_HIP_PLATFORM}" STREQUAL "hcc" ) OR
    ( "${CMAKE_HIP_PLATFORM}" STREQUAL "amd" ) )
   find_library( HIPToolkit_amdhip64_LIBRARY
      NAMES "amdhip64"
      PATHS "${HIP_ROOT_DIR}"
            ENV ROCM_PATH
            ENV HIP_PATH
            "/opt/rocm"
            "/opt/rocm/hip"
      PATH_SUFFIXES "lib" "lib64"
      DOC "AMD/HIP Runtime Library" )
   mark_as_advanced( HIPToolkit_amdhip64_LIBRARY )
   set( HIPToolkit_RUNTIME_LIBRARY "${HIPToolkit_amdhip64_LIBRARY}" )
   list( APPEND HIPToolkit_LIBRARIES "${HIPToolkit_amdhip64_LIBRARY}" )
   list( APPEND _requiredVars HIPToolkit_RUNTIME_LIBRARY )
elseif( ( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvcc" ) OR
        ( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvidia" ) )
   set( HIPToolkit_RUNTIME_LIBRARY "${CUDA_CUDART}" )
   list( APPEND HIPToolkit_LIBRARIES CUDA::cudart )
else()
   message( SEND_ERROR
      "Invalid CMAKE_HIP_PLATFORM setting (${CMAKE_HIP_PLATFORM}) received" )
endif()

# Set up the compiler definitions needed to use the HIP headers.
if( ( "${CMAKE_HIP_PLATFORM}" STREQUAL "hcc" ) OR
    ( "${CMAKE_HIP_PLATFORM}" STREQUAL "amd" ) )
   set( HIPToolkit_DEFINITIONS "__HIP_PLATFORM_HCC__"
                               "__HIP_PLATFORM_AMD__" )
elseif( ( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvcc" ) OR
        ( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvidia" ) )
   set( HIPToolkit_DEFINITIONS "__HIP_PLATFORM_NVCC__"
                               "__HIP_PLATFORM_NVIDIA__" )
else()
   message( SEND_ERROR "Invalid (CMAKE_)HIP_PLATFORM setting received" )
endif()

# Handle the standard find_package arguments.
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( HIPToolkit
   FOUND_VAR HIPToolkit_FOUND
   REQUIRED_VARS HIPToolkit_INCLUDE_DIR ${_requiredVars}
   VERSION_VAR HIPToolkit_VERSION )

# Set up the imported target(s).
if( NOT TARGET HIP::hiprt )
   add_library( HIP::hiprt UNKNOWN IMPORTED )
   set_target_properties( HIP::hiprt PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${HIPToolkit_INCLUDE_DIRS}"
      IMPORTED_LOCATION "${HIPToolkit_RUNTIME_LIBRARY}"
      INTERFACE_LINK_LIBRARIES "${HIPToolkit_LIBRARIES}"
      INTERFACE_COMPILE_DEFINITIONS "${HIPToolkit_DEFINITIONS}" )
endif()

# Clean up.
unset( _quietFlag )
unset( _requiredVars )
