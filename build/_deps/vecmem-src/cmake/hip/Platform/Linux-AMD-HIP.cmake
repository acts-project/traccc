# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Use the standard GNU compiler options for AMD's HIP.
include( Platform/Linux-GNU )
__linux_compiler_gnu( HIP )
include( Compiler/GNU )
__compiler_gnu( HIP )

# Set up the dependency file generation for this platform. Note that HIP
# compilation only works with Makefile and Ninja generators, so no check is made
# here for the current generator.
set( CMAKE_HIP_DEPENDS_USE_COMPILER TRUE )
set( CMAKE_HIP_DEPFILE_FORMAT gcc )

# Decide how to do the build for the AMD (hcc) and NVidia (nvcc) backends.
set( CMAKE_INCLUDE_FLAG_HIP "-I" )
set( CMAKE_INCLUDE_SYSTEM_FLAG_HIP "-isystem " )
set( CMAKE_HIP_STANDARD_LIBRARIES "${HIPToolkit_RUNTIME_LIBRARY}" )
if( ( "${CMAKE_HIP_PLATFORM}" STREQUAL "hcc" ) OR
    ( "${CMAKE_HIP_PLATFORM}" STREQUAL "amd" ) )

   if( "${CMAKE_HIP_COMPILER_VERSION}" VERSION_LESS "3.7" )
      set( CMAKE_HIP_COMPILE_SOURCE_TYPE_FLAG "-x c++" )
   else()
      set( CMAKE_HIP_COMPILE_SOURCE_TYPE_FLAG "" )
   endif()
   set( CMAKE_HIP_STANDARD_INCLUDE_DIRECTORIES ${HIPToolkit_INCLUDE_DIRS} )

elseif( ( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvcc" ) OR
        ( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvidia" ) )

   include( Compiler/NVIDIA-CUDA )

   set( CMAKE_HIP_COMPILE_SOURCE_TYPE_FLAG "-x cu" )
   set( CMAKE_HIP_STANDARD_INCLUDE_DIRECTORIES ${HIPToolkit_INCLUDE_DIRS}
                                               ${CUDAToolkit_INCLUDE_DIRS} )
   set( CMAKE_HIP_COMPILE_OPTIONS_PIC "${CMAKE_CUDA_COMPILE_OPTIONS_PIC}" )
   set( CMAKE_HIP_COMPILE_OPTIONS_PIE "${CMAKE_CUDA_COMPILE_OPTIONS_PIE}" )
   set( CMAKE_HIP_COMPILE_OPTIONS_VISIBILITY
      "${CMAKE_CUDA_COMPILE_OPTIONS_VISIBILITY}" )
   set( CMAKE_HIP_FLAGS_DEBUG_INIT "${CMAKE_CUDA_FLAGS_DEBUG_INIT}" )
   set( CMAKE_HIP_FLAGS_RELEASE_INIT "${CMAKE_CUDA_FLAGS_RELEASE_INIT}" )
   set( CMAKE_HIP_FLAGS_RELWITHDEBINFO_INIT
      "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO_INIT}" )

else()
   message( FATAL_ERROR
      "Invalid setting for CMAKE_HIP_PLATFORM (\"${CMAKE_HIP_PLATFORM}\").\n" )
endif()

# Set a compiler command from scratch for this platform.
set( CMAKE_HIP_COMPILE_OBJECT
   "HIP_PLATFORM=${CMAKE_HIP_PLATFORM} <CMAKE_HIP_COMPILER> <DEFINES> <INCLUDES> <FLAGS> ${CMAKE_HIP_COMPILE_SOURCE_TYPE_FLAG} -o <OBJECT> -c <SOURCE>" )

# Set the archive (static library) creation command explicitly for this platform.
set( CMAKE_HIP_CREATE_STATIC_LIBRARY
   "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>" )

# Set the flags controlling the C++ standard used by the HIP compiler.
set( CMAKE_HIP11_STANDARD_COMPILE_OPTION "-std=c++11" )
set( CMAKE_HIP11_EXTENSION_COMPILE_OPTION "-std=c++11" )

set( CMAKE_HIP14_STANDARD_COMPILE_OPTION "-std=c++14" )
set( CMAKE_HIP14_EXTENSION_COMPILE_OPTION "-std=c++14" )

set( CMAKE_HIP17_STANDARD_COMPILE_OPTION "-std=c++17" )
set( CMAKE_HIP17_EXTENSION_COMPILE_OPTION "-std=c++17" )

set( CMAKE_HIP20_STANDARD_COMPILE_OPTION "-std=c++20" )
set( CMAKE_HIP20_EXTENSION_COMPILE_OPTION "-std=c++20" )

set( CMAKE_HIP23_STANDARD_COMPILE_OPTION "-std=c++23" )
set( CMAKE_HIP23_EXTENSION_COMPILE_OPTION "-std=c++23" )
