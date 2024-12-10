# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CUDAToolkit requires CMake 3.17.
cmake_minimum_required( VERSION 3.17 )

# Project include(s).
include( vecmem-compiler-options-cpp )

# External dependency/dependencies.
find_package( CUDAToolkit REQUIRED )

# Set up the build of the VecMem CUDA library.
vecmem_add_library( vecmem_cuda cuda
   # Memory resources.
   "include/vecmem/memory/cuda/device_memory_resource.hpp"
   "src/memory/device_memory_resource.cpp"
   "include/vecmem/memory/cuda/host_memory_resource.hpp"
   "src/memory/host_memory_resource.cpp"
   "include/vecmem/memory/cuda/managed_memory_resource.hpp"
   "src/memory/managed_memory_resource.cpp"
   # Utilities.
   "include/vecmem/utils/cuda/copy.hpp"
   "src/utils/cuda/copy.cpp"
   "include/vecmem/utils/cuda/async_copy.hpp"
   "src/utils/cuda/async_copy.cpp"
   "src/utils/cuda_error_handling.hpp"
   "src/utils/cuda_error_handling.cpp"
   "src/utils/cuda_wrappers.hpp"
   "src/utils/cuda_wrappers.cpp"
   "src/utils/get_device_name.hpp"
   "src/utils/get_device_name.cpp"
   "src/utils/select_device.hpp"
   "src/utils/select_device.cpp"
   "include/vecmem/utils/cuda/stream_wrapper.hpp"
   "src/utils/stream_wrapper.cpp"
   "src/utils/opaque_stream.hpp"
   "src/utils/opaque_stream.cpp" )
target_link_libraries( vecmem_cuda
   PUBLIC vecmem::core
   PRIVATE CUDA::cudart )

# Hide the library's symbols by default.
set_target_properties( vecmem_cuda PROPERTIES
   CXX_VISIBILITY_PRESET  "hidden" )
include( CheckCXXCompilerFlag )
check_cxx_compiler_flag(
   "-fvisibility-ms-compat" VECMEM_HAVE_VISIBILITY_MS_COMPAT )
if( VECMEM_HAVE_VISIBILITY_MS_COMPAT )
   target_compile_options( vecmem_cuda PRIVATE "-fvisibility-ms-compat" )
endif()

# Set up whether asynchronous synchronization errors should be fatal.
if( VECMEM_FAIL_ON_ASYNC_ERRORS )
   target_compile_definitions( vecmem_cuda PRIVATE VECMEM_FAIL_ON_ASYNC_ERRORS )
endif()

# Test the public headers of vecmem::cuda.
if( BUILD_TESTING AND VECMEM_BUILD_TESTING )
   file( GLOB_RECURSE vecmem_cuda_public_headers
      RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "include/*/*.hpp" )
   vecmem_test_public_headers( vecmem_cuda ${vecmem_cuda_public_headers} )
endif()
