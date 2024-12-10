# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( vecmem-compiler-options-cpp )

# External dependency/dependencies.
find_package( HIPToolkit REQUIRED )

# Set up the build of the VecMem HIP library.
vecmem_add_library( vecmem_hip hip
   # Memory management.
   "include/vecmem/memory/hip/device_memory_resource.hpp"
   "src/memory/device_memory_resource.cpp"
   "include/vecmem/memory/hip/host_memory_resource.hpp"
   "src/memory/host_memory_resource.cpp"
   "include/vecmem/memory/hip/managed_memory_resource.hpp"
   "src/memory/managed_memory_resource.cpp"
   # Utilities.
   "include/vecmem/utils/hip/copy.hpp"
   "src/utils/hip/copy.cpp"
   "src/utils/get_device.hpp"
   "src/utils/get_device.cpp"
   "src/utils/hip_error_handling.hpp"
   "src/utils/hip_error_handling.cpp"
   "src/utils/run_on_device.hpp"
   "src/utils/select_device.hpp"
   "src/utils/select_device.cpp" )
target_link_libraries( vecmem_hip
   PUBLIC vecmem::core
   PRIVATE HIP::hiprt )

# Hide the library's symbols by default.
set_target_properties( vecmem_hip PROPERTIES
   CXX_VISIBILITY_PRESET "hidden" )
include( CheckCXXCompilerFlag )
check_cxx_compiler_flag(
   "-fvisibility-ms-compat" VECMEM_HAVE_VISIBILITY_MS_COMPAT )
if( VECMEM_HAVE_VISIBILITY_MS_COMPAT )
   target_compile_options( vecmem_hip PRIVATE "-fvisibility-ms-compat" )
endif()

# Test the public headers of vecmem::hip.
if( BUILD_TESTING AND VECMEM_BUILD_TESTING )
   file( GLOB_RECURSE vecmem_hip_public_headers
      RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "include/*/*.hpp" )
   vecmem_test_public_headers( vecmem_hip ${vecmem_hip_public_headers} )
endif()
