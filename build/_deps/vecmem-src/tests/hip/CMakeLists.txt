# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Enable HIP as a language.
enable_language( HIP )

# Project include(s).
include( vecmem-compiler-options-cpp )
include( vecmem-compiler-options-hip )

# External dependency/dependencies.
find_package( HIPToolkit REQUIRED )

# Test all of the HIP library's features.
vecmem_add_test( hip
   "test_hip_memory_resources.cpp"
   "test_hip_containers.cpp" "test_hip_jagged_containers.cpp"
   "test_hip_containers_kernels.hpp" "test_hip_containers_kernels.hip"
   "test_hip_edm.cpp"
   "test_hip_edm_kernels.hpp" "test_hip_edm_kernels.hip"
   "test_hip_copy.cpp"
   "${CMAKE_CURRENT_SOURCE_DIR}/../../hip/src/utils/hip_error_handling.hpp"
   "${CMAKE_CURRENT_SOURCE_DIR}/../../hip/src/utils/hip_error_handling.cpp"
   LINK_LIBRARIES HIP::hiprt vecmem::core vecmem::hip GTest::gtest_main
                  vecmem_testing_common )

# Add UBSAN for the tests, if it's available.
include( CheckCXXCompilerFlag )
check_cxx_compiler_flag( "-fsanitize=undefined" VECMEM_HAVE_UBSAN )
if( VECMEM_HAVE_UBSAN AND VECMEM_TEST_UBSAN )
   target_compile_options( vecmem_test_hip PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=undefined> )
   target_link_options( vecmem_test_hip PRIVATE "-fsanitize=undefined" )
endif()

# The executable's source code needs to be built into position independent
# code in certain HIP + GCC combinations. But not most of them... Still, for
# the test executable let's just take the hit, and always use PIC.
set_target_properties( vecmem_test_hip PROPERTIES
   POSITION_INDEPENDENT_CODE TRUE )
