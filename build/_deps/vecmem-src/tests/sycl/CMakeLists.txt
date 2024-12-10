# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Enable SYCL as a language.
enable_language( SYCL )

# Project include(s).
include( vecmem-compiler-options-cpp )
include( vecmem-compiler-options-sycl )
include( vecmem-check-sycl-source-compiles )

# CMake include(s).
include( CheckCXXCompilerFlag )

# Test all of the SYCL library's features.
vecmem_add_test( sycl
   "test_sycl_memory_resources.cpp"
   "test_sycl_containers.sycl" "test_sycl_jagged_containers.sycl"
   "test_sycl_edm.sycl"
   "test_sycl_copy.cpp"
   LINK_LIBRARIES vecmem::core vecmem::sycl GTest::gtest_main
                  vecmem_testing_common )

# Add UBSAN for the tests, if it's available.
check_cxx_compiler_flag( "-fsanitize=undefined" VECMEM_HAVE_UBSAN )
vecmem_check_sycl_source_compiles( "
   #include <sycl/sycl.hpp>
   int main() { return 0; }
   " VECMEM_HAVE_SYCL_UBSAN
   CMAKE_FLAGS -DCOMPILE_DEFINITIONS=-fsanitize=undefined )
if( VECMEM_HAVE_UBSAN AND VECMEM_HAVE_SYCL_UBSAN AND VECMEM_TEST_UBSAN )
   target_compile_options( vecmem_test_sycl PRIVATE "-fsanitize=undefined" )
   target_link_options( vecmem_test_sycl PRIVATE "-fsanitize=undefined" )
endif()
