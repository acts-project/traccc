# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( vecmem-compiler-options-cpp )

# Test all of the core library's features.
vecmem_add_test( core
   "test_core_allocator.cpp"
   "test_core_array.cpp"
   "test_core_atomic_ref.cpp"
   "test_core_containers.cpp"
   "test_core_contiguous_memory_resource.cpp" "test_core_copy.cpp"
   "test_core_device_containers.cpp" "test_core_memory_resources.cpp"
   "test_core_static_vector.cpp" "test_core_vector.cpp"
   "test_core_jagged_vector_view.cpp" "test_core_static_array.cpp" "test_core_default_resource.cpp"
   "test_core_instrumenting_memory_resource.cpp"
   "test_core_terminal_memory_resource.cpp"
   "test_core_conditional_memory_resource.cpp"
   "test_core_choice_memory_resource.cpp"
   "test_core_coalescing_memory_resource.cpp"
   "test_core_debug_memory_resource.cpp"
   "test_core_unique_alloc_ptr.cpp"
   "test_core_unique_obj_ptr.cpp"
   "test_core_tuple.cpp"
   "test_core_edm_buffer.cpp"
   "test_core_edm_device.cpp"
   "test_core_edm_host.cpp"
   "test_core_edm_view.cpp"
   LINK_LIBRARIES vecmem::core GTest::gtest_main vecmem_testing_common )

# Add UBSAN for the tests, if it's available.
include( CheckCXXCompilerFlag )
check_cxx_compiler_flag( "-fsanitize=undefined" VECMEM_HAVE_UBSAN )
if( VECMEM_HAVE_UBSAN AND VECMEM_TEST_UBSAN )
   target_compile_options( vecmem_test_core PRIVATE "-fsanitize=undefined" )
   target_link_options( vecmem_test_core PRIVATE "-fsanitize=undefined" )
endif()
