# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

set(TRACCC_ALPAKA_TEST_SOURCES
  alpaka_basic.cpp
  test_cca.cpp
)


include(traccc-alpaka-functions)
traccc_enable_language_alpaka()

if(alpaka_ACC_GPU_CUDA_ENABLE)
  set_source_files_properties(${TRACCC_ALPAKA_TEST_SOURCES} PROPERTIES LANGUAGE CUDA)
  list(APPEND DEVICE_LIBRARIES vecmem::cuda)
elseif(alpaka_ACC_GPU_HIP_ENABLE)
  set_source_files_properties(${TRACCC_ALPAKA_TEST_SOURCES} PROPERTIES LANGUAGE HIP)
  list(APPEND DEVICE_LIBRARIES vecmem::hip)
elseif(alpaka_ACC_SYCL_ENABLE)
  list(APPEND DEVICE_LIBRARIES vecmem::sycl)
  set_source_files_properties(${TRACCC_ALPAKA_TEST_SOURCES} PROPERTIES LANGUAGE SYCL)
endif()

traccc_add_test( alpaka
   ${TRACCC_ALPAKA_TEST_SOURCES}
   LINK_LIBRARIES
   GTest::gtest_main
   traccc_tests_common
   alpaka::alpaka
   vecmem::core
   traccc::alpaka
   ${DEVICE_LIBRARIES}
)

#Can only do this once target is defined, so need another if here
if(alpaka_ACC_GPU_HIP_ENABLE)
  set_target_properties( traccc_test_alpaka PROPERTIES
     POSITION_INDEPENDENT_CODE TRUE )
endif()
