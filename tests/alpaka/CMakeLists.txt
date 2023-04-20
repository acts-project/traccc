# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

if(alpaka_ACC_GPU_CUDA_ENABLE)
  enable_language(CUDA)
  set_source_files_properties(alpaka_basic.cpp PROPERTIES LANGUAGE CUDA)
endif()

traccc_add_test( alpaka
   alpaka_basic.cpp
   LINK_LIBRARIES
   GTest::gtest_main
   alpaka::alpaka
)
