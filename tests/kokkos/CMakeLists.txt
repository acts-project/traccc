# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

add_library(
  traccc_tests_kokkos_main
  STATIC
  kokkos_main.cpp
)

target_link_libraries(
  traccc_tests_kokkos_main
  PRIVATE
  GTest::gtest
)

traccc_add_test(
  kokkos

  kokkos_basic.cpp

  LINK_LIBRARIES
  GTest::gtest
  traccc_test_kokkos_main
)
