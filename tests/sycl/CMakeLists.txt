# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

include(traccc-compiler-options-sycl)

enable_language(SYCL)

traccc_add_test(
    sycl

    # Define the sources for the test.
    test_kalman_filter.sycl

    LINK_LIBRARIES
    GTest::gtest_main
    vecmem::sycl
    detray::core
    detray::utils
    traccc::core
    traccc::device_common
    traccc::sycl
    traccc::performance
    traccc::io
    traccc_tests_common
)
