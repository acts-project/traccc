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
    test_kalman_fitter_telescope.sycl
    test_clusterization.sycl
    test_spacepoint_formation.sycl
    test_barrier.sycl

    LINK_LIBRARIES
    GTest::gtest_main
    vecmem::sycl
    detray::core
    detray::io
    detray::utils
    traccc::core
    traccc::device_common
    traccc::sycl
    traccc::performance
    traccc::io
    traccc::simulation
    traccc_tests_common
)
