# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Declare the cpu algorithm test(s).
traccc_add_test(cpu
    "compare_with_acts_seeding.cpp"
    "seq_single_module.cpp"
    "test_cca.cpp"
    "test_ckf_sparse_tracks.cpp"
    "test_clusterization_resolution.cpp"
    "test_copy.cpp"
    "test_kalman_fitter.cpp"
    "test_ranges.cpp"
    "test_seeding.cpp"
    "test_track_params_estimation.cpp"
    LINK_LIBRARIES GTest::gtest_main vecmem::core 
    traccc_tests_common traccc::core traccc::io traccc::performance
    detray::core detray::utils covfie::core )
