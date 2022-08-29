# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

traccc_add_test(
    futhark

    test_cca.cpp

    LINK_LIBRARIES
    GTest::gtest
    GTest::gtest_main
    traccc::core
    traccc::io
    traccc::futhark
    traccc_tests_common
)
