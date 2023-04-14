# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Declare the core library test(s).
traccc_add_test( core "test_algorithm.cpp" "test_module_map.cpp" "test_options.cpp"
   LINK_LIBRARIES GTest::gtest_main traccc_tests_common
                  traccc::core traccc::io traccc::options )
