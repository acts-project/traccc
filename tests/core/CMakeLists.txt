traccc_add_test(
    core

    # Define the sources for the test.
    test_algorithm.cpp
    test_module_map.cpp

    LINK_LIBRARIES
    traccc::testing traccc::core traccc::io gtest_main gtest
)
