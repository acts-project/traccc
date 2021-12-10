traccc_add_test(
    io

    # Define the sources for the test.
    test_csv.cpp

    LINK_LIBRARIES
    traccc::testing traccc::core traccc::io gtest_main gtest
)
