traccc_add_test(
    cpu

    # Define the sources for the test.
    test_component_connection.cpp
    test_cca.cpp

    LINK_LIBRARIES
    traccc::testing traccc::core traccc::io gtest_main gtest
)
