traccc_add_test(
    cpu

    # Define the sources for the test.
    compare_with_acts_seeding.cpp
    test_component_connection.cpp
    test_cca.cpp

    LINK_LIBRARIES
    traccc::testing traccc::core traccc::algorithms traccc::io gtest_main gtest
)
