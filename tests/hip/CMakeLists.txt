enable_language(HIP)
traccc_add_test(
    hip 
    # Define the sources for the test.
    test_thrust.hip
    LINK_LIBRARIES
    rocthrust
    GTest::gtest_main
    vecmem::core
    vecmem::hip
)
