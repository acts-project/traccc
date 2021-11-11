function(traccc_add_test name)
    # Parse the function's options.
    cmake_parse_arguments(ARG "" "" "LINK_LIBRARIES" ${ARGN})

    # Create the test executable.
    set(TEST_NAME "test_${name}")
    add_executable(${TEST_NAME} ${ARG_UNPARSED_ARGUMENTS})

    if(ARG_LINK_LIBRARIES)
        target_link_libraries(${TEST_NAME} PRIVATE ${ARG_LINK_LIBRARIES})
    endif()

    # Run the executable as a test.
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

    # Make sure this test knows where to find its data.
    set_tests_properties(
        ${TEST_NAME}
        PROPERTIES
        ENVIRONMENT TRACCC_TEST_DATA_DIR=${PROJECT_SOURCE_DIR}/data/
    )
endfunction(traccc_add_test)

add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(cpu)
add_subdirectory(io)
