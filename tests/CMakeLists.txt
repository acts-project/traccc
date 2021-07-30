enable_testing()

macro(add_traccc_test TESTNAME FILES PLUGIN_LIBRARY)
    add_executable(${TESTNAME} ${FILES})
    target_link_libraries(${TESTNAME} gtest gmock gtest_main)
    target_link_libraries(${TESTNAME} traccc::core)
    target_link_libraries(${TESTNAME} traccc::tests::common)
    target_link_libraries(${TESTNAME} ${PLUGIN_LIBRARY})
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
    add_test(unit_test_${TESTNAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TESTNAME})
    set_tests_properties(unit_test_${TESTNAME} PROPERTIES
    ENVIRONMENT TRACCC_TEST_DATA_DIR=${PROJECT_SOURCE_DIR}/data/)
endmacro()

add_library(traccc_tests_common INTERFACE)

target_include_directories(
    traccc_tests_common
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

add_library(traccc::tests::common ALIAS traccc_tests_common)

add_subdirectory(cpu)
