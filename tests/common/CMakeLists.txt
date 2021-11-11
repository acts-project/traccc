add_library(traccc_tests_common INTERFACE)

target_include_directories(
    traccc_tests_common
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

add_library(traccc::testing ALIAS traccc_tests_common)
