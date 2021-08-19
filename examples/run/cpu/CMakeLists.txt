add_executable (seq_single_module seq_single_module.cpp)
target_link_libraries (seq_single_module LINK_PUBLIC traccc::io traccc::algorithms)
add_test(NAME example_seq_single_module
         COMMAND seq_single_module)

add_executable (seq_example seq_example.cpp)
target_link_libraries (seq_example LINK_PUBLIC traccc::io traccc::algorithms)
add_test(NAME example_seq
         COMMAND seq_example tml_detector/trackml-detector.csv tml_pixels/ 10)
set_tests_properties(example_seq PROPERTIES ENVIRONMENT TRACCC_TEST_DATA_DIR=${PROJECT_SOURCE_DIR}/data/)

add_executable (ccl_example ccl_example.cpp)
target_link_libraries (ccl_example LINK_PUBLIC traccc::io traccc::core)
add_test(NAME example_ccl
         COMMAND ccl_example ${PROJECT_SOURCE_DIR}/data/tml_pixels/event000000000-cells.csv)
