add_executable (par_example par_example.cpp)
target_link_libraries (par_example LINK_PUBLIC traccc::core traccc::io vecmem::core OpenMP::OpenMP_CXX)
add_test(example_par ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/par_example
		     tml_detector/trackml-detector.csv tml_pixels/ 10)
set_tests_properties(example_par PROPERTIES ENVIRONMENT TRACCC_TEST_DATA_DIR=${PROJECT_SOURCE_DIR}/data/)

add_subdirectory(io_decoupled)