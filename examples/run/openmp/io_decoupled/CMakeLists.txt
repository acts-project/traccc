add_executable(io_dec_par_example io_dec_par_example.cpp)
target_link_libraries(io_dec_par_example LINK_PUBLIC traccc::core traccc::io vecmem::core OpenMP::OpenMP_CXX)

add_test(NAME example_io_dec_par
         COMMAND io_dec_par_example tml_detector/trackml-detector.csv tml_pixels/ 10)
set_tests_properties(example_io_dec_par PROPERTIES ENVIRONMENT TRACCC_TEST_DATA_DIR=${PROJECT_SOURCE_DIR}/data/)
