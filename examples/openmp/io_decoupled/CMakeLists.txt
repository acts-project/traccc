add_executable(io_dec_par_example io_dec_par_example.cpp)
target_link_libraries(io_dec_par_example LINK_PUBLIC traccc::core traccc::io vecmem::core OpenMP::OpenMP_CXX)