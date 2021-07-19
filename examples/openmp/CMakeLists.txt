add_executable (par_example par_example.cpp)
target_link_libraries (par_example LINK_PUBLIC traccc::core traccc::io vecmem::core OpenMP::OpenMP_CXX)
