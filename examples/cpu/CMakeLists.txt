add_executable (seq_single_module seq_single_module.cpp)
target_link_libraries (seq_single_module LINK_PUBLIC traccc::core vecmem::core)

add_executable (seq_example seq_example.cpp)
target_link_libraries (seq_example LINK_PUBLIC traccc::core traccc::io vecmem::core)

add_executable (ccl_example ccl_example.cpp)
target_link_libraries (ccl_example LINK_PUBLIC traccc::core traccc::io)

# find OpenMP
find_package(OpenMP REQUIRED)
set(GCC_OMP "-fopenmp")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_OMP}")
add_executable (par_example par_example.cpp)
target_link_libraries (par_example LINK_PUBLIC traccc::core traccc::io vecmem::core OpenMP::OpenMP_CXX)
