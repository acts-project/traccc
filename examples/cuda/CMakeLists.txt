add_executable (seq_example_cuda seq_example_cuda.cpp)
target_link_libraries (seq_example_cuda LINK_PUBLIC traccc::core traccc::cpu traccc::cuda traccc::io)

