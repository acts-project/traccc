enable_language(CUDA)

add_executable (seq_example_cuda seq_example_cuda.cpp)
add_executable (seeding_example_cuda seeding_example_cuda.cpp)
add_executable (bfield_cuda bfield_cuda.cu)

target_link_libraries (seq_example_cuda LINK_PUBLIC
  traccc::io
  traccc::algorithms
  traccc::algorithms_cuda)

target_link_libraries (seeding_example_cuda LINK_PUBLIC
  traccc::io
  traccc::algorithms
  traccc::algorithms_cuda)

target_link_libraries (bfield_cuda LINK_PUBLIC
  traccc::cuda::utils)
