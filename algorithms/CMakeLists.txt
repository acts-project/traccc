add_library(traccc_algorithms INTERFACE)

add_subdirectory(cpu)
target_link_libraries(traccc_algorithms INTERFACE traccc_cpu)

if(TRACCC_BUILD_CUDA)
  add_subdirectory(cuda)
  target_link_libraries(traccc_algorithms INTERFACE traccc_cuda)
endif()
