add_subdirectory(cpu)
if (TRACCC_BUILD_CUDA)
  add_subdirectory(cuda)
endif()
