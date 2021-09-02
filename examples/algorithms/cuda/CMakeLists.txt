add_library(traccc_algorithms_cuda INTERFACE)

target_include_directories(traccc_algorithms_cuda
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(traccc_algorithms_cuda
  INTERFACE traccc::io traccc::core traccc::cuda)

add_library(traccc::algorithms_cuda ALIAS traccc_algorithms_cuda)
