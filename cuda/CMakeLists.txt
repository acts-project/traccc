add_library(traccc_cuda SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/spacepoint_formation/component_connection_kernels.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/spacepoint_formation/spacepoint_formation_kernels.cu)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

target_include_directories(traccc_cuda
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<INSTALL_INTERFACE:$include>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

install(
  DIRECTORY include/edm
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cuda)

install(
  DIRECTORY include/algorithms
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cuda)

install(
  DIRECTORY include/utils
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cuda)

target_link_libraries(traccc_cuda PUBLIC vecmem::core vecmem::cuda traccc::core)

add_library(traccc::cuda ALIAS traccc_cuda)
