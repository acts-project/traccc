add_library(traccc_cuda SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/clusterization/component_connection_kernels.cu)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

target_include_directories(traccc_cuda
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

install(
  DIRECTORY include
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

target_link_libraries(traccc_cuda PUBLIC vecmem::core vecmem::cuda traccc_core)
