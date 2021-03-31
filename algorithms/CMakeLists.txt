add_library(traccc_algorithms SHARED ${CMAKE_CURRENT_SOURCE_DIR}/cuda/src/clusterization/component_connection_kernels.cu)

if(TRACCC_BUILD_CUDA)
  find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

target_include_directories(traccc_algorithms
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpu/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cuda/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

install(
  DIRECTORY cpu/include/algorithms
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  DIRECTORY cuda/include/algorithms
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

target_link_libraries(traccc_algorithms PUBLIC vecmem::core vecmem::cuda traccc_core)
