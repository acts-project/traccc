add_library(traccc_core INTERFACE)

if(TRACCC_BUILD_CUDA)
  find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

target_include_directories(traccc_core
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

install(
  DIRECTORY include/geometry
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  DIRECTORY include/definitions
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  DIRECTORY include/edm
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_library(traccc::core ALIAS traccc_core)

#install(TARGETS traccc_core
#        PUBLIC_HEADER
#          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/core)

#install(
#  DIRECTORY common/include/geometry
#  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#install(
#  DIRECTORY common/include/definitions
#  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#install(
#  DIRECTORY common/include/edm
#  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# CPU algorithm (Default)
#install(
#  DIRECTORY cpu/include/algorithms
#  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# CUDA algorithm
#if(TRACCC_BUILD_CUDA)
#  add_library(CudaAlgorithms SHARED ${CMAKE_CURRENT_SOURCE_DIR}/cuda/src/algorithms/clusterization/component_connection_kernels.cu)
  
#  target_link_libraries(CudaAlgorithms PUBLIC vecmem::core vecmem::cuda traccc::core)
#  target_include_directories(
#    CudaAlgorithms
#    PUBLIC
#    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
#    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
#    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
#    )  
#endif()


