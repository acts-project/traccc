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
