add_library(traccc_core INTERFACE)

target_include_directories(traccc_core
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(traccc_core
  INTERFACE vecmem::core)

install(TARGETS traccc_core
        PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/core)

install(
  DIRECTORY include/definitions
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  DIRECTORY include/edm
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  DIRECTORY include/geometry
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  DIRECTORY include/algorithms
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if (TRACCC_BUILD_cuda)
  find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  target_include_directories(traccc_core
    PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

add_library(traccc::core ALIAS traccc_core)
