add_library(traccc_cpu INTERFACE)

target_include_directories(traccc_cpu
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

install(
  DIRECTORY include
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

target_link_libraries(traccc_cpu INTERFACE traccc_core)
