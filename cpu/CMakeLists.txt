add_library(traccc_cpu INTERFACE)

target_include_directories(traccc_cpu
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
  $<INSTALL_INTERFACE:include>
)

install(
  DIRECTORY include/edm
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cpu)

install(
  DIRECTORY include/algorithms
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cpu)

target_link_libraries(traccc_cpu INTERFACE traccc::core)

add_library(traccc::cpu ALIAS traccc_cpu)
