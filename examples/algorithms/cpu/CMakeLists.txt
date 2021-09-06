add_library(traccc_algorithms INTERFACE)

target_include_directories(traccc_algorithms
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(traccc_algorithms
  INTERFACE traccc::core)

add_library(traccc::algorithms ALIAS traccc_algorithms)
