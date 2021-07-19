add_library(traccc_core INTERFACE)

target_include_directories(traccc_core
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(traccc_core
  INTERFACE vecmem::core ActsCore)

install(TARGETS traccc_core
        PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/core)

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_library(traccc::core ALIAS traccc_core)
