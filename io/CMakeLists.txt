add_library(traccc_io INTERFACE)

target_include_directories(traccc_io
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

install(TARGETS traccc_io
        PUBLIC_HEADER
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/io
)

target_link_libraries(traccc_io INTERFACE traccc::core traccc::cpu dfelibs)

install(
  DIRECTORY include/csv
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_library(traccc::io ALIAS traccc_io)
