# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

message(STATUS "Building 'traccc::smatrix' plugin")

add_library(traccc_smatrix INTERFACE)

target_include_directories(traccc_smatrix
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

install(
  DIRECTORY include/plugins
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

target_compile_definitions(
  traccc_smatrix
  INTERFACE TRACCC_CUSTOM_SCALARTYPE=${TRACCC_CUSTOM_SCALARTYPE})

target_link_libraries(traccc_smatrix INTERFACE algebra::smatrix_smatrix vecmem::core)

add_library(traccc::smatrix ALIAS traccc_smatrix)
