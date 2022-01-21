# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

message(STATUS "Building 'traccc::vc_array' plugin")

add_library(traccc_vc_array INTERFACE)

target_include_directories(traccc_vc_array
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

install(
  DIRECTORY include/plugins
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

target_compile_definitions(
  traccc_vc_array
  INTERFACE TRACCC_CUSTOM_SCALARTYPE=${TRACCC_CUSTOM_SCALARTYPE})

target_link_libraries(traccc_vc_array INTERFACE algebra::vc_vc vecmem::core)

add_library(traccc::vc_array ALIAS traccc_vc_array)
