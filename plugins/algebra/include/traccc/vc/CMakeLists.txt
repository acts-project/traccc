message(STATUS "Building 'traccc::vc_array' plugin")

add_library(traccc_vc_array INTERFACE)

target_include_directories(traccc_vc_array
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

install(
  DIRECTORY include/plugins
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(ALGEBRA_CUSTOM_SCALARTYPE)
  target_compile_definitions(
    traccc_vc_array
    INTERFACE -DALGEBRA_CUSTOM_SCALARTYPE=${ALGEBRA_CUSTOM_SCALARTYPE})
endif()

target_link_libraries(traccc_vc_array INTERFACE algebra::vc_array Vc)

add_library(traccc::vc_array ALIAS traccc_vc_array)
