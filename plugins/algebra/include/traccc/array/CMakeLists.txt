message(STATUS "Building 'traccc::array' plugin")

add_library(traccc_array INTERFACE)

target_include_directories(traccc_array
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

install(
  DIRECTORY include/plugins
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(ALGEBRA_CUSTOM_SCALARTYPE)
  target_compile_definitions(
    traccc_array
    INTERFACE -DALGEBRA_CUSTOM_SCALARTYPE=${ALGEBRA_CUSTOM_SCALARTYPE})
endif()

target_link_libraries(traccc_array INTERFACE algebra::array_cmath vecmem::core)

add_library(traccc::array ALIAS traccc_array)
