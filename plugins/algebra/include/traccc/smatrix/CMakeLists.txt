message(STATUS "Building 'traccc::smatrix' plugin")

add_library(traccc_smatrix INTERFACE)

target_include_directories(traccc_smatrix
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

install(
  DIRECTORY include/plugins
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(ALGEBRA_CUSTOM_SCALARTYPE)
  target_compile_definitions(
    traccc_smatrix
    INTERFACE -DALGEBRA_CUSTOM_SCALARTYPE=${ALGEBRA_CUSTOM_SCALARTYPE})
endif()

target_link_libraries(traccc_smatrix INTERFACE algebra::smatrix ROOT::MathCore)

add_library(traccc::smatrix ALIAS traccc_smatrix)
