message(STATUS "Building 'traccc::eigen' plugin")

add_library(traccc_eigen INTERFACE)

target_include_directories(traccc_eigen
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

install(
  DIRECTORY include/plugins
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(ALGEBRA_CUSTOM_SCALARTYPE)
  target_compile_definitions(
    traccc_eigen
    INTERFACE -DALGEBRA_CUSTOM_SCALARTYPE=${ALGEBRA_CUSTOM_SCALARTYPE})
endif()

target_link_libraries(traccc_eigen INTERFACE algebra::eigen)

add_library(traccc::eigen ALIAS traccc_eigen)
