message(STATUS "Building 'traccc::vecmem' plugin")

add_library(traccc_vecmem INTERFACE)

target_include_directories(traccc_vecmem
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

install(
  DIRECTORY include/plugins
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(ALGEBRA_CUSTOM_SCALARTYPE)
  target_compile_definitions(
    traccc_vecmem
    INTERFACE -DALGEBRA_CUSTOM_SCALARTYPE=${ALGEBRA_CUSTOM_SCALARTYPE})
endif()

target_link_libraries(traccc_vecmem INTERFACE algebra::vecmem_cmath vecmem::core)

add_library(traccc::vecmem ALIAS traccc_vecmem)
