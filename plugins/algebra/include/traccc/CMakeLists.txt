add_library(traccc_algebra_plugins INTERFACE)

# array
if (${TRACCC_ALGEBRA_PLUGIN} STREQUAL array)
  set(ALGEBRA_PLUGIN_INCLUDE_ARRAY ON CACHE INTERNAL "")
  target_link_libraries(traccc_algebra_plugins INTERFACE traccc::array)  
  add_subdirectory(array)
# eigen  
elseif (${TRACCC_ALGEBRA_PLUGIN} STREQUAL eigen)
  set(ALGEBRA_PLUGIN_INCLUDE_EIGEN ON CACHE INTERNAL "")
  find_package(Eigen3 REQUIRED)
  include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})
  target_link_libraries(traccc_algebra_plugins INTERFACE traccc::eigen)  
  add_subdirectory(eigen)
# smatrix
elseif (${TRACCC_ALGEBRA_PLUGIN} STREQUAL smatrix)
  set(ALGEBRA_PLUGIN_INCLUDE_SMATRIX ON CACHE INTERNAL "")
  find_package(ROOT REQUIRED COMPONENTS MathCore)
  include_directories(SYSTEM ${ROOT_INCLUDE_DIRS})
  target_link_libraries(traccc_algebra_plugins INTERFACE traccc::smatrix)  
  add_subdirectory(smatrix)
# vc
elseif (${TRACCC_ALGEBRA_PLUGIN} STREQUAL vc)
  set(ALGEBRA_PLUGIN_INCLUDE_VC ON CACHE INTERNAL "")
  target_link_libraries(traccc_algebra_plugins INTERFACE traccc::vc_array)  
  add_subdirectory(vc)
endif()

add_library(traccc::algebra_plugins ALIAS traccc_algebra_plugins)
