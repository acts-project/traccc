add_library(traccc_algebra_plugins INTERFACE)

# array
target_link_libraries(traccc_algebra_plugins INTERFACE traccc::array)
add_subdirectory(array)

# eigen
if (ALGEBRA_PLUGINS_INCLUDE_EIGEN)
  target_link_libraries(traccc_algebra_plugins INTERFACE traccc::eigen)
  add_subdirectory(eigen)
endif()

# smatrix
if (ALGEBRA_PLUGINS_INCLUDE_SMATRIX)
  target_link_libraries(traccc_algebra_plugins INTERFACE traccc::smatrix)
  add_subdirectory(smatrix)
endif()

# vc
if (ALGEBRA_PLUGINS_INCLUDE_VC)
  target_link_libraries(traccc_algebra_plugins INTERFACE traccc::vc_array)
  add_subdirectory(vc)
endif()

# vecmem
if (ALGEBRA_PLUGINS_INCLUDE_VECMEM)
  target_link_libraries(traccc_algebra_plugins INTERFACE traccc::vecmem)
  add_subdirectory(vecmem)
endif()


add_library(traccc::algebra_plugins ALIAS traccc_algebra_plugins)
