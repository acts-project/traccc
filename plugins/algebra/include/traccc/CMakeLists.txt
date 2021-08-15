add_library(traccc_algebra_plugins INTERFACE)

# array
if (${ALGEBRA_PLUGIN_INCLUDE_ARRAY})  
  target_link_libraries(traccc_algebra_plugins INTERFACE traccc::array)  
  add_subdirectory(array)
endif()
  
# eigen  
if (${ALGEBRA_PLUGIN_INCLUDE_EIGEN})
  target_link_libraries(traccc_algebra_plugins INTERFACE traccc::eigen)  
  add_subdirectory(eigen)
endif()

# smatrix
if (${ALGEBRA_PLUGIN_INCLUDE_SMATRIX})
  target_link_libraries(traccc_algebra_plugins INTERFACE traccc::smatrix)    
  add_subdirectory(smatrix)
endif()
  
# vc
if (${ALGEBRA_PLUGIN_INCLUDE_VC})
  target_link_libraries(traccc_algebra_plugins INTERFACE traccc::vc_array)  
  add_subdirectory(vc)
endif()

add_library(traccc::algebra_plugins ALIAS traccc_algebra_plugins)
