# cpu
add_traccc_test(component_connection component_connection_tests.cpp "")

# cuda
if(TRACCC_BUILD_CUDA)
add_traccc_test(component_connection_cuda component_connection_tests_cuda.cpp "")
endif()
