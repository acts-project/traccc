add_library(traccc_cuda_utils INTERFACE)

target_include_directories(traccc_cuda_utils
  INTERFACE
  $<BUILD_INTERFACE:${TRACCC_CUDA_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

target_compile_options(traccc_cuda_utils
  INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:  
  --fmad=false>)

add_library(traccc::cuda::utils ALIAS traccc_cuda_utils)
