add_library(traccc_cuda_seeding SHARED
  ${TRACCC_CUDA_SEEDING_SOURCE_DIR}/doublet_counting.cu
  ${TRACCC_CUDA_SEEDING_SOURCE_DIR}/doublet_finding.cu
  ${TRACCC_CUDA_SEEDING_SOURCE_DIR}/triplet_counting.cu
  ${TRACCC_CUDA_SEEDING_SOURCE_DIR}/triplet_finding.cu
  ${TRACCC_CUDA_SEEDING_SOURCE_DIR}/weight_updating.cu
  ${TRACCC_CUDA_SEEDING_SOURCE_DIR}/seed_selecting.cu)

target_include_directories(traccc_cuda_seeding
  PUBLIC
  $<BUILD_INTERFACE:${TRACCC_CUDA_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

target_compile_options(traccc_cuda_seeding
  PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
  #-arch=sm_52 
  #-gencode=arch=compute_75,code=sm_75 
  --expt-relaxed-constexpr
  #--use_fast_math>)
  -fmad=false>)

target_link_libraries(traccc_cuda_seeding PUBLIC traccc::core vecmem::cuda)

add_library(traccc::cuda::seeding ALIAS traccc_cuda_seeding)
