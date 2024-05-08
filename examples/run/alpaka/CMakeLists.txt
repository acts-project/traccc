# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

set(LIBRARIES vecmem::core traccc::io traccc::performance
    traccc::core traccc::device_common traccc::alpaka alpaka::alpaka
    traccc::options)

if(alpaka_ACC_GPU_CUDA_ENABLE)
  enable_language(CUDA)
  include( traccc-compiler-options-cuda )

  list (APPEND LIBRARIES vecmem::cuda traccc::cuda)
elseif(alpaka_ACC_GPU_HIP_ENABLE)
  enable_language(HIP)
  include( traccc-compiler-options-hip )
  find_package( HIPToolkit REQUIRED )
  list(APPEND LIBRARIES HIP::hiprt vecmem::hip)
 endif()

traccc_add_executable( seeding_example_alpaka "seeding_example_alpaka.cpp"
    LINK_LIBRARIES ${LIBRARIES} )

