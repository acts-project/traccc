# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2023-2025 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-alpaka-functions )
include( traccc-compiler-options-cpp )

set(PUBLIC_LIBRARIES traccc::core traccc::device_common vecmem::core covfie::core)
set(PRIVATE_LIBRARIES alpaka::alpaka)

traccc_enable_language_alpaka()

if(alpaka_ACC_GPU_CUDA_ENABLE)
  list(APPEND PRIVATE_LIBRARIES CUDA::cudart vecmem::cuda)
elseif(alpaka_ACC_GPU_HIP_ENABLE)
  find_package( HIPToolkit REQUIRED )
  list(APPEND PRIVATE_LIBRARIES HIP::hiprt vecmem::hip)
elseif(alpaka_ACC_SYCL_ENABLE)
  include( traccc-compiler-options-sycl )
  list(APPEND PRIVATE_LIBRARIES vecmem::sycl)
endif()

traccc_add_alpaka_library( traccc_alpaka alpaka TYPE SHARED
  # Utility definitions.
  "include/traccc/alpaka/utils/make_prefix_sum_buff.hpp"
  "src/utils/make_prefix_sum_buff.cpp"
  "src/utils/get_device_info.cpp"
  # Clusterization
  "include/traccc/alpaka/clusterization/clusterization_algorithm.hpp"
  "src/clusterization/clusterization_algorithm.cpp"
  "include/traccc/alpaka/clusterization/measurement_sorting_algorithm.hpp"
  "src/clusterization/measurement_sorting_algorithm.cpp"
  # Seeding code
  "include/traccc/alpaka/seeding/spacepoint_formation_algorithm.hpp"
  "src/seeding/spacepoint_formation_algorithm.cpp"
  "include/traccc/alpaka/seeding/details/spacepoint_binning.hpp"
  "src/seeding/spacepoint_binning.cpp"
  "include/traccc/alpaka/seeding/details/seed_finding.hpp"
  "src/seeding/seed_finding.cpp"
  "include/traccc/alpaka/seeding/seeding_algorithm.hpp"
  "src/seeding/seeding_algorithm.cpp"
  "include/traccc/alpaka/seeding/track_params_estimation.hpp"
  "src/seeding/track_params_estimation.cpp"
)
# Set up Thrust specifically for the traccc::alpaka library.
if(alpaka_ACC_GPU_CUDA_ENABLE)
thrust_create_target( traccc::cuda_thrust
  HOST CPP
  DEVICE CUDA )
  target_link_libraries(traccc::cuda_thrust INTERFACE CUDA::cudart)
  list(APPEND PRIVATE_LIBRARIES traccc::cuda_thrust)
else()
thrust_create_target (traccc::alpaka_thrust
  HOST CPP
  DEVICE CPP )
  list(APPEND PRIVATE_LIBRARIES traccc::alpaka_thrust)
endif()

target_link_libraries(traccc_alpaka PUBLIC ${PUBLIC_LIBRARIES} PRIVATE ${PRIVATE_LIBRARIES})
