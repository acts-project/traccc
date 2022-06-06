# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Enable CUDA as a language.
enable_language( CUDA )

# Project include(s).
include( traccc-compiler-options-cpp )
include( traccc-compiler-options-cuda )

# Set up the build of the traccc::cuda library.
traccc_add_library( traccc_cuda cuda TYPE SHARED
  # Utility definitions.
  "include/traccc/cuda/utils/definitions.hpp"
  "include/traccc/cuda/utils/cuda_helper.cuh"
  # Seed finding code.
  "include/traccc/cuda/seeding/detail/triplet_counter.hpp"
  "include/traccc/cuda/seeding/weight_updating.hpp"
  "include/traccc/cuda/seeding/triplet_counting.hpp"
  "include/traccc/cuda/seeding/seed_selecting.hpp"
  "include/traccc/cuda/seeding/track_params_estimation.hpp"
  "include/traccc/cuda/seeding/triplet_finding.hpp"
  "include/traccc/cuda/seeding/seed_finding.hpp"
  "include/traccc/cuda/seeding/seeding_algorithm.hpp"
  "include/traccc/cuda/seeding/spacepoint_binning.hpp"
  # CCL code.
  "include/traccc/cuda/cca/component_connection.hpp"
  "src/seeding/triplet_counting.cu"
  "src/seeding/weight_updating.cu"
  "src/seeding/track_params_estimation.cu"
  "src/seeding/seed_finding.cu"
  "src/seeding/seed_selecting.cu"
  "src/seeding/spacepoint_binning.cu"
  "src/seeding/triplet_finding.cu"
  "src/seeding/seeding_algorithm.cpp"
  "src/cca/component_connection.cu"
  # sparseCCL clusterization headers
  "include/traccc/cuda/clusterization/clusterization_algorithm.hpp"
  "src/clusterization/cluster_counting.hpp"
  "src/clusterization/clusters_sum.hpp"
  "src/clusterization/component_connection.hpp"
  "src/clusterization/measurements_creation.hpp"
  "src/clusterization/spacepoint_formation.hpp"
  # sparseCCL clusterization code
  "src/clusterization/clusterization_algorithm.cpp"
  "src/clusterization/cluster_counting.cu"
  "src/clusterization/clusters_sum.cu"
  "src/clusterization/component_connection.cu"
  "src/clusterization/measurements_creation.cu"
  "src/clusterization/spacepoint_formation.cu"
)
target_compile_options( traccc_cuda
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr
  --generate-line-info> )
target_link_libraries( traccc_cuda
  PUBLIC traccc::core vecmem::core
  PRIVATE traccc::device_common vecmem::cuda )
