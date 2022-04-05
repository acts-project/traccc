# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-compiler-options-cpp )
include( traccc-compiler-options-cuda )

# Enable CUDA as a language.
enable_language( CUDA )

# Set up the build of the traccc::cuda library.
traccc_add_library( traccc_cuda cuda TYPE SHARED
  # Utility definitions.
  "include/traccc/cuda/utils/definitions.hpp"
  "include/traccc/cuda/utils/cuda_helper.cuh"
  # Seed finding code.
  "include/traccc/cuda/seeding/detail/triplet_counter.hpp"
  "include/traccc/cuda/seeding/detail/doublet_counter.hpp"
  "include/traccc/cuda/seeding/weight_updating.hpp"
  "include/traccc/cuda/seeding/doublet_counting.hpp"
  "include/traccc/cuda/seeding/counting_grid_capacities.hpp"
  "include/traccc/cuda/seeding/triplet_counting.hpp"
  "include/traccc/cuda/seeding/populating_grid.hpp"
  "include/traccc/cuda/seeding/seed_selecting.hpp"
  "include/traccc/cuda/seeding/track_params_estimation.hpp"
  "include/traccc/cuda/seeding/doublet_finding.hpp"
  "include/traccc/cuda/seeding/triplet_finding.hpp"
  "include/traccc/cuda/seeding/seed_finding.hpp"
  "include/traccc/cuda/seeding/seeding_algorithm.hpp"
  "include/traccc/cuda/seeding/spacepoint_binning.hpp"
  # CCL code.
  "include/traccc/cuda/cca/component_connection.hpp"
  "src/seeding/doublet_counting.cu"
  "src/seeding/triplet_counting.cu"
  "src/seeding/weight_updating.cu"
  "src/seeding/populating_grid.cu"
  "src/seeding/track_params_estimation.cu"
  "src/seeding/counting_grid_capacities.cu"
  "src/seeding/seed_selecting.cu"
  "src/seeding/triplet_finding.cu"
  "src/seeding/doublet_finding.cu"
  "src/cca/component_connection.cu"
)
target_compile_options( traccc_cuda
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> )
target_link_libraries( traccc_cuda
  PUBLIC traccc::core vecmem::cuda )
