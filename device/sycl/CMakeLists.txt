# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-compiler-options-cpp )
include( traccc-compiler-options-sycl )

# We need the SYCL language in this directory.
enable_language( SYCL )

# Set up the build of the traccc::sycl library.
traccc_add_library( traccc_sycl sycl TYPE SHARED
  # header files
  "include/traccc/sycl/seeding/seed_finding.hpp"
  "include/traccc/sycl/seeding/seeding_algorithm.hpp"
  "include/traccc/sycl/seeding/spacepoint_binning.hpp"
  "include/traccc/sycl/seeding/track_params_estimation.hpp"
  "include/traccc/sycl/utils/queue_wrapper.hpp"
  # implementation files
  "src/seeding/counting_grid_capacities.hpp"
  "src/seeding/counting_grid_capacities.sycl"
  "src/seeding/doublet_counter.hpp"
  "src/seeding/doublet_counting.hpp"
  "src/seeding/doublet_counting.sycl"
  "src/seeding/doublet_finding.hpp"
  "src/seeding/doublet_finding.sycl"
  "src/seeding/populating_grid.hpp"
  "src/seeding/populating_grid.sycl"
  "src/seeding/seed_finding.cpp"
  "src/seeding/seed_selecting.hpp"
  "src/seeding/seed_selecting.sycl"
  "src/seeding/spacepoint_binning.cpp"
  "src/seeding/sycl_helper.hpp"
  "src/seeding/track_params_estimation.sycl"
  "src/seeding/triplet_counter.hpp"
  "src/seeding/triplet_counting.hpp"
  "src/seeding/triplet_counting.sycl"
  "src/seeding/triplet_finding.hpp"
  "src/seeding/triplet_finding.sycl"
  "src/seeding/weight_updating.hpp"
  "src/seeding/weight_updating.sycl"
  "src/utils/get_queue.hpp"
  "src/utils/get_queue.sycl"
  "src/utils/queue_wrapper.cpp" )
target_link_libraries( traccc_sycl
  PUBLIC vecmem::core traccc::core
  PRIVATE vecmem::sycl )

# Prevent Eigen from getting confused when building code for a HIP backend.
target_compile_definitions( traccc_sycl PRIVATE EIGEN_NO_CUDA )
