# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# We need the SYCL language in this directory.
enable_language( SYCL )

# Set up the build of the traccc::sycl library.
traccc_add_library( traccc_sycl sycl TYPE SHARED
  # header files
  "include/traccc/sycl/seeding/detail/doublet_counter.hpp"
  "include/traccc/sycl/seeding/detail/triplet_counter.hpp"
  "include/traccc/sycl/seeding/detail/sycl_helper.hpp"
  "include/traccc/sycl/seeding/counting_grid_capacities.hpp"
  "include/traccc/sycl/seeding/doublet_counting.hpp"
  "include/traccc/sycl/seeding/doublet_finding.hpp"
  "include/traccc/sycl/seeding/populating_grid.hpp"
  "include/traccc/sycl/seeding/seed_finding.hpp"
  "include/traccc/sycl/seeding/seed_selecting.hpp"
  "include/traccc/sycl/seeding/spacepoint_binning.hpp"
  "include/traccc/sycl/seeding/track_params_estimation.hpp"
  "include/traccc/sycl/seeding/triplet_counting.hpp"
  "include/traccc/sycl/seeding/triplet_finding.hpp"
  "include/traccc/sycl/seeding/weight_updating.hpp"
  # implementation files
  "src/seeding/counting_grid_capacities.sycl"
  "src/seeding/doublet_counting.sycl"
  "src/seeding/doublet_finding.sycl"
  "src/seeding/populating_grid.sycl"
  "src/seeding/seed_selecting.sycl"
  "src/seeding/track_params_estimation.sycl"
  "src/seeding/triplet_counting.sycl"
  "src/seeding/triplet_finding.sycl"
  "src/seeding/weight_updating.sycl" )
target_link_libraries( traccc_sycl
  PUBLIC traccc::core vecmem::sycl )
