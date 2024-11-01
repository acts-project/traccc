# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-compiler-options-cpp )
include( traccc-compiler-options-sycl )

# We need the SYCL language in this directory.
enable_language( SYCL )

# Set up the build of the traccc::sycl library.
traccc_add_library( traccc_sycl sycl TYPE SHARED
  # Spacepoint formation algorithm.
  "include/traccc/sycl/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp"
  "src/seeding/silicon_pixel_spacepoint_formation_algorithm.cpp"
  "src/seeding/silicon_pixel_spacepoint_formation_algorithm_default_detector.sycl"
  "src/seeding/silicon_pixel_spacepoint_formation_algorithm_telescope_detector.sycl"
  "src/seeding/silicon_pixel_spacepoint_formation.hpp"
  # Track fitting algorithm.
  "include/traccc/sycl/fitting/kalman_fitting_algorithm.hpp"
  "src/fitting/kalman_fitting_algorithm.cpp"
  "src/fitting/kalman_fitting_algorithm_constant_field_default_detector.sycl"
  "src/fitting/kalman_fitting_algorithm_constant_field_telescope_detector.sycl"
  "src/fitting/fit_tracks.hpp"
  # header files
  "include/traccc/sycl/seeding/seeding_algorithm.hpp"
  "include/traccc/sycl/seeding/seed_finding.hpp"
  "include/traccc/sycl/seeding/spacepoint_binning.hpp"
  "include/traccc/sycl/seeding/track_params_estimation.hpp"
  "include/traccc/sycl/utils/queue_wrapper.hpp"
  "include/traccc/sycl/utils/calculate1DimNdRange.hpp"
  "include/traccc/sycl/utils/make_prefix_sum_buff.hpp"
  # implementation files
  "src/clusterization/clusterization_algorithm.sycl"
  "src/seeding/seed_finding.sycl"
  "src/seeding/seeding_algorithm.cpp"
  "src/seeding/spacepoint_binning.sycl"
  "src/seeding/track_params_estimation.sycl"
  "src/utils/barrier.hpp"
  "src/utils/get_queue.hpp"
  "src/utils/get_queue.sycl"
  "src/utils/queue_wrapper.cpp"
  "src/utils/calculate1DimNdRange.sycl"
  "src/utils/make_prefix_sum_buff.sycl"
  "src/sanity/contiguous_on.hpp"
  "src/sanity/ordered_on.hpp"
)
target_link_libraries( traccc_sycl
  PUBLIC traccc::core detray::core vecmem::core covfie::core
  PRIVATE traccc::device_common vecmem::sycl oneDPL )
