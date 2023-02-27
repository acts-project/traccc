# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
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
  "include/traccc/sycl/clusterization/clusterization_algorithm.hpp"
  "include/traccc/sycl/fitting/fitting_algorithm.hpp"
  "include/traccc/sycl/seeding/seeding_algorithm.hpp"
  "include/traccc/sycl/seeding/seed_finding.hpp"
  "include/traccc/sycl/seeding/spacepoint_binning.hpp"
  "include/traccc/sycl/seeding/track_params_estimation.hpp"
  "include/traccc/sycl/utils/queue_wrapper.hpp"
  "include/traccc/sycl/utils/calculate1DimNdRange.hpp"
  "include/traccc/sycl/utils/make_prefix_sum_buff.hpp"
  # implementation files
  "src/clusterization/clusterization_algorithm.sycl"
  "src/fitting/fitting_algorithm.sycl"
  "src/seeding/seed_finding.sycl"
  "src/seeding/seeding_algorithm.cpp"
  "src/seeding/spacepoint_binning.sycl"
  "src/seeding/track_params_estimation.sycl"
  "src/utils/get_queue.hpp"
  "src/utils/get_queue.sycl"
  "src/utils/queue_wrapper.cpp" 
  "src/utils/calculate1DimNdRange.sycl"
  "src/utils/make_prefix_sum_buff.sycl" )
target_link_libraries( traccc_sycl
  PUBLIC traccc::core detray::core detray::utils vecmem::core covfie::core
  PRIVATE traccc::device_common vecmem::sycl )
