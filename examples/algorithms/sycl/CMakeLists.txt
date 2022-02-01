# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the "build" of the traccc::algorithms_sycl library.
traccc_add_library( traccc_algorithms_sycl algorithms_sycl TYPE INTERFACE
  "include/traccc/sycl/track_finding/seeding_algorithm.hpp" )
target_link_libraries( traccc_algorithms_sycl
  INTERFACE vecmem::core traccc::core traccc::sycl )
