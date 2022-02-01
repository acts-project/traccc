# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the "build" of the traccc::algorithms_cuda library.
traccc_add_library( traccc_algorithms_cuda algorithms_cuda TYPE INTERFACE
  "include/traccc/cuda/track_finding/seeding_algorithm.hpp" )
target_link_libraries( traccc_algorithms_cuda
  INTERFACE vecmem::core traccc::core traccc::cuda )
