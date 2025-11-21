# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2025 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake 3.12+ is needed for OBJECT libraries.
cmake_minimum_required( VERSION 3.12 )

# Project include(s).
include( traccc-compiler-options-sycl )

# We need the SYCL language in this directory.
enable_language( SYCL )

#
# Set up the library used by the applications.
#
add_library( traccc_examples_sycl OBJECT
   "device_backend.hpp"
   "device_backend.cpp"
   "full_chain_algorithm.hpp"
   "full_chain_algorithm.sycl" )
target_link_libraries( traccc_examples_sycl
   PUBLIC vecmem::core vecmem::sycl detray::core detray::detectors
          traccc::core traccc::device_common traccc::sycl
          traccc_examples_common )

#
# Set up the applications.
#
traccc_add_executable( throughput_st_sycl "throughput_st.cpp"
   LINK_LIBRARIES indicators::indicators traccc_examples_common
                  traccc_examples_sycl )

traccc_add_executable( throughput_mt_sycl "throughput_mt.cpp"
   LINK_LIBRARIES TBB::tbb indicators::indicators traccc_examples_common
                  traccc_examples_sycl )

traccc_add_executable( track_finding_validation_sycl
   "track_finding_validation.cpp"
   LINK_LIBRARIES traccc_examples_common traccc_examples_sycl )

traccc_add_executable( reconstruction_validation_sycl
   "reconstruction_validation.cpp"
   LINK_LIBRARIES traccc_examples_common traccc_examples_sycl )
