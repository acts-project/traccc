# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake 3.12+ is needed for OBJECT libraries.
cmake_minimum_required( VERSION 3.12 )

# Project include(s).
include( traccc-compiler-options-sycl )

# We need the SYCL language in this directory.
enable_language( SYCL )

# Example executable(s).
traccc_add_executable( traccc_sycl_language_example
   "sycl_language_example.sycl" )

# SYCL seeding executable(s).
traccc_add_executable( seeding_example_sycl "seeding_example_sycl.sycl"
   LINK_LIBRARIES traccc::options vecmem::core vecmem::sycl traccc::io
                  traccc::core traccc::device_common traccc::sycl
                  traccc::performance )

traccc_add_executable( seq_example_sycl "seq_example_sycl.sycl"
   LINK_LIBRARIES traccc::options vecmem::core vecmem::sycl traccc::io
                  traccc::core traccc::device_common traccc::sycl
                  traccc::performance )

#
# Set up the "throughput applications".
#
add_library( traccc_examples_sycl OBJECT
   "full_chain_algorithm.hpp"
   "full_chain_algorithm.sycl" )
target_link_libraries( traccc_examples_sycl
   PUBLIC vecmem::core vecmem::sycl traccc::core traccc::device_common traccc::sycl )

traccc_add_executable( throughput_st_sycl "throughput_st.cpp"
   LINK_LIBRARIES vecmem::core vecmem::sycl traccc::io traccc::performance
                  traccc::core traccc::device_common traccc::sycl
                  traccc::options traccc_examples_sycl )

traccc_add_executable( throughput_mt_sycl "throughput_mt.cpp"
   LINK_LIBRARIES TBB::tbb vecmem::core vecmem::sycl traccc::io traccc::performance
                  traccc::core traccc::device_common traccc::sycl
                  traccc::options traccc_examples_sycl )
