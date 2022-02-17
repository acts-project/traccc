# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-compiler-options-sycl )

# We need the SYCL language in this directory.
enable_language( SYCL )

# Example executable(s).
traccc_add_executable( traccc_sycl_language_example
   "sycl_language_example.sycl" )

# SYCL seeding executable(s).
traccc_add_executable( seeding_example_sycl "seeding_example_sycl.sycl"
   LINK_LIBRARIES vecmem::core vecmem::sycl traccc::io
                  traccc::algorithms traccc::algorithms_sycl Boost::program_options)
traccc_add_executable( seq_example_sycl "seq_example_sycl.sycl"
   LINK_LIBRARIES vecmem::core vecmem::sycl traccc::io
                  traccc::algorithms traccc::algorithms_sycl Boost::program_options)
