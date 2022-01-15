# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# We need the SYCL language in this directory.
enable_language( SYCL )

# Example executable(s).
add_executable( traccc_sycl_language_example "sycl_language_example.sycl" )

# SYCL seeding executables
add_executable (seeding_example_sycl seeding_example_sycl.sycl)
# add_executable(seq_example_sycl seq_example_sycl.sycl)

target_link_libraries (seeding_example_sycl LINK_PUBLIC
   traccc::io
   traccc::algorithms
   traccc::algorithms_sycl)

# target_link_libraries (seq_example_sycl LINK_PUBLIC
#    traccc::io
#    traccc::algorithms
#    traccc::algorithms_sycl)
