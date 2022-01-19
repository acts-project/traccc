# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# We need the SYCL language in this directory.
enable_language( SYCL )

# Example executable(s).
add_executable( traccc_sycl_language_example "sycl_language_example.sycl" )

# BUG, 2022-01-19, commit 4bdd0db7f4068d978878e6fb92a6d96b0968bf87
# Does not compile when linking any internal library to a .sycl
target_link_libraries (traccc_sycl_language_example LINK_PUBLIC
  traccc::algorithms)