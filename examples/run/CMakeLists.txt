# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-compiler-options-cpp )

# Add all the subdirectories that can be built.
add_subdirectory(cpu)

if (TRACCC_BUILD_CUDA)
  add_subdirectory(cuda)
endif()

if (TRACCC_BUILD_SYCL)
  add_subdirectory(sycl)
endif()

if (TRACCC_BUILD_KOKKOS)
  add_subdirectory(kokkos)
endif()

if (TRACCC_BUILD_ALPAKA)
  add_subdirectory(alpaka)
endif()
