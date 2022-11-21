# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
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

find_package(OpenMP COMPONENTS CXX)
if (OpenMP_CXX_FOUND)
    add_subdirectory(openmp)
    message(STATUS "OpenMP found! Include /examples/openmp subdirectory")
endif()
