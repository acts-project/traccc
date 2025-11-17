# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2025 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-compiler-options-cpp )

# Create the common library.
add_library(traccc_examples_common STATIC
   "common/make_magnetic_field.hpp"
   "common/make_magnetic_field.cpp"
   "common/print_fitted_tracks_statistics.hpp"
   "common/print_fitted_tracks_statistics.cpp"
   "common/throughput_mt.hpp"
   "common/throughput_mt.ipp"
   "common/throughput_st.hpp"
   "common/throughput_st.ipp")
target_link_libraries(traccc_examples_common
   PUBLIC traccc::core traccc::options
   PRIVATE traccc::io)

# Add all the subdirectories that can be built.
add_subdirectory(cpu)

if (TRACCC_BUILD_CUDA)
  add_subdirectory(cuda)
endif()

if (TRACCC_BUILD_SYCL)
  add_subdirectory(sycl)
endif()

if (TRACCC_BUILD_ALPAKA)
  add_subdirectory(alpaka)
endif()
