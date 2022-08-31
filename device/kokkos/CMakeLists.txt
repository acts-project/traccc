# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Enable CUDA as a language.
enable_language( CUDA )

# Project include(s).
include( traccc-compiler-options-cpp )
include( traccc-compiler-options-kokkos )
include( traccc-compiler-options-kokkos )

add_subdirectory( device/kokkos ) 

# Set up the build of the traccc::kokkos library.
traccc_add_library( traccc_kokkos kokkos TYPE SHARED
  # Utility definitions.
  "include/traccc/kokkos/utils/definitions.hpp"
  #"include/traccc/kokkos/utils/cuda_helper.cuh"
  "include/traccc/kokkos/utils/kokkos_helper.cpp"
  # Seed finding code.
  "include/traccc/kokkos/seeding/detail/triplet_counter.hpp"
  "include/traccc/kokkos/seeding/weight_updating.hpp"
  "include/traccc/kokkos/seeding/triplet_counting.hpp"
  "include/traccc/kokkos/seeding/seed_selecting.hpp"
  "include/traccc/kokkos/seeding/track_params_estimation.hpp"
  "include/traccc/kokkos/seeding/triplet_finding.hpp"
  "include/traccc/kokkos/seeding/seed_finding.hpp"
  "include/traccc/kokkos/seeding/seeding_algorithm.hpp"
  "include/traccc/kokkos/seeding/spacepoint_binning.hpp"
  # CCL code.
  "include/traccc/kokkos/cca/component_connection.hpp"
  "src/seeding/triplet_counting.cpp"
  "src/seeding/weight_updating.cpp"
  "src/seeding/track_params_estimation.cu"
  "src/seeding/seed_finding.cpp"
  "src/seeding/seed_selecting.cpp"
  "src/seeding/spacepoint_binning.cu"
  "src/seeding/triplet_finding.cpp"
  "src/seeding/seeding_algorithm.cpp"
  "src/cca/component_connection.cu"
)
target_compile_options( traccc_kokkos
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> )
target_link_libraries( traccc_kokkos
  PUBLIC traccc::core vecmem::core
  PRIVATE traccc::device_common vecmem::kokkos )
