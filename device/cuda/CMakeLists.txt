# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Enable CUDA as a language.
enable_language( CUDA )

# Find the CUDA toolkit
find_package( CUDAToolkit REQUIRED )

# Project include(s).
include( traccc-compiler-options-cpp )
include( traccc-compiler-options-cuda )

# Set up the build of the traccc::cuda library.
traccc_add_library( traccc_cuda cuda TYPE SHARED
  # Utility definitions.
  "include/traccc/cuda/utils/definitions.hpp"
  "include/traccc/cuda/utils/make_prefix_sum_buff.hpp"
  "src/utils/make_prefix_sum_buff.cu"
  "include/traccc/cuda/utils/stream.hpp"
  "src/utils/stream.cpp"
  "src/utils/opaque_stream.hpp"
  "src/utils/opaque_stream.cpp"
  "src/utils/utils.hpp"
  "src/utils/utils.cpp"
  # Seed finding code.
  "include/traccc/cuda/seeding/track_params_estimation.hpp"
  "include/traccc/cuda/seeding/seed_finding.hpp"
  "include/traccc/cuda/seeding/seeding_algorithm.hpp"
  "include/traccc/cuda/seeding/spacepoint_binning.hpp"
  # CCL code.
  "include/traccc/cuda/cca/component_connection.hpp"
  "src/seeding/track_params_estimation.cu"
  "src/seeding/seed_finding.cu"
  "src/seeding/spacepoint_binning.cu"
  "src/seeding/seeding_algorithm.cpp"
  "src/cca/component_connection.cu"
  # Clusterization
  "include/traccc/cuda/clusterization/clusterization_algorithm.hpp"
  "src/clusterization/clusterization_algorithm.cu"
  # Fitting
  "include/traccc/cuda/fitting/fitting_algorithm.hpp"
  "src/fitting/fitting_algorithm.cu")
if(TRACCC_SETUP_PROFILER)
target_compile_options( traccc_cuda
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xcompiler -finstrument-functions -fPIC > )
target_link_libraries( traccc_cuda
        PUBLIC traccc::core detray::core detray::utils vecmem::core covfie::core
        PRIVATE CUDA::cudart traccc::device_common vecmem::cuda -lnvToolsExt)
else()
    target_compile_options( traccc_cuda
            PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_link_libraries( traccc_cuda
            PUBLIC traccc::core detray::core detray::utils vecmem::core covfie::core
            PRIVATE CUDA::cudart traccc::device_common vecmem::cuda)
endif(TRACCC_SETUP_PROFILER)


