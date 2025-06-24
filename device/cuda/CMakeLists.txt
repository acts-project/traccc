# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2025 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Enable CUDA as a language.
enable_language( CUDA )

# CMake 3.26 is the first to support C++20 in CUDA.
cmake_minimum_required( VERSION 3.26 )

# Find the CUDA toolkit
find_package( CUDAToolkit REQUIRED )

# Project include(s).
include( traccc-compiler-options-cpp )
include( traccc-compiler-options-cuda )

# Set up the build of the traccc::cuda library.
traccc_add_library( traccc_cuda cuda TYPE SHARED
  # Utility definitions.
  "include/traccc/cuda/utils/make_prefix_sum_buff.hpp"
  "src/utils/make_prefix_sum_buff.cu"
  "include/traccc/cuda/utils/stream.hpp"
  "src/utils/barrier.hpp"
  "src/utils/cuda_error_handling.hpp"
  "src/utils/cuda_error_handling.cpp"
  "src/utils/stream.cpp"
  "src/utils/opaque_stream.hpp"
  "src/utils/opaque_stream.cpp"
  "src/utils/utils.hpp"
  "src/utils/utils.cpp"
  "src/sanity/contiguous_on.cuh"
  "src/sanity/ordered_on.cuh"
  # Seed finding code.
  "include/traccc/cuda/seeding/track_params_estimation.hpp"
  "src/seeding/track_params_estimation.cu"
  "include/traccc/cuda/seeding/details/spacepoint_binning.hpp"
  "src/seeding/spacepoint_binning.cu"
  "include/traccc/cuda/seeding/details/seed_finding.hpp"
  "src/seeding/seed_finding.cu"
  "include/traccc/cuda/seeding/seeding_algorithm.hpp"
  "src/seeding/seeding_algorithm.cpp"
  "include/traccc/cuda/seeding/spacepoint_formation_algorithm.hpp"
  "src/seeding/spacepoint_formation_algorithm.cu"
  # Clusterization
  "include/traccc/cuda/clusterization/clusterization_algorithm.hpp"
  "src/clusterization/clusterization_algorithm.cu"
  "include/traccc/cuda/clusterization/measurement_sorting_algorithm.hpp"
  "src/clusterization/measurement_sorting_algorithm.cu"
  "src/clusterization/kernels/ccl_kernel.cu"
  "src/clusterization/kernels/ccl_kernel.cuh"
  "src/clusterization/kernels/reify_cluster_data.cu"
  # Track finding algorithm(s).
  "include/traccc/cuda/finding/combinatorial_kalman_filter_algorithm.hpp"
  "src/finding/combinatorial_kalman_filter_algorithm.cpp"
  "src/finding/combinatorial_kalman_filter_algorithm_constant_field_default_detector.cu"
  "src/finding/combinatorial_kalman_filter_algorithm_constant_field_telescope_detector.cu"
  "src/finding/combinatorial_kalman_filter.cuh"
  "src/finding/kernels/make_barcode_sequence.cu"
  "src/finding/kernels/make_barcode_sequence.cuh"
  "src/finding/kernels/apply_interaction.hpp"
  "src/finding/kernels/fill_sort_keys.cu"
  "src/finding/kernels/fill_sort_keys.cuh"
  "src/finding/kernels/build_tracks.cu"
  "src/finding/kernels/build_tracks.cuh"
  "src/finding/kernels/find_tracks.cuh"
  "src/finding/kernels/propagate_to_next_surface.hpp"
  "src/finding/kernels/specializations/find_tracks_default_detector.cu"
  "src/finding/kernels/specializations/find_tracks_telescope_detector.cu"
  "src/finding/kernels/specializations/apply_interaction_default_detector.cu"
  "src/finding/kernels/specializations/apply_interaction_telescope_detector.cu"
  "src/finding/kernels/specializations/propagate_to_next_surface_default_detector.cu"
  "src/finding/kernels/specializations/propagate_to_next_surface_telescope_detector.cu"
  # Ambiguity resolution
  "include/traccc/cuda/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
  "src/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.cu"
  "src/ambiguity_resolution/kernels/add_block_offset.cu"
  "src/ambiguity_resolution/kernels/add_block_offset.cuh"
  "src/ambiguity_resolution/kernels/block_inclusive_scan.cu"
  "src/ambiguity_resolution/kernels/block_inclusive_scan.cuh"
  "src/ambiguity_resolution/kernels/count_removable_tracks.cu"
  "src/ambiguity_resolution/kernels/count_removable_tracks.cuh"
  "src/ambiguity_resolution/kernels/count_shared_measurements.cu"
  "src/ambiguity_resolution/kernels/count_shared_measurements.cuh"
  "src/ambiguity_resolution/kernels/exclusive_scan.cu"
  "src/ambiguity_resolution/kernels/exclusive_scan.cuh"
  "src/ambiguity_resolution/kernels/find_max_shared.cu"
  "src/ambiguity_resolution/kernels/find_max_shared.cuh"
  "src/ambiguity_resolution/kernels/fill_inverted_ids.cu"
  "src/ambiguity_resolution/kernels/fill_inverted_ids.cuh"
  "src/ambiguity_resolution/kernels/fill_tracks_per_measurement.cu"
  "src/ambiguity_resolution/kernels/fill_tracks_per_measurement.cuh"
  "src/ambiguity_resolution/kernels/fill_track_candidates.cu"
  "src/ambiguity_resolution/kernels/fill_track_candidates.cuh"
  "src/ambiguity_resolution/kernels/fill_vectors.cu"
  "src/ambiguity_resolution/kernels/fill_vectors.cuh"
  "src/ambiguity_resolution/kernels/gather_tracks.cu"
  "src/ambiguity_resolution/kernels/gather_tracks.cuh"
  "src/ambiguity_resolution/kernels/rearrange_tracks.cu"
  "src/ambiguity_resolution/kernels/rearrange_tracks.cuh"
  "src/ambiguity_resolution/kernels/reset_status.cu"
  "src/ambiguity_resolution/kernels/reset_status.cuh"
  "src/ambiguity_resolution/kernels/scan_block_offsets.cu"
  "src/ambiguity_resolution/kernels/scan_block_offsets.cuh"
  "src/ambiguity_resolution/kernels/sort_updated_tracks.cu"
  "src/ambiguity_resolution/kernels/sort_updated_tracks.cuh"
  "src/ambiguity_resolution/kernels/remove_tracks.cu"
  "src/ambiguity_resolution/kernels/remove_tracks.cuh"
  # Track fitting algorithm(s).
  "include/traccc/cuda/fitting/kalman_fitting_algorithm.hpp"
  "src/fitting/kalman_fitting_algorithm.cpp"
  "src/fitting/kalman_fitting_algorithm_constant_field_default_detector.cu"
  "src/fitting/kalman_fitting_algorithm_constant_field_telescope_detector.cu"
  "src/fitting/kalman_fitting.cuh"
  "src/fitting/kernels/fill_sort_keys.cu"
  "src/fitting/kernels/fit_prelude.cu"
  "src/fitting/kernels/specializations/fit_forward_default_detector.cu"
  "src/fitting/kernels/specializations/fit_forward_telescope_detector.cu"
  "src/fitting/kernels/specializations/fit_backward_default_detector.cu"
  "src/fitting/kernels/specializations/fit_backward_telescope_detector.cu"
)

if(TRACCC_ENABLE_NVTX_PROFILING)
    traccc_add_library(
        nvtx_instrumentation
        TYPE OBJECT
        "src/inst_nvtx.cpp"
    )

    set_target_properties(
        nvtx_instrumentation
        PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )

    target_link_libraries(
        nvtx_instrumentation
        PRIVATE CUDA::nvToolsExt ${CMAKE_DL_LIBS}
    )

    target_compile_options(
        traccc_cuda
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-finstrument-functions>
        $<$<COMPILE_LANGUAGE:CXX>:-finstrument-functions>)

    target_link_libraries(
        traccc_cuda
        PRIVATE
        nvtx_instrumentation
    )
endif()

target_compile_options( traccc_cuda
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> )
target_link_libraries( traccc_cuda
  PUBLIC traccc::core detray::core vecmem::core covfie::core
  PRIVATE CUDA::cudart traccc::device_common vecmem::cuda )

# Set up Thrust specifically for the traccc::cuda library.
thrust_create_target( traccc::cuda_thrust
  HOST CPP
  DEVICE CUDA )
target_link_libraries( traccc::cuda_thrust INTERFACE CUDA::cudart )
target_link_libraries( traccc_cuda PRIVATE traccc::cuda_thrust )

# For CUDA 11 turn on separable compilation. This is necessary for using
# Thrust 2.1.0.
if( ( "${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA" ) AND
    ( "${CMAKE_CUDA_COMPILER_VERSION}" VERSION_LESS "12.0" ) )
  set_target_properties( traccc_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON )
endif()
