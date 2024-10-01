# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-compiler-options-cpp )

# Set up the "build" of the traccc::core library.
traccc_add_library( traccc_core core TYPE SHARED
  # Common definitions.
  "include/traccc/definitions/track_parametrization.hpp"
  "include/traccc/definitions/math.hpp"
  "include/traccc/definitions/primitives.hpp"
  "include/traccc/definitions/common.hpp"
  "include/traccc/definitions/qualifiers.hpp"
  # Event data model.
  "include/traccc/edm/details/container_base.hpp"
  "include/traccc/edm/details/container_element.hpp"
  "include/traccc/edm/details/device_container.hpp"
  "include/traccc/edm/details/host_container.hpp"
  "include/traccc/edm/cluster.hpp"
  "include/traccc/edm/spacepoint.hpp"
  "include/traccc/edm/measurement.hpp"
  "include/traccc/edm/particle.hpp"
  "include/traccc/edm/track_parameters.hpp"
  "include/traccc/edm/container.hpp"
  "include/traccc/edm/internal_spacepoint.hpp"
  "include/traccc/edm/seed.hpp"
  "include/traccc/edm/track_candidate.hpp"
  "include/traccc/edm/track_state.hpp"
  "include/traccc/edm/cell.hpp"
  # Geometry description.
  "include/traccc/geometry/detector.hpp"
  "include/traccc/geometry/module_map.hpp"
  "include/traccc/geometry/geometry.hpp"
  "include/traccc/geometry/pixel_data.hpp"
  "include/traccc/geometry/silicon_detector_description.hpp"
  # Utilities.
  "include/traccc/utils/algorithm.hpp"
  "include/traccc/utils/type_traits.hpp"
  "include/traccc/utils/memory_resource.hpp"
  "include/traccc/utils/seed_generator.hpp"
  "include/traccc/utils/subspace.hpp"
  # Clusterization algorithmic code.
  "include/traccc/clusterization/details/sparse_ccl.hpp"
  "include/traccc/clusterization/impl/sparse_ccl.ipp"
  "include/traccc/clusterization/sparse_ccl_algorithm.hpp"
  "src/clusterization/sparse_ccl_algorithm.cpp"
  "include/traccc/clusterization/details/measurement_creation.hpp"
  "include/traccc/clusterization/impl/measurement_creation.ipp"
  "include/traccc/clusterization/measurement_creation_algorithm.hpp"
  "src/clusterization/measurement_creation_algorithm.cpp"
  "include/traccc/clusterization/measurement_sorting_algorithm.hpp"
  "src/clusterization/measurement_sorting_algorithm.cpp"
  "include/traccc/clusterization/clusterization_algorithm.hpp"
  "src/clusterization/clusterization_algorithm.cpp"
  # Finding algorithmic code
  "include/traccc/finding/candidate_link.hpp"
  "include/traccc/finding/ckf_aborter.hpp"
  "include/traccc/finding/finding_algorithm.hpp"
  "include/traccc/finding/finding_algorithm.ipp"
  "include/traccc/finding/finding_config.hpp"
  "include/traccc/finding/interaction_register.hpp"
  # Fitting algorithmic code
  "include/traccc/fitting/kalman_filter/gain_matrix_smoother.hpp"
  "include/traccc/fitting/kalman_filter/gain_matrix_updater.hpp"
  "include/traccc/fitting/kalman_filter/kalman_actor.hpp"
  "include/traccc/fitting/kalman_filter/kalman_fitter.hpp"
  "include/traccc/fitting/kalman_filter/kalman_step_aborter.hpp"
  "include/traccc/fitting/kalman_filter/statistics_updater.hpp"
  "include/traccc/fitting/fitting_algorithm.hpp"
  # Seed finding algorithmic code.
  "include/traccc/seeding/detail/lin_circle.hpp"
  "include/traccc/seeding/detail/doublet.hpp"
  "include/traccc/seeding/detail/triplet.hpp"
  "include/traccc/seeding/detail/singlet.hpp"
  "include/traccc/seeding/detail/seeding_config.hpp"
  "include/traccc/seeding/detail/spacepoint_grid.hpp"
  "include/traccc/seeding/seed_selecting_helper.hpp"
  "include/traccc/seeding/seed_filtering.hpp"
  "src/seeding/seed_filtering.cpp"
  "include/traccc/seeding/seeding_algorithm.hpp"
  "src/seeding/seeding_algorithm.cpp"
  "include/traccc/seeding/track_params_estimation_helper.hpp"
  "include/traccc/seeding/doublet_finding_helper.hpp"
  "include/traccc/seeding/spacepoint_binning_helper.hpp"
  "include/traccc/seeding/track_params_estimation.hpp"
  "src/seeding/track_params_estimation.cpp"
  "include/traccc/seeding/triplet_finding_helper.hpp"
  "include/traccc/seeding/doublet_finding.hpp"
  "include/traccc/seeding/triplet_finding.hpp"
  "include/traccc/seeding/seed_finding.hpp"
  "src/seeding/seed_finding.cpp"
  "include/traccc/seeding/spacepoint_binning.hpp"
  "src/seeding/spacepoint_binning.cpp"
  "include/traccc/seeding/spacepoint_formation_algorithm.hpp"
  "include/traccc/seeding/impl/spacepoint_formation_algorithm.ipp"
  "include/traccc/seeding/detail/spacepoint_formation.hpp"
  "include/traccc/seeding/impl/spacepoint_formation.ipp"
  # Ambiguity resolution
  "include/traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
  "src/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.cpp" )
target_link_libraries( traccc_core
  PUBLIC Eigen3::Eigen vecmem::core detray::core detray::detectors traccc::Thrust
         traccc::algebra )

# Prevent Eigen from getting confused when building code for a
# CUDA or HIP backend with SYCL.
target_compile_definitions( traccc_core
  PUBLIC $<$<COMPILE_LANGUAGE:SYCL>:EIGEN_NO_CUDA EIGEN_NO_HIP> )

# Set the algebra-plugins plugin to use.
message(STATUS "Building with plugin type: " ${TRACCC_ALGEBRA_PLUGINS})
target_compile_definitions(traccc_core PUBLIC ALGEBRA_PLUGINS_INCLUDE_${TRACCC_ALGEBRA_PLUGINS})
