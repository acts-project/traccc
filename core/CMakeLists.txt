# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the "build" of the traccc::core library.
traccc_add_library( traccc_core core TYPE SHARED
  # Common definitions.
  "include/traccc/definitions/track_parametrization.hpp"
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
  "include/traccc/edm/track_parameters.hpp"
  "include/traccc/edm/container.hpp"
  "include/traccc/edm/internal_spacepoint.hpp"
  "include/traccc/edm/seed.hpp"
  "include/traccc/edm/cell.hpp"
  # Geometry description.
  "include/traccc/geometry/module_map.hpp"
  "include/traccc/geometry/geometry.hpp"
  "include/traccc/geometry/pixel_data.hpp"
  # Utilities.
  "include/traccc/utils/algorithm.hpp"
  "include/traccc/utils/type_traits.hpp"
  "include/traccc/utils/unit_vectors.hpp"
  "include/traccc/utils/memory_resource.hpp"
  # Clusterization algorithmic code.
  "include/traccc/clusterization/detail/measurement_creation_helper.hpp"
  "include/traccc/clusterization/detail/sparse_ccl.hpp"
  "include/traccc/clusterization/component_connection.hpp"
  "src/clusterization/component_connection.cpp"
  "include/traccc/clusterization/clusterization_algorithm.hpp"
  "src/clusterization/clusterization_algorithm.cpp"
  "include/traccc/clusterization/spacepoint_formation.hpp"
  "src/clusterization/spacepoint_formation.cpp"
  "include/traccc/clusterization/measurement_creation.hpp"
  "src/clusterization/measurement_creation.cpp"
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
  "src/seeding/spacepoint_binning.cpp" )
target_link_libraries( traccc_core
  PUBLIC Eigen3::Eigen vecmem::core detray::core ActsCore ActsPluginJson
         traccc::Thrust traccc::algebra )
