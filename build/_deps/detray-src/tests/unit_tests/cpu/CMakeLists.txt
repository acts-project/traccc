# Detray library, part of the ACTS project (R&D line)
#
# (c) 2023-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the test(s) that are algebra agnostic.
detray_add_unit_test(cpu
   "core/containers.cpp"
   "core/typed_index.cpp"
   "core/pdg_particle.cpp"
   "geometry/barcode.cpp"
   "grid2/populator.cpp"
   "propagator/actor_chain.cpp"
   "utils/find_bounds.cpp"
   "utils/hash_tree.cpp"
   "utils/invalid_values.cpp"
   "utils/ranges.cpp"
   "utils/tuple_helpers.cpp"
   "utils/type_list.cpp"
   "utils/sort.cpp"
   LINK_LIBRARIES GTest::gtest GTest::gtest_main detray::test_utils detray::core
)

# Macro setting up the CPU tests for a specific algebra plugin.
macro(detray_add_cpu_test algebra)
    # Build the test executable.
    detray_add_unit_test(cpu_${algebra}
       "builders/detector_builder.cpp"
       "builders/grid_builder.cpp"
       "builders/homogeneous_volume_material_builder.cpp"
       "builders/homogeneous_material_builder.cpp"
       "builders/material_map_builder.cpp"
       "builders/volume_builder.cpp"
       "core/detector.cpp"
       "core/mask_store.cpp"
       "core/transform_store.cpp"
       "detectors/telescope_detector.cpp"
       "detectors/toy_detector.cpp"
       "detectors/wire_chamber.cpp"
       "geometry/coordinates/cartesian2D.cpp"
       "geometry/coordinates/cartesian3D.cpp"
       "geometry/coordinates/cylindrical2D.cpp"
       "geometry/coordinates/cylindrical3D.cpp"
       "geometry/coordinates/line2D.cpp"
       "geometry/coordinates/polar2D.cpp"
       "geometry/masks/annulus2D.cpp"
       "geometry/masks/cylinder.cpp"
       "geometry/masks/line.cpp"
       "geometry/masks/rectangle2D.cpp"
       "geometry/masks/ring2D.cpp"
       "geometry/masks/single3D.cpp"
       "geometry/masks/trapezoid2D.cpp"
       "geometry/masks/unbounded.cpp"
       "geometry/masks/unmasked.cpp"
       "geometry/tracking_surface.cpp"
       "geometry/tracking_volume.cpp"
       "grid2/axis.cpp"
       "grid2/grid2.cpp"
       "grid2/serializer.cpp"
       "material/bethe_equation.cpp"
       "material/bremsstrahlung.cpp"
       "material/material_maps.cpp"
       "material/materials.cpp"
       "material/stopping_power_derivative.cpp"
       "material/stopping_power.cpp"
       "navigation/intersection/cuboid_intersector.cpp"
       "navigation/intersection/cylinder_intersector.cpp"
       "navigation/intersection/helix_intersector.cpp"
       "navigation/intersection/helix_trajectory.cpp"
       "navigation/intersection/intersection_kernel.cpp"
       "navigation/intersection/intersection2D.cpp"
       "navigation/intersection/line_intersector.cpp"
       "navigation/intersection/plane_intersector.cpp"
       "navigation/brute_force_finder.cpp"
       "navigation/volume_graph.cpp"
       "navigation/navigator.cpp"
       "propagator/covariance_transport.cpp"
       "propagator/jacobian_cartesian.cpp"
       "propagator/jacobian_cylindrical.cpp"
       "propagator/jacobian_line.cpp"
       "propagator/jacobian_polar.cpp"
       "propagator/line_stepper.cpp"
       "propagator/rk_stepper.cpp"
       "simulation/landau_sampling.cpp"
       "simulation/detector_scanner.cpp"
       "simulation/scattering.cpp"
       "simulation/track_generators.cpp"
       "tracks/bound_track_parameters.cpp"
       "tracks/free_track_parameters.cpp"
       "utils/grids/axis.cpp"
       "utils/grids/grid_collection.cpp"
       "utils/grids/grid.cpp"
       "utils/grids/populators.cpp"
       "utils/grids/serializers.cpp"
       "utils/bounding_volume.cpp"
       "utils/curvilinear_frame.cpp"
       "utils/axis_rotation.cpp"
       "utils/matrix_helper.cpp"
       "utils/quadratic_equation.cpp"
       "utils/unit_vectors.cpp"
       LINK_LIBRARIES GTest::gtest GTest::gtest_main detray::core_${algebra}
       covfie::core vecmem::core detray::io detray::test_utils
    )
endmacro()

# Build the array tests.
detray_add_cpu_test(array)

# Build the Eigen tests.
if(DETRAY_EIGEN_PLUGIN)
    detray_add_cpu_test(eigen)
endif()

# Build the SMatrix tests.
if(DETRAY_SMATRIX_PLUGIN)
    detray_add_cpu_test(smatrix)
endif()

# Build the Vc AoS tests.
if(DETRAY_VC_AOS_PLUGIN)
    detray_add_cpu_test(vc_aos)
endif()
