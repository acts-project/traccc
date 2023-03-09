# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-compiler-options-cpp )

# Declare the traccc::device_common library.
traccc_add_library( traccc_device_common device_common TYPE SHARED
   # General function(s).
   "include/traccc/device/fill_prefix_sum.hpp"
   "include/traccc/device/impl/fill_prefix_sum.ipp"
   "include/traccc/device/make_prefix_sum_buffer.hpp"
   "src/make_prefix_sum_buffer.cpp"
   # General algorithm(s).
   "include/traccc/device/container_h2d_copy_alg.hpp"
   "include/traccc/device/impl/container_h2d_copy_alg.ipp"
   "include/traccc/device/container_d2h_copy_alg.hpp"
   "include/traccc/device/impl/container_d2h_copy_alg.ipp"
   # EDM class(es).
   "include/traccc/edm/device/seeding_global_counter.hpp"
   "include/traccc/edm/device/doublet_counter.hpp"
   "include/traccc/edm/device/triplet_counter.hpp"
   "include/traccc/edm/device/device_doublet.hpp"
   "include/traccc/edm/device/device_triplet.hpp"
   # Clusterization function(s).
   "include/traccc/clusterization/device/form_spacepoints.hpp"
   "include/traccc/clusterization/device/impl/form_spacepoints.ipp"
   "include/traccc/clusterization/device/reduce_problem_cell.hpp"
   "include/traccc/clusterization/device/impl/reduce_problem_cell.ipp"
   "include/traccc/clusterization/device/aggregate_cluster.hpp"
   "include/traccc/clusterization/device/impl/aggregate_cluster.ipp"
   # Spacepoint binning function(s).
   "include/traccc/seeding/device/count_grid_capacities.hpp"
   "include/traccc/seeding/device/impl/count_grid_capacities.ipp"
   "include/traccc/seeding/device/populate_grid.hpp"
   "include/traccc/seeding/device/impl/populate_grid.ipp"
   # Seed finding function(s).
   "include/traccc/seeding/device/count_doublets.hpp"
   "include/traccc/seeding/device/impl/count_doublets.ipp"
   "include/traccc/seeding/device/find_doublets.hpp"
   "include/traccc/seeding/device/impl/find_doublets.ipp"
   "include/traccc/seeding/device/count_triplets.hpp"
   "include/traccc/seeding/device/impl/count_triplets.ipp"
   "include/traccc/seeding/device/reduce_triplet_counts.hpp"
   "include/traccc/seeding/device/impl/reduce_triplet_counts.ipp"
   "include/traccc/seeding/device/find_triplets.hpp"
   "include/traccc/seeding/device/impl/find_triplets.ipp"
   "include/traccc/seeding/device/update_triplet_weights.hpp"
   "include/traccc/seeding/device/impl/update_triplet_weights.ipp"
   "include/traccc/seeding/device/select_seeds.hpp"
   "include/traccc/seeding/device/impl/select_seeds.ipp"
   # Track parameters estimation function(s).
   "include/traccc/seeding/device/estimate_track_params.hpp"
   "include/traccc/seeding/device/impl/estimate_track_params.ipp" )
target_link_libraries( traccc_device_common
   PUBLIC traccc::Thrust traccc::core vecmem::core )
