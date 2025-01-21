# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-compiler-options-cpp )

# Declare the traccc::device_common library.
traccc_add_library( traccc_device_common device_common TYPE INTERFACE
   # General function(s).
   "include/traccc/device/fill_prefix_sum.hpp"
   "include/traccc/device/impl/fill_prefix_sum.ipp"
   "include/traccc/device/make_prefix_sum_buffer.hpp"
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
   "include/traccc/edm/device/sort_key.hpp"
   # Clusterization function(s).
   "include/traccc/clusterization/device/reduce_problem_cell.hpp"
   "include/traccc/clusterization/device/impl/reduce_problem_cell.ipp"
   "include/traccc/clusterization/device/aggregate_cluster.hpp"
   "include/traccc/clusterization/device/impl/aggregate_cluster.ipp"
   "include/traccc/clusterization/device/ccl_kernel_definitions.hpp"
   "include/traccc/clusterization/device/ccl_kernel.hpp"
   # Spacepoint binning function(s).
   "include/traccc/seeding/device/count_grid_capacities.hpp"
   "include/traccc/seeding/device/impl/count_grid_capacities.ipp"
   "include/traccc/seeding/device/populate_grid.hpp"
   "include/traccc/seeding/device/impl/populate_grid.ipp"
   # Seed finding function(s).
   "include/traccc/seeding/device/form_spacepoints.hpp"
   "include/traccc/seeding/device/impl/form_spacepoints.ipp"
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
   "include/traccc/seeding/device/impl/estimate_track_params.ipp"
   # Track finding funtions(s).
   "include/traccc/finding/device/apply_interaction.hpp"
   "include/traccc/finding/device/build_tracks.hpp"
   "include/traccc/finding/device/find_tracks.hpp"
   "include/traccc/finding/device/fill_sort_keys.hpp"
   "include/traccc/finding/device/make_barcode_sequence.hpp"
   "include/traccc/finding/device/propagate_to_next_surface.hpp"
   "include/traccc/finding/device/prune_tracks.hpp"
   "include/traccc/finding/device/impl/apply_interaction.ipp"
   "include/traccc/finding/device/impl/build_tracks.ipp"
   "include/traccc/finding/device/impl/find_tracks.ipp"
   "include/traccc/finding/device/impl/fill_sort_keys.ipp"
   "include/traccc/finding/device/impl/make_barcode_sequence.ipp"
   "include/traccc/finding/device/impl/propagate_to_next_surface.ipp"
   "include/traccc/finding/device/impl/prune_tracks.ipp"
   # Track fitting funtions(s).
   "include/traccc/fitting/device/fit.hpp"
   "include/traccc/fitting/device/impl/fit.ipp"
   "include/traccc/fitting/device/fill_sort_keys.hpp"
   "include/traccc/fitting/device/impl/fill_sort_keys.ipp"
   )
target_link_libraries( traccc_device_common
   INTERFACE traccc::core vecmem::core )
