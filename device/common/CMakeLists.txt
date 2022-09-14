# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-compiler-options-cpp )

# Declare the traccc::device_common library.
traccc_add_library( traccc_device_common device_common TYPE SHARED
   # General function(s).
   "include/traccc/device/get_prefix_sum.hpp"
   "include/traccc/device/impl/get_prefix_sum.ipp"
   "include/traccc/device/fill_prefix_sum.hpp"
   "include/traccc/device/impl/fill_prefix_sum.ipp"
   "src/get_prefix_sum.cpp"
   "include/traccc/device/make_prefix_sum_buffer.hpp"
   "src/make_prefix_sum_buffer.cpp"
   # EDM class(es).
   "include/traccc/edm/device/doublet_counter.hpp"
   "include/traccc/edm/device/triplet_counter.hpp"
   # Clusterization function(s).
   "include/traccc/clusterization/device/find_clusters.hpp"
   "include/traccc/clusterization/device/impl/find_clusters.ipp"
   "include/traccc/clusterization/device/count_cluster_cells.hpp"
   "include/traccc/clusterization/device/impl/count_cluster_cells.ipp"
   "include/traccc/clusterization/device/connect_components.hpp"
   "include/traccc/clusterization/device/impl/connect_components.ipp"
   "include/traccc/clusterization/device/create_measurements.hpp"
   "include/traccc/clusterization/device/impl/create_measurements.ipp"
   "include/traccc/clusterization/device/form_spacepoints.hpp"
   "include/traccc/clusterization/device/impl/form_spacepoints.ipp"
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
   "include/traccc/seeding/device/make_doublet_counter_buffer.hpp"
   "src/seeding/make_doublet_counter_buffer.cpp"
   "include/traccc/seeding/device/make_doublet_buffers.hpp"
   "src/seeding/make_doublet_buffers.cpp"
   "include/traccc/seeding/device/count_triplets.hpp"
   "include/traccc/seeding/device/impl/count_triplets.ipp"
   "include/traccc/seeding/device/find_triplets.hpp"
   "include/traccc/seeding/device/impl/find_triplets.ipp"
   "include/traccc/seeding/device/make_triplet_counter_buffer.hpp"
   "src/seeding/make_triplet_counter_buffer.cpp"
   "include/traccc/seeding/device/make_triplet_buffer.hpp"
   "src/seeding/make_triplet_buffer.cpp"
   "include/traccc/seeding/device/update_triplet_weights.hpp"
   "include/traccc/seeding/device/impl/update_triplet_weights.ipp"
   "include/traccc/seeding/device/select_seeds.hpp"
   "include/traccc/seeding/device/impl/select_seeds.ipp" )
target_link_libraries( traccc_device_common
   PUBLIC traccc::Thrust traccc::core vecmem::core )
