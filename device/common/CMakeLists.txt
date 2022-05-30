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
   "src/get_prefix_sum.cpp"
   # EDM class(es).
   "include/traccc/edm/device/doublet_counter.hpp"
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
   "src/seeding/make_doublet_buffers.cpp" )
target_link_libraries( traccc_device_common
   PUBLIC traccc::core vecmem::core )
