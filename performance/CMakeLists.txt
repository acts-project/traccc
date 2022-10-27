# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the build of the traccc::performance library.
traccc_add_library( traccc_performance performance TYPE SHARED
   # Efficiency calculation code.
   "include/traccc/efficiency/duplication_plot_tool.hpp"
   "include/traccc/efficiency/eff_plot_tool.hpp"
   "include/traccc/efficiency/helpers.hpp"
   "include/traccc/efficiency/seeding_performance_writer.hpp"
   "include/traccc/efficiency/track_classification.hpp"
   "include/traccc/event/event_map.hpp"
   # Value/object comparison code.
   "include/traccc/performance/details/is_same_angle.hpp"
   "src/performance/details/is_same_angle.cpp"
   "include/traccc/performance/details/is_same_scalar.hpp"
   "src/performance/details/is_same_scalar.cpp"
   "include/traccc/performance/details/is_same_object.hpp"
   "include/traccc/performance/impl/is_same_object.ipp"
   "include/traccc/performance/impl/is_same_seed.ipp"
   "include/traccc/performance/impl/is_same_measurement.ipp"
   "include/traccc/performance/impl/is_same_spacepoint.ipp"
   "include/traccc/performance/impl/is_same_track_parameters.ipp"
   "src/performance/details/is_same_object.cpp"
   "include/traccc/performance/details/comparator_factory.hpp"
   "include/traccc/performance/impl/comparator_factory.ipp"
   "include/traccc/performance/impl/seed_comparator_factory.ipp"
   "src/performance/details/comparator_factory.cpp"
   # Collection/container comparison code.
   "include/traccc/performance/collection_comparator.hpp"
   "include/traccc/performance/impl/collection_comparator.ipp"
   "include/traccc/performance/container_comparator.hpp"
   "include/traccc/performance/impl/container_comparator.ipp" )
target_link_libraries( traccc_performance
   PUBLIC ROOT::Core ROOT::RIO ROOT::Tree ROOT::Hist traccc::core )
