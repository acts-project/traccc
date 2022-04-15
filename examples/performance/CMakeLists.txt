# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the "build" of the traccc::performance library.
traccc_add_library( traccc_performance performance TYPE INTERFACE
  #efficiency
  "include/traccc/efficiency/seeding_performance_writer.hpp" 
  "include/traccc/efficiency/duplication_plot_tool.hpp" 
  "include/traccc/efficiency/eff_plot_tool.hpp" 
  "include/traccc/efficiency/helpers.hpp"
  "include/traccc/efficiency/track_classification.hpp" 
  )
target_link_libraries( traccc_performance
  INTERFACE traccc::core traccc::io ROOT::RIO ROOT::Hist ROOT::Tree )