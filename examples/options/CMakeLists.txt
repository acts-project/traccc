# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the "build" of the traccc::option library.
traccc_add_library( traccc_options options TYPE SHARED
  # header files
  "include/traccc/options/common_options.hpp"
  "include/traccc/options/handle_argument_errors.hpp"
  "include/traccc/options/mt_options.hpp"
  "include/traccc/options/options.hpp"
  "include/traccc/options/particle_gen_options.hpp"
  "include/traccc/options/propagation_options.hpp"
  "include/traccc/options/finding_input_options.hpp"
  "include/traccc/options/seeding_input_options.hpp"
  "include/traccc/options/full_tracking_input_options.hpp"   
  "include/traccc/options/throughput_options.hpp"
  # source files
  "src/options/common_options.cpp"
  "src/options/handle_argument_errors.cpp"
  "src/options/mt_options.cpp"
  "src/options/seeding_input_options.cpp"
  "src/options/finding_input_options.cpp"
  "src/options/full_tracking_input_options.cpp"
  "src/options/throughput_options.cpp"
  )
target_link_libraries( traccc_options PUBLIC traccc::io 
                       traccc::performance Boost::program_options)