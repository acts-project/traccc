# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the "build" of the traccc::option library.
traccc_add_library( traccc_options options TYPE SHARED
  # header files
  "include/traccc/options/handle_argument_errors.hpp"
  "include/traccc/options/seeding_input_options.hpp"
  "include/traccc/options/full_tracking_input_options.hpp" 
  # source files
  "src/options/seeding_input_options.cpp"
  "src/options/full_tracking_input_options.cpp"
  )
target_link_libraries( traccc_options PUBLIC Boost::program_options)