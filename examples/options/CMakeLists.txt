# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the "build" of the traccc::option library.
traccc_add_library( traccc_options options TYPE INTERFACE
  "include/traccc/handle_argument_errors.hpp"
  "include/traccc/seeding_input_options.hpp"
  "include/traccc/full_tracking_input_options.hpp" )
target_link_libraries( traccc_options INTERFACE Boost::program_options)