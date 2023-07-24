/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// options
#include "traccc/options/finding_input_options.hpp"

traccc::finding_input_config::finding_input_config(
    po::options_description& desc) {

    desc.add_options()("track_candidates_range",
                       po::value<Reals<unsigned int, 2>>()
                           ->value_name("MIN:MAX")
                           ->default_value({6, 30}),
                       "Range of track candidates number");
}

void traccc::finding_input_config::read(const po::variables_map& vm) {
    track_candidates_range =
        vm["track_candidates_range"].as<Reals<unsigned int, 2>>();
}
