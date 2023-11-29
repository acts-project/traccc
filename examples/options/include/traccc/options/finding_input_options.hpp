/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/options.hpp"

// Boost
#include <boost/program_options.hpp>

namespace traccc {

namespace po = boost::program_options;

template <typename scalar_t>
struct finding_input_config {
    Reals<unsigned int, 2> track_candidates_range;
    scalar_t chi2_max;

    finding_input_config(po::options_description& desc) {

        desc.add_options()("track-candidates-range",
                           po::value<Reals<unsigned int, 2>>()
                               ->value_name("MIN:MAX")
                               ->default_value({3, 10000}),
                           "Range of track candidates number");
        desc.add_options()(
            "chi2-max",
            po::value<scalar_t>()->value_name("chi2-max")->default_value(30.f),
            "Maximum Chi suqare that measurements can be included in the "
            "track");
    }
    void read(const po::variables_map& vm) {
        track_candidates_range =
            vm["track-candidates-range"].as<Reals<unsigned int, 2>>();
        chi2_max = vm["chi2-max"].as<scalar_t>();
    }
};

}  // namespace traccc