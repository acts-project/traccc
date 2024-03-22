/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/finding_input_options.hpp"

namespace traccc {

/// Convenience namespace shorthand
namespace po = boost::program_options;

finding_input_options::finding_input_options(po::options_description& desc) {

    desc.add_options()("track-candidates-range",
                       po::value(&track_candidates_range)
                           ->value_name("MIN:MAX")
                           ->default_value(track_candidates_range),
                       "Range of track candidates number");
    desc.add_options()(
        "chi2-max", po::value(&chi2_max)->default_value(chi2_max),
        "Maximum Chi suqare that measurements can be included in the track");
    desc.add_options()(
        "nmax_per_seed",
        po::value<unsigned int>(&nmax_per_seed)->default_value(nmax_per_seed),
        "Maximum number of branches which each initial seed can have at a "
        "step.");
}

void finding_input_options::read(const po::variables_map&) {}

std::ostream& operator<<(std::ostream& out, const finding_input_options& opt) {

    out << ">>> Track finding options <<<\n"
        << "  Track candidates range    : " << opt.track_candidates_range
        << "\n"
        << "  Maximum Chi2              : " << opt.chi2_max << "\n"
        << "  Maximum branches per step : " << opt.nmax_per_seed;
    return out;
}

}  // namespace traccc
