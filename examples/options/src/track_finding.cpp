/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/track_finding.hpp"

// System include(s).
#include <iostream>

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

track_finding::track_finding(po::options_description& desc)
    : interface("Track Finding Options") {

    m_desc.add_options()("track-candidates-range",
                         po::value(&track_candidates_range)
                             ->value_name("MIN:MAX")
                             ->default_value(track_candidates_range),
                         "Range of track candidates number");
    m_desc.add_options()(
        "chi2-max", po::value(&chi2_max)->default_value(chi2_max),
        "Maximum Chi suqare that measurements can be included in the track");
    m_desc.add_options()(
        "nmax_per_seed",
        po::value<unsigned int>(&nmax_per_seed)->default_value(nmax_per_seed),
        "Maximum number of branches which each initial seed can have at a "
        "step.");
    desc.add(m_desc);
}

std::ostream& track_finding::print_impl(std::ostream& out) const {

    out << "  Track candidates range   : " << track_candidates_range << "\n"
        << "  Maximum Chi2             : " << chi2_max << "\n"
        << "  Maximum branches per step: " << nmax_per_seed;
    return out;
}

}  // namespace traccc::opts
