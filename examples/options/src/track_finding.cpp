/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/track_finding.hpp"

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

/// Description of this option group
static const char* description = "Track Finding Options";

track_finding::track_finding(po::options_description& desc)
    : m_desc{description} {

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

void track_finding::read(const po::variables_map&) {}

std::ostream& operator<<(std::ostream& out, const track_finding& opt) {

    out << ">>> " << description << " <<<\n"
        << "  Track candidates range    : " << opt.track_candidates_range
        << "\n"
        << "  Maximum Chi2              : " << opt.chi2_max << "\n"
        << "  Maximum branches per step : " << opt.nmax_per_seed;
    return out;
}

}  // namespace traccc::opts
