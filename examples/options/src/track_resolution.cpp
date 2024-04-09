/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/track_resolution.hpp"

// System include(s).
#include <iostream>

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

track_resolution::track_resolution(po::options_description& desc)
    : interface("Track Ambiguity Resolution Options") {

    m_desc.add_options()("perform-ambiguity-resolution",
                         po::value(&run)->default_value(run),
                         "Perform track ambiguity resolution");
    desc.add(m_desc);
}

std::ostream& track_resolution::print_impl(std::ostream& out) const {

    out << "  Run ambiguity resolution : " << (run ? "yes" : "no");
    return out;
}

}  // namespace traccc::opts
