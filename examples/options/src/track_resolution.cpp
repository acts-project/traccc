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

/// Description of this option group
static const char* description = "Track Ambiguity Resolution Options";

track_resolution::track_resolution(po::options_description& desc)
    : m_desc{description} {

    m_desc.add_options()("perform-ambiguity-resolution",
                         po::value(&run)->default_value(run),
                         "Perform track ambiguity resolution");
    desc.add(m_desc);
}

void track_resolution::read(const po::variables_map&) {}

std::ostream& operator<<(std::ostream& out, const track_resolution& opts) {

    out << ">>> " << description << " <<<\n"
        << "  Run ambiguity resolution : " << (opts.run ? "yes" : "no");
    return out;
}

}  // namespace traccc::opts
