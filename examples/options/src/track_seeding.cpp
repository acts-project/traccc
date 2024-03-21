/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/track_seeding.hpp"

// System include(s).
#include <iostream>

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

/// Description of this option group
static const char* description = "Track Seeding Options";

track_seeding::track_seeding(po::options_description& desc)
    : m_desc{description} {

    desc.add(m_desc);
}

void track_seeding::read(const po::variables_map&) {}

std::ostream& operator<<(std::ostream& out, const track_seeding&) {

    out << ">>> " << description << " <<<\n"
        << "  None";
    return out;
}

}  // namespace traccc::opts
