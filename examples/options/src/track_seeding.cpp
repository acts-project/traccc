/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/track_seeding.hpp"

namespace traccc::opts {

track_seeding::track_seeding(boost::program_options::options_description& desc)
    : interface("Track Seeding Options") {

    desc.add(m_desc);
}

}  // namespace traccc::opts
