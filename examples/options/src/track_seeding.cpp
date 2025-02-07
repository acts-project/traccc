/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/track_seeding.hpp"

#include "traccc/examples/utils/printable.hpp"

namespace traccc::opts {

track_seeding::track_seeding() : interface("Track Seeding Options") {}

std::unique_ptr<configuration_printable> track_seeding::as_printable() const {
    auto cat = std::make_unique<configuration_category>(m_description);
    return cat;
}
}  // namespace traccc::opts
