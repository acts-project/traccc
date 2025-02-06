/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/track_resolution.hpp"

#include "details/configuration_category.hpp"
#include "details/configuration_value.hpp"

// System include(s).
#include <format>

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

track_resolution::track_resolution()
    : interface("Track Ambiguity Resolution Options") {

    m_desc.add_options()("perform-ambiguity-resolution",
                         po::value(&run)->default_value(run),
                         "Perform track ambiguity resolution");
}

std::unique_ptr<configuration_printable> track_resolution::as_printable()
    const {

    auto result = std::make_unique<configuration_category>(m_description);

    result->add_child(std::make_unique<configuration_value>(
        "Run ambiguity resolution", std::format("{}", run)));

    return result;
}
}  // namespace traccc::opts
