/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/track_resolution.hpp"

#include "traccc/examples/utils/printable.hpp"

// System include(s).
#include <format>

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

track_resolution::track_resolution()
    : interface("Track Ambiguity Resolution Options") {

    m_desc.add_options()("min-meas-per-track",
                         po::value(&m_config.min_meas_per_track)
                             ->default_value(m_config.min_meas_per_track),
                         "Min number of measurements per track");
    m_desc.add_options()(
        "min-shared-meas-for-competition",
        po::value(&m_config.min_shared_meas_for_competition)
            ->default_value(m_config.min_shared_meas_for_competition),
        "Min number of shared measurements for competition");
}

track_resolution::operator resolution_config() const {
    return m_config;
}

std::unique_ptr<configuration_printable> track_resolution::as_printable()
    const {
    auto cat = std::make_unique<configuration_category>(m_description);

    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Min number of measurements per track",
        std::to_string(m_config.min_meas_per_track)));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Max shared meas for competition",
        std::to_string(m_config.min_shared_meas_for_competition)));
    return cat;
}
}  // namespace traccc::opts
