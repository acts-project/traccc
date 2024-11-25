/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/finding/combinatorial_kalman_filter_algorithm.hpp"

// System include(s).
#include <stdexcept>

namespace traccc::host {

combinatorial_kalman_filter_algorithm::combinatorial_kalman_filter_algorithm(
    const config_type& config)
    : m_config{config} {

    // Check the configuration.
    if (m_config.min_track_candidates_per_track == 0) {
        throw std::invalid_argument(
            "The minimum number of track candidates per track must be at least "
            "1.");
    }
}

}  // namespace traccc::host
