/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/cuda/finding/combinatorial_kalman_filter_algorithm.hpp"

// System include(s).
#include <stdexcept>

namespace traccc::cuda {

combinatorial_kalman_filter_algorithm::combinatorial_kalman_filter_algorithm(
    const config_type& config, const traccc::memory_resource& mr,
    vecmem::copy& copy, stream& str, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)),
      m_config{config},
      m_mr{mr},
      m_copy{copy},
      m_stream{str} {

    // Check the configuration.
    if (m_config.min_track_candidates_per_track == 0) {
        throw std::invalid_argument(
            "The minimum number of track candidates per track must be at least "
            "1.");
    }
}

}  // namespace traccc::cuda
