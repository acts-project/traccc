/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Event Data) Payload for the @c
/// traccc::device::reset_status function
struct reset_status_payload {

    /**
     * @brief If the current iteration is the first iteration
     */
    int* is_first_iteration;

    /**
     * @brief Whether to terminate the calculation
     */
    int* terminate;

    /**
     * @brief The number of max shared
     */
    unsigned int* max_shared;

    /**
     * @brief The number of updated tracks
     */
    unsigned int* n_updated_tracks;
};

}  // namespace traccc::device
