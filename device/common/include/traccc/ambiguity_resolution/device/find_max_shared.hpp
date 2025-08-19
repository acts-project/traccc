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
/// traccc::device::find_max_shared function
struct find_max_shared_payload {

    /**
     * @brief View object to the vector of sorted track ids
     */
    vecmem::data::vector_view<const unsigned int> sorted_ids_view;

    /**
     * @brief The number of accepted tracks
     */
    unsigned int* n_accepted;

    /**
     * @brief View object to the vector of number of shared measurements
     */
    vecmem::data::vector_view<const unsigned int> n_shared_view;

    /**
     * @brief Whether to terminate the calculation
     */
    int* terminate;

    /**
     * @brief The number of max shared
     */
    unsigned int* max_shared;
};

}  // namespace traccc::device
