/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/global_index.hpp"

// Project include(s)
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>

namespace traccc::device {

/// (Event Data) Payload for the @c
/// traccc::device::find_lower_bounds_payload function
struct find_lower_bounds_payload {

    /**
     * @brief The number of updated tracks
     */
    unsigned int n_updated_tracks;

    /**
     * @brief View object to the vector of affected track IDs
     */
    vecmem::data::vector_view<const std::size_t> updated_tracks_view;

    /**
     * @brief View object to the sorted track ids
     */
    vecmem::data::vector_view<std::size_t> sorted_ids_view;

    /**
     * @brief View object to the lower bounds
     */
    vecmem::data::vector_view<std::size_t> lower_bounds_view;
};

/// Function used for updating vectors
///
/// @param[in] globalIndex   The index of the current thread
/// @param[inout] payload      The function call payload
///
TRACCC_HOST_DEVICE inline void find_lower_bounds_payload(
    global_index_t globalIndex,
    const find_lower_bounds_payload_payload& payload);

}  // namespace traccc::device

// Include the implementation.
#include "./impl/find_lower_bounds_payload.ipp"
