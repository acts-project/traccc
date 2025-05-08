/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/global_index.hpp"
#include "traccc/edm/device/update_result.hpp"

// Project include(s)
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>

namespace traccc::device {

/// (Event Data) Payload for the @c
/// traccc::device::gather_tracks function
struct gather_tracks_payload {

    /**
     * @brief Update result
     */
    update_result* update_res;

    /**
     * @brief View object to the inverted ids
     */
    vecmem::data::vector_view<const unsigned int> temp_sorted_ids_view;

    /**
     * @brief View object to the sorted track
     */
    vecmem::data::vector_view<unsigned int> sorted_ids_view;
};

/// Function used for updating vectors
///
/// @param[in] globalIndex   The index of the current thread
/// @param[inout] payload      The function call payload
///
TRACCC_DEVICE inline void gather_tracks(global_index_t globalIndex,
                                        const gather_tracks_payload& payload);

}  // namespace traccc::device

// Include the implementation.
#include "./impl/gather_tracks.ipp"
