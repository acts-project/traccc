/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/global_index.hpp"

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_candidate.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Event Data) Payload for the @c traccc::device::prune_tracks function
struct prune_tracks_payload {
    /**
     * @brief View object to the vector of track candidates
     */
    track_candidate_container_types::const_view track_candidates_view;

    /**
     * @brief View object to the vector containing the indices of valid tracks
     */
    vecmem::data::vector_view<const unsigned int> valid_indices_view;

    /**
     * @brief View object to the vector of pruned track candidates
     */
    track_candidate_container_types::view prune_candidates_view;
};

/// Return a new track_candidates based on the criteria in configuration
///
/// @param[in] globalIndex         The index of the current thread
/// @param[inout] payload      The function call payload
///
TRACCC_DEVICE inline void prune_tracks(global_index_t globalIndex,
                                       const prune_tracks_payload& payload);

}  // namespace traccc::device

// Include the implementation.
#include "./impl/prune_tracks.ipp"
