/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "traccc/edm/track_candidate.hpp"

// VecMem include(s).
#include <vecmem/containers/data/jagged_vector_view.hpp>
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Event Data) Payload for the @c traccc::device::fill_track_candidates
/// function
struct fill_track_candidates_payload {

    /**
     * @brief View object to the input track candidates
     */
    track_candidate_container_types::const_view track_candidates_view;

    /**
     * @brief The number of accepted tracks
     */
    unsigned int n_accepted;

    /**
     * @brief View object to the sorted ids
     */
    vecmem::data::vector_view<const unsigned int> sorted_ids_view;

    /**
     * @brief View object to the output track candidates
     */
    track_candidate_container_types::view res_track_candidates_view;
};

}  // namespace traccc::device
