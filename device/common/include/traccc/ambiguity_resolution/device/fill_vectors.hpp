/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/track_candidate.hpp"

// Project include(s).
#include "traccc/definitions/primitives.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Event Data) Payload for the @c traccc::device::fill_vectors function
struct fill_vectors_payload {

    /**
     * @brief View object to the input track candidates
     */
    track_candidate_container_types::const_view track_candidates_view;

    /**
     * @brief View object to the vector of measured ids per track
     */
    vecmem::data::jagged_vector_view<std::size_t> meas_ids_view;

    /**
     * @brief View object to the measurement ids in flat vector
     */
    vecmem::data::vector_view<std::size_t> flat_meas_ids_view;

    /**
     * @brief View object to the vector of pvalues
     */
    vecmem::data::vector_view<traccc::scalar> pvals_view;

    /**
     * @brief View object to the number of measurements per track
     */
    vecmem::data::vector_view<std::size_t> n_meas_view;

    /**
     * @brief View object to the status of track acceptance
     */
    vecmem::data::vector_view<int> status_view;
};

}  // namespace traccc::device
