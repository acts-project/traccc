/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/global_index.hpp"

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/finding_config.hpp"

// VecMem include(s).
#include <vecmem/containers/data/jagged_vector_view.hpp>
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Event Data) Payload for the @c traccc::device::build_tracks function
struct build_tracks_payload {
    /**
     * @brief View object to the vector of measurements
     *
     * @warning Measurements on the same surface must be adjacent
     */
    measurement_collection_types::const_view measurements_view;

    /**
     * @brief View object to the vector of measurements
     */
    bound_track_parameters_collection_types::const_view seeds_view;

    /**
     * @brief View object to the vector of candidate links
     */
    vecmem::data::jagged_vector_view<const candidate_link> links_view;

    /**
     * @brief View object to the vector of tips
     */
    vecmem::data::vector_view<const typename candidate_link::link_index_type>
        tips_view;

    /**
     * @brief View object to the vector of track candidates
     */
    track_candidate_container_types::view track_candidates_view;

    /**
     * @brief View object to the vector of indices meeting the selection
     * criteria
     */
    vecmem::data::vector_view<unsigned int> valid_indices_view;

    /**
     * @brief The number of valid tracks meeting criteria
     */
    unsigned int* n_valid_tracks;
};

/// Function for building full tracks from the link container:
/// The full tracks are built using the link container and tip link container.
/// Since every link holds an information of the link from the previous step,
/// we can build a full track by iterating from a tip link backwardly.
///
/// @param[in] globalIndex         The index of the current thread
/// @param[in] cfg                    Track finding config object
/// @param[inout] payload      The function call payload
///
TRACCC_DEVICE inline void build_tracks(global_index_t globalIndex,
                                       const finding_config& cfg,
                                       const build_tracks_payload& payload);

}  // namespace traccc::device

// Include the implementation.
#include "./impl/build_tracks.ipp"
