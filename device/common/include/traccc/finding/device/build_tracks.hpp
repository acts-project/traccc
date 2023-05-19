/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"

namespace traccc::device {

/// Function for building full tracks from the link container:
/// The full tracks are built using the link container and tip link container.
/// Since every link holds an information of the link from the previous step,
/// we can build a full track by iterating from a tip link backwardly.
///
/// @param[in] globalIndex         The index of the current thread
/// @param[in] measurements_view   Measurements container view
/// @param[in] seeds_view          Seed container view
/// @param[in] link_view           Link container view
/// @param[in] param_to_link_view  Container for param index -> link index
/// @param[in] tips_view           Tip link container view
/// @param[out] track_candidates_view  Track candidate container view

TRACCC_DEVICE inline void build_tracks(
    std::size_t globalIndex,
    measurement_container_types::const_view measurements_view,
    bound_track_parameters_collection_types::const_view seeds_view,
    vecmem::data::jagged_vector_view<const candidate_link> links_view,
    vecmem::data::jagged_vector_view<const unsigned int> param_to_link_view,
    vecmem::data::vector_view<const typename candidate_link::link_index_type>
        tips_view,
    track_candidate_container_types::view track_candidates_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/finding/device/impl/build_tracks.ipp"