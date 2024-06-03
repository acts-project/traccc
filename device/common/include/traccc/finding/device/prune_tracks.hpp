/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_candidate.hpp"

namespace traccc::device {

/// Return a new track_candidates based on the criteria in configuration
///
/// @param[in] globalIndex         The index of the current thread
/// @param[in] track_candidates_view  Track candidate container view
/// @param[in] valid_indices_view   Valid indices meeting criteria
/// @param[out] prune_candidates_view  Track candidate container view
TRACCC_DEVICE inline void prune_tracks(
    std::size_t globalIndex,
    track_candidate_container_types::const_view track_candidates_view,
    vecmem::data::vector_view<const unsigned int> valid_indices_view,
    track_candidate_container_types::view prune_candidates_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/finding/device/impl/prune_tracks.ipp"