/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_state.hpp"

// System include(s).
#include <cstddef>

namespace traccc::device {

/// Function used for fitting a track for a given track candidates
///
/// @param[in] globalIndex   The index of the current thread
/// @param[in] det_data      Detector view object
/// @param[in] nav_candidates_buffer Buffer for navigation candidate objects
/// @param[in] track_candidates_view The input track candidates
/// @param[out] track_states_view The output of fitted track states
///
template <typename fitter_t, typename detector_view_t>
TRACCC_HOST_DEVICE inline void fit(
    std::size_t globalIndex, detector_view_t det_data,
    const typename fitter_t::config_type cfg,
    vecmem::data::jagged_vector_view<typename fitter_t::intersection_type>
        nav_candidates_buffer,
    track_candidate_container_types::const_view track_candidates_view,
    track_state_container_types::view track_states_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/fitting/device/impl/fit.ipp"
