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

/// Function applying the Pre material interaction to tracks spawned by bound
/// track parameters
///
/// @param[in] globalIndex     The index of the current thread
/// @param[in] det_data        Detector view object
/// @param[in] nav_candidates_buffer  Buffer for navigation candidates
/// @param[in] n_params        The number of parameters (or tracks)
/// @param[out] params_view    Collection of output bound track_parameters
///
template <typename detector_t>
TRACCC_DEVICE inline void apply_interaction(
    std::size_t globalIndex, typename detector_t::detector_view_type det_data,
    vecmem::data::jagged_vector_view<detray::intersection2D<
        typename detector_t::surface_type, typename detector_t::transform3>>
        nav_candidates_buffer,
    const int n_params,
    bound_track_parameters_collection_types::view params_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/finding/device/impl/apply_interaction.ipp"