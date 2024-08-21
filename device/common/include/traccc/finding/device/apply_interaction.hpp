/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/finding/finding_config.hpp"

namespace traccc::device {

/// Function applying the Pre material interaction to tracks spawned by bound
/// track parameters
///
/// @param[in] globalIndex     The index of the current thread
/// @param[in] cfg             Track finding config object
/// @param[in] det_data        Detector view object
/// @param[in] n_params        The number of parameters (or tracks)
/// @param[out] params_view    Collection of output bound track_parameters
///
template <typename detector_t>
TRACCC_DEVICE inline void apply_interaction(
    std::size_t globalIndex, const finding_config& cfg,
    typename detector_t::view_type det_data, const int n_params,
    bound_track_parameters_collection_types::view params_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/finding/device/impl/apply_interaction.ipp"
