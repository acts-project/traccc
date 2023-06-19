/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/seed.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/edm/track_parameters.hpp"

namespace traccc::device {

/// Function used for calculating the bound track parameters for each seed
///
/// @param[in] globalIndex      The index of the current thread
/// @param[in] spacepoints_view Collection storing the spacepoints
/// @param[in] seeds_view       Collection storing the seeds
/// @param[out] params_view     Collection storing the bound track parameters
///
TRACCC_HOST_DEVICE
inline void estimate_track_params(
    const std::size_t globalIndex,
    const spacepoint_collection_types::const_view& spacepoints_view,
    const seed_collection_types::const_view& seeds_view, const vector3& bfield,
    bound_track_parameters_collection_types::view params_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/seeding/device/impl/estimate_track_params.ipp"
