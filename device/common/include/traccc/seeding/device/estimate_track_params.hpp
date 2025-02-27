/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/global_index.hpp"

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/seed_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/edm/track_parameters.hpp"

namespace traccc::device {

/// Function used for calculating the bound track parameters for each seed
///
/// @param[in] globalIndex      The index of the current thread
/// @param[in] measurements_view All measurements of the event
/// @param[in] spacepoints_view Collection storing the spacepoints
/// @param[in] seeds_view       Collection storing the seeds
/// @param[in] bfield           B field
/// @param[in] stddev           Standard deviation of seed parameters
/// @param[out] params_view     Collection storing the bound track parameters
///
TRACCC_HOST_DEVICE
inline void estimate_track_params(
    global_index_t globalIndex,
    const measurement_collection_types::const_view& measurements_view,
    const edm::spacepoint_collection::const_view& spacepoints_view,
    const edm::seed_collection::const_view& seeds_view, const vector3& bfield,
    const std::array<traccc::scalar, traccc::e_bound_size>& stddev,
    bound_track_parameters_collection_types::view params_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/seeding/device/impl/estimate_track_params.ipp"
