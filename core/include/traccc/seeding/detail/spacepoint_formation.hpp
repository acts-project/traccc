/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint_collection.hpp"

namespace traccc::details {

/// Function helping with checking a measurement obejct for spacepoint creation
///
/// @param[in]  measurement The input measurement
TRACCC_HOST_DEVICE inline bool is_valid_measurement(const measurement& meas);

/// Fill a spacepoint object with the information from a measurement
///
/// @param[out] sp          The spacepoint to fill
/// @param[in]  det         The tracking geometry
/// @param[in]  measurement The measurement to create the spacepoint out of
/// @param[in]  gctx        The current geometry context
///
template <typename soa_t, typename detector_t>
TRACCC_HOST_DEVICE inline void fill_pixel_spacepoint(
    edm::spacepoint<soa_t>& sp, const detector_t& det, const measurement& meas,
    const typename detector_t::geometry_context gctx = {});

}  // namespace traccc::details

// Include the implementation.
#include "traccc/seeding/impl/spacepoint_formation.ipp"
