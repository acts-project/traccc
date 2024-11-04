/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"

namespace traccc::details {

/// Function helping with checking a measurement obejct for spacepoint creation
///
/// @param[in]  measurement The input measurement
TRACCC_HOST_DEVICE inline bool is_valid_measurement(const measurement& meas);

/// Function helping with filling/setting up a spacepoint object
///
/// @param[in]  det         The tracking geometry
/// @param[in]  measurement The measurement to create the spacepoint out of
///
template <typename detector_t>
TRACCC_HOST_DEVICE inline spacepoint create_spacepoint(const detector_t& det,
                                                       const measurement& meas);

}  // namespace traccc::details

// Include the implementation.
#include "traccc/seeding/impl/spacepoint_formation.ipp"
