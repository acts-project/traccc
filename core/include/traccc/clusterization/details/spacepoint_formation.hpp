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
#include "traccc/geometry/silicon_detector_description.hpp"

namespace traccc::details {

/// Function helping with filling/setting up a spacepoint object
///
/// @param[out] sp          The spacepoint to fill / set up
/// @param[in]  measurement The measurement to create the spacepoint out of
/// @param[in]  dd          The detector description
///
TRACCC_HOST_DEVICE inline void fill_spacepoint(
    spacepoint& sp, const measurement& meas,
    const silicon_detector_description::const_device& dd);

}  // namespace traccc::details

// Include the implementation.
#include "traccc/clusterization/impl/spacepoint_formation.ipp"
