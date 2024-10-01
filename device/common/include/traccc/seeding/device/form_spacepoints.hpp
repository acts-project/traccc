/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"

// System include(s).
#include <cstddef>

namespace traccc::device {

/// Function for creating 3D spacepoints out of 2D measurements
///
/// @param[in] globalIndex          The index for the current thread
/// @param[in] det_view             A view type object of detector
/// @param[in] measurements_view    Collection of measurements
/// @param[in] measurement_count    Number of measurements
/// @param[out] spacepoints_view    Collection of spacepoints
///
template <typename detector_t>
TRACCC_HOST_DEVICE inline void form_spacepoints(
    std::size_t globalIndex, typename detector_t::view_type det_view,
    const measurement_collection_types::const_view& measurements_view,
    unsigned int measurement_count,
    spacepoint_collection_types::view spacepoints_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/seeding/device/impl/form_spacepoints.ipp"
