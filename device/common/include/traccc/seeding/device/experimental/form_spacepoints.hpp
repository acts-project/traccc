/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"

namespace traccc::device::experimental {

/// Function for creating 3D spacepoints out of 2D measurements
///
template <typename detector_t>
TRACCC_HOST_DEVICE inline void form_spacepoints(
    const std::size_t globalIndex,
    typename detector_t::detector_view_type det_data,
    measurement_collection_types::const_view measurements_view,
    spacepoint_collection_types::view spacepoints_view);

}  // namespace traccc::device::experimental

// Include the implementation.
#include "traccc/seeding/device/experimental/impl/form_spacepoints.ipp"