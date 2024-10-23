/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/seeding/detail/spacepoint_formation.hpp"

// System include(s).
#include <cassert>

namespace traccc::device {

template <typename detector_t>
TRACCC_HOST_DEVICE inline void form_spacepoints(
    std::size_t globalIndex, typename detector_t::view_type det_view,
    const measurement_collection_types::const_view& measurements_view,
    unsigned int measurement_count,
    spacepoint_collection_types::view spacepoints_view) {

    // Check if anything needs to be done
    if (globalIndex >= measurement_count) {
        return;
    }

    // Create the tracking geometry
    detector_t det(det_view);

    // Get device copy of input/output parameters
    const measurement_collection_types::const_device measurements(
        measurements_view);
    assert(measurements.size() == measurement_count);
    spacepoint_collection_types::device spacepoints(spacepoints_view);

    const auto& meas = measurements.at(static_cast<unsigned int>(globalIndex));

    // Fill the spacepoint using the common function.
    if (details::is_valid_measurement(meas)) {
        spacepoints.push_back(details::create_spacepoint(det, meas));
    }
}

}  // namespace traccc::device
