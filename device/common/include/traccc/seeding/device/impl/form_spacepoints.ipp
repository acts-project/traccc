/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
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
    const global_index_t globalIndex, typename detector_t::view_type det_view,
    const measurement_collection_types::const_view& measurements_view,
    edm::spacepoint_collection::view spacepoints_view) {

    // Set up the input container(s).
    const measurement_collection_types::const_device measurements(
        measurements_view);

    // Check if anything needs to be done
    if (globalIndex >= measurements.size()) {
        return;
    }

    // Create the tracking geometry
    detector_t det(det_view);

    // Set up the output container(s).
    edm::spacepoint_collection::device spacepoints(spacepoints_view);

    const measurement& meas = measurements.at(globalIndex);

    // Fill the spacepoint using the common function.
    if (details::is_valid_measurement(meas)) {
        const edm::spacepoint_collection::device::size_type i =
            spacepoints.push_back_default();
        edm::spacepoint_collection::device::proxy_type sp = spacepoints.at(i);
        traccc::details::fill_pixel_spacepoint(sp, det, meas);
        sp.measurement_index() = globalIndex;
    }
}

}  // namespace traccc::device
