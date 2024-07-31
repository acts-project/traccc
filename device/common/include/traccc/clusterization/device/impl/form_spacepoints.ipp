/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/clusterization/details/spacepoint_formation.hpp"

// System include(s).
#include <cassert>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void form_spacepoints(
    std::size_t globalIndex,
    const measurement_collection_types::const_view& measurements_view,
    const detector_description::const_view& det_descr_view,
    unsigned int measurement_count,
    spacepoint_collection_types::view spacepoints_view) {

    // Check if anything needs to be done
    if (globalIndex >= measurement_count) {
        return;
    }

    // Get device copy of input/output parameters
    const measurement_collection_types::const_device measurements(
        measurements_view);
    assert(measurements.size() == measurement_count);
    const detector_description::const_device det_descr(det_descr_view);
    spacepoint_collection_types::device spacepoints(spacepoints_view);
    assert(spacepoints.size() == measurement_count);

    // Fill the spacepoint using the common function.
    details::fill_spacepoint(spacepoints.at(globalIndex),
                             measurements.at(globalIndex), det_descr);
}

}  // namespace traccc::device
