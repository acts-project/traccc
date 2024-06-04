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
    const std::size_t globalIndex,
    measurement_collection_types::const_view measurements_view,
    cell_module_collection_types::const_view modules_view,
    const unsigned int measurement_count,
    spacepoint_collection_types::view spacepoints_view) {

    // Check if anything needs to be done
    if (globalIndex >= measurement_count) {
        return;
    }

    // Get device copy of input/output parameters
    const measurement_collection_types::const_device measurements(
        measurements_view);
    assert(measurements.size() == measurement_count);
    const cell_module_collection_types::const_device modules(modules_view);
    spacepoint_collection_types::device spacepoints(spacepoints_view);
    assert(spacepoints.size() == measurement_count);

    // Get the measurement for this index
    const measurement& measurement = measurements.at(globalIndex);
    // Get the current cell module
    const cell_module& mod = modules.at(measurement.module_link);

    // Fill the spacepoint using the common function.
    details::fill_spacepoint(spacepoints.at(globalIndex), measurement, mod);
}

}  // namespace traccc::device
