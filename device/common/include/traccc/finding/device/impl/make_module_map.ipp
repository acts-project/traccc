/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s)
#include <climits>

namespace traccc::device {

TRACCC_DEVICE inline void make_module_map(
    std::size_t globalIndex,
    measurement_container_types::const_view measurements_view,
    vecmem::data::vector_view<thrust::pair<geometry_id, unsigned int>>
        module_map_view) {

    measurement_container_types::const_device measurements(measurements_view);

    const unsigned int meas_size = measurements.size();

    if (globalIndex >= meas_size) {
        return;
    }

    vecmem::device_vector<thrust::pair<geometry_id, unsigned int>> module_map(
        module_map_view);

    module_map.at(globalIndex) = thrust::make_pair(
        measurements.at(globalIndex).header.module, globalIndex);
}

}  // namespace traccc::device
