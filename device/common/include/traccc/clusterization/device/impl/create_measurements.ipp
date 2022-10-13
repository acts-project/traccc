/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <cassert>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void create_measurements(
    std::size_t globalIndex, cluster_container_types::const_view clusters_view,
    const cell_container_types::const_view& cells_view,
    measurement_container_types::view measurements_view) {

    // Initialize device vector that gives us the execution range
    const cluster_container_types::const_device clusters_device(clusters_view);

    // Ignore if idx is out of range
    if (globalIndex >= clusters_device.size())
        return;

    // Create other device containers
    measurement_container_types::device measurements_device(measurements_view);
    cell_container_types::const_device cells_device(cells_view);

    // items: cluster of cells at current idx
    // header: module idx
    const auto& cluster = clusters_device[globalIndex].items;
    const auto& module_link = clusters_device[globalIndex].header;
    const auto& module = cells_device.at(module_link).header;

    // Should not happen
    assert(cluster.empty() == false);

    // Fill measurement from cluster
    detail::fill_measurement(measurements_device, cluster, module, module_link,
                             globalIndex);
}

}  // namespace traccc::device