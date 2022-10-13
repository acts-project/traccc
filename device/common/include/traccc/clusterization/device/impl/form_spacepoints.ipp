/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void form_spacepoints(
    std::size_t globalIndex,
    measurement_container_types::const_view measurements_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        measurements_prefix_sum_view,
    spacepoint_container_types::view spacepoints_view) {

    // Initialize device container for for the prefix sum
    vecmem::device_vector<const device::prefix_sum_element_t>
        measurements_prefix_sum(measurements_prefix_sum_view);

    // Ignore if idx is out of range
    if (globalIndex >= measurements_prefix_sum.size())
        return;

    // Initialize the rest of the device containers
    measurement_container_types::const_device measurements_device(
        measurements_view);
    spacepoint_container_types::device spacepoints_device(spacepoints_view);

    // Get the indices from the prefix sum vector
    const auto module_idx = measurements_prefix_sum[globalIndex].first;
    const auto measurement_idx = measurements_prefix_sum[globalIndex].second;

    // Get the measurement for this idx
    const auto& m = measurements_device[module_idx].items.at(measurement_idx);

    // Get the current cell module
    const auto& module = measurements_device[module_idx].header;

    // Form a spacepoint based on this measurement
    point3 local_3d = {m.local[0], m.local[1], 0.};
    point3 global = module.placement.point_to_global(local_3d);
    spacepoint s({global, m});

    // Push the speacpoint into the container at the appropriate
    // module idx
    spacepoints_device[module_idx].header = module.module;
    spacepoints_device[module_idx].items.push_back(s);
}

}  // namespace traccc::device