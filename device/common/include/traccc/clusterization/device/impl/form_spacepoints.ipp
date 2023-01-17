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
    const std::size_t globalIndex,
    alt_measurement_collection_types::const_view measurements_view,
    cell_module_collection_types::const_view modules_view,
    const unsigned int measurement_count,
    spacepoint_collection_types::view spacepoints_view) {

    // Get device copy of input parameters
    const alt_measurement_collection_types::const_device measurements_device(
        measurements_view);

    // Check if anything needs to be done
    if (globalIndex >= measurement_count) {
        return;
    }

    // Get device copy of input parameters
    const cell_module_collection_types::const_device modules_device(
        modules_view);

    spacepoint_collection_types::device spacepoints_device(spacepoints_view);

    // Get the measurement for this index
    const alt_measurement& meas = measurements_device.at(globalIndex);
    // Get the current cell module
    const cell_module& mod = modules_device.at(meas.module_link);
    // Form a spacepoint based on this measurement
    point3 local_3d = {meas.local[0], meas.local[1], 0.};
    point3 global = mod.placement.point_to_global(local_3d);

    // Fill the result object with this spacepoint
    spacepoints_device[globalIndex] = {global, meas};
}

}  // namespace traccc::device