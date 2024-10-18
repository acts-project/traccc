/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

TRACCC_HOST_DEVICE inline void fill_sort_keys(
    std::size_t globalIndex,
    bound_track_parameters_collection_types::const_view params_view,
    vecmem::data::vector_view<device::sort_key> keys_view,
    vecmem::data::vector_view<unsigned int> ids_view) {

    bound_track_parameters_collection_types::const_device params(params_view);

    // Keys
    vecmem::device_vector<device::sort_key> keys_device(keys_view);

    // Param id
    vecmem::device_vector<unsigned int> ids_device(ids_view);

    if (globalIndex >= keys_device.size()) {
        return;
    }

    keys_device.at(static_cast<unsigned int>(globalIndex)) =
        device::get_sort_key(params.at(static_cast<unsigned int>(globalIndex)));
    ids_device.at(static_cast<unsigned int>(globalIndex)) =
        static_cast<unsigned int>(globalIndex);
}

}  // namespace traccc::device
